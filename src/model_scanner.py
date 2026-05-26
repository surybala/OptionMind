"""ML scanner hook for OptionMind.

This is the only scanner entry point the app should use going forward.
The inherited deterministic scanner lives in ``src/scanner.py`` as legacy
reference code, but it should not be used as a fallback for live candidate
generation.
"""
from __future__ import annotations

import importlib
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from ml.datasets.candidate_dataset import (
    _dte,
    _dividend_features,
    _event_features,
    _macro_event_features,
    _market_regime_features,
    _moneyness,
    _option_greeks_features,
    _range_pct,
    _strike_distance_pct,
    _underlying_features,
    _vix_features,
)
from ml.models.registry import ModelRegistryEntry, load_champion_artifact
from ml.providers.calendar import fomc_events
from ml.providers.models import DividendEvent, EarningsEvent, EconomicEvent, OptionChainSnapshot, OptionContract, PriceBar
from src.pick_selection import select_top_picks_with_scanner_controls

_log = logging.getLogger("optionwheel")


class ModelScanner:
    """Bridge between the application and a future trained model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scanner_config = config.get("ml_scanner", {})
        self.model_provider = self._load_provider() if self.scanner_config.get("enabled", False) else None

    def get_top_picks(self, ticker_list: list[str], n: int = 10) -> list[dict]:
        """Return model-ranked candidates, or no picks until a model exists."""
        if not self.scanner_config.get("enabled", False):
            _log.warning(
                "ML scanner is disabled or no model is configured. "
                "No deterministic scanner fallback will run."
            )
            return []

        if self.model_provider is None:
            _log.warning(
                "ML scanner is enabled but no provider was loaded. "
                "Set ml_scanner.provider to 'module:callable'."
            )
            return []

        candidates = self._call_provider(ticker_list, n)
        normalized = [self._normalize_candidate(candidate) for candidate in candidates]
        normalized = [candidate for candidate in normalized if candidate is not None]
        return select_top_picks_with_scanner_controls(normalized, n=n, config=self.config)

    def _load_provider(self) -> Any | None:
        provider_path = self.scanner_config.get("provider")
        if not provider_path:
            return None
        if ":" not in provider_path:
            raise ValueError("ml_scanner.provider must use 'module:callable' format")

        module_name, attr_name = provider_path.split(":", 1)
        module = importlib.import_module(module_name)
        provider = getattr(module, attr_name)
        if isinstance(provider, type):
            return provider(self.config)
        return provider

    def _call_provider(self, ticker_list: list[str], n: int) -> list[dict]:
        provider = self.model_provider
        if hasattr(provider, "get_top_picks"):
            return list(provider.get_top_picks(ticker_list, n=n))
        if callable(provider):
            return list(provider(ticker_list=ticker_list, n=n, config=self.config))
        raise TypeError("ML scanner provider must be callable or expose get_top_picks()")

    def _normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        item = dict(candidate)
        item.setdefault("source", "ml_model")
        item.setdefault("score", item.get("model_score", 0.0))
        item.setdefault("prob_win", item.get("probability_of_profit", 0.0))
        item.setdefault("quantity", 1)
        item.setdefault(
            "mispricing_score_basis",
            "ML model score: expected utility / P&L-centered inference",
        )

        min_score = float(self.scanner_config.get("min_score", float("-inf")))
        try:
            score = float(item.get("model_score", item.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            return None
        return item


OptionScanner = ModelScanner


@dataclass(frozen=True)
class _ScoredOption:
    row: dict[str, Any]
    snapshot: OptionChainSnapshot
    contract: OptionContract
    score: float
    option_price: float
    bid: float | None
    ask: float | None


class LivePaperInferenceProvider:
    """First live/paper inference provider for the app-facing model scanner.

    It loads the registry champion, builds current short-option candidates from
    normalized provider data, computes the same feature columns used in training,
    scores each short leg, and returns executable credit-spread pick dicts.
    """

    def __init__(self, config: dict[str, Any], provider: Any | None = None, now_fn: Callable[[], datetime] | None = None):
        self.config = config
        self.scanner_config = config.get("ml_scanner", {})
        self.provider = provider or self._load_data_provider()
        self.now_fn = now_fn or (lambda: datetime.now(UTC))
        self.registry_entry: ModelRegistryEntry | None = None
        self.model: _ChampionModel | None = None
        self._load_champion()

    def get_top_picks(self, ticker_list: list[str], n: int = 10) -> list[dict]:
        if self.model is None:
            _log.warning("ML scanner inference provider has no champion model loaded.")
            return []
        if self.provider is None:
            _log.warning("ML scanner inference provider has no market-data provider.")
            return []

        now = _ensure_utc(self.now_fn())
        min_dte = int(self.scanner_config.get("min_dte", 7))
        max_dte = int(self.scanner_config.get("max_dte", self.config.get("expiry_days_max", 45) or 45))
        stock_lookback_days = int(self.scanner_config.get("stock_lookback_days", 80))
        market_symbol = str(self.scanner_config.get("market_regime_symbol", "SPY")).upper()
        vix_symbol = str(self.scanner_config.get("vix_symbol", "I:VIX")).upper()
        forward_days = int(self.scanner_config.get("forward_days", 30))

        underlyings = [symbol.strip().upper() for symbol in ticker_list if symbol and symbol.strip()]
        if not underlyings:
            return []

        stock_symbols = _unique([*underlyings, market_symbol, vix_symbol])
        stock_bars = self.provider.get_stock_bars(
            stock_symbols,
            now - timedelta(days=stock_lookback_days),
            now,
            str(self.scanner_config.get("stock_timeframe", "1Day")),
        )
        market_history = stock_bars.get(market_symbol, [])
        vix_bars = stock_bars.get(vix_symbol, [])
        macro_events = self._macro_events(now.date(), now.date() + timedelta(days=90))

        picks: list[dict] = []
        expiration_gte = now.date() + timedelta(days=min_dte)
        expiration_lte = now.date() + timedelta(days=max_dte)
        chain_limit = self.scanner_config.get("chain_limit")
        for underlying in underlyings:
            underlying_history = stock_bars.get(underlying, [])
            if not underlying_history:
                continue
            chain = self.provider.get_current_option_chain(
                underlying,
                expiration_gte=expiration_gte,
                expiration_lte=expiration_lte,
                limit=int(chain_limit) if chain_limit else None,
            )
            if not chain:
                continue
            earnings = self._earnings_events(underlying, now.date(), now.date() + timedelta(days=forward_days))
            dividends = self._dividend_events(underlying, now.date(), now.date() + timedelta(days=forward_days))
            scored = self._score_chain(
                underlying=underlying,
                chain=chain,
                timestamp=now,
                underlying_history=underlying_history,
                market_history=market_history,
                market_symbol=market_symbol,
                vix_bars=vix_bars,
                earnings_events=earnings,
                dividend_events=dividends,
                macro_events=macro_events,
                min_dte=min_dte,
                max_dte=max_dte,
                forward_days=forward_days,
            )
            picks.extend(self._spread_picks(underlying, scored, now))

        return select_top_picks_with_scanner_controls(picks, n=n, config=self.config)

    def _load_champion(self) -> None:
        registry_path = self.scanner_config.get("registry_path", "artifacts/model_registry.json")
        try:
            entry, artifact = load_champion_artifact(registry_path)
        except Exception as exc:
            artifact_path = self.scanner_config.get("artifact_path")
            if not artifact_path:
                _log.warning("Unable to load ML champion from registry %s: %s", registry_path, exc)
                return
            artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            entry = None
        self.registry_entry = entry
        if entry is not None and entry.artifact_manifest.model_path:
            artifact = dict(artifact)
            artifact["model_path"] = entry.artifact_manifest.model_path
        self.model = _ChampionModel(artifact)

    def _load_data_provider(self) -> Any:
        provider_path = self.scanner_config.get("data_provider")
        if provider_path:
            if ":" not in provider_path:
                raise ValueError("ml_scanner.data_provider must use 'module:callable' format")
            module_name, attr_name = provider_path.split(":", 1)
            module = importlib.import_module(module_name)
            provider = getattr(module, attr_name)
            return provider(self.config) if isinstance(provider, type) else provider

        from ml.providers import AlpacaProvider

        paper = bool(self.config.get("alpaca", {}).get("paper", True))
        return AlpacaProvider.from_env(paper=paper)

    def _score_chain(
        self,
        *,
        underlying: str,
        chain: dict[str, OptionChainSnapshot],
        timestamp: datetime,
        underlying_history: list[PriceBar],
        market_history: list[PriceBar],
        market_symbol: str,
        vix_bars: list[PriceBar],
        earnings_events: list[EarningsEvent],
        dividend_events: list[DividendEvent],
        macro_events: list[EconomicEvent],
        min_dte: int,
        max_dte: int,
        forward_days: int,
    ) -> list[_ScoredOption]:
        rows: list[tuple[dict[str, Any], OptionChainSnapshot, OptionContract, float, float | None, float | None]] = []
        for symbol, snapshot in chain.items():
            contract = _contract_from_snapshot(symbol, underlying, snapshot)
            dte = _dte(timestamp, contract.expiration)
            if dte is None or dte < min_dte or dte > max_dte:
                continue
            option_price = _feature_option_price(snapshot)
            if option_price is None or option_price <= 0:
                continue
            row = _live_feature_row(
                contract=contract,
                snapshot=snapshot,
                entry_timestamp=timestamp,
                option_price=option_price,
                underlying_history=underlying_history,
                market_history=market_history,
                market_symbol=market_symbol,
                vix_bars=vix_bars,
                earnings_events=earnings_events,
                dividend_events=dividend_events,
                macro_events=macro_events,
                forward_days=forward_days,
                risk_free_rate=float(self.scanner_config.get("risk_free_rate", 0.045)),
            )
            rows.append((row, snapshot, contract, option_price, snapshot.bid, snapshot.ask))
        if not rows or self.model is None:
            return []
        scores = self.model.score_rows([row for row, *_ in rows])
        return [
            _ScoredOption(row=row, snapshot=snapshot, contract=contract, score=score, option_price=price, bid=bid, ask=ask)
            for (row, snapshot, contract, price, bid, ask), score in zip(rows, scores)
        ]

    def _spread_picks(self, underlying: str, scored: list[_ScoredOption], timestamp: datetime) -> list[dict]:
        by_key = {
            (item.contract.expiration, item.contract.option_type, _money(item.contract.strike)): item
            for item in scored
            if item.contract.expiration and item.contract.option_type and item.contract.strike is not None
        }
        spot = _latest_close([item.row for item in scored], underlying)
        if spot is None:
            return []

        picks: list[dict] = []
        pcs_cfg = self.config.get("strategies", {}).get("put_credit_spread", {})
        ccs_cfg = self.config.get("strategies", {}).get("call_credit_spread", {})
        if pcs_cfg.get("enabled", True):
            picks.extend(self._spread_side_picks(underlying, scored, by_key, spot, timestamp, "put", pcs_cfg))
        if ccs_cfg.get("enabled", True):
            picks.extend(self._spread_side_picks(underlying, scored, by_key, spot, timestamp, "call", ccs_cfg))
        return picks

    def _spread_side_picks(
        self,
        underlying: str,
        scored: list[_ScoredOption],
        by_key: dict[tuple[date | None, str | None, float | None], _ScoredOption],
        spot: float,
        timestamp: datetime,
        option_type: str,
        strategy_config: dict[str, Any],
    ) -> list[dict]:
        strategy = "PCS" if option_type == "put" else "CCS"
        widths = _spread_widths(strategy_config, self.scanner_config)
        min_credit = float(strategy_config.get("min_net_credit", self.scanner_config.get("min_net_credit", 0.01)) or 0.01)
        min_prob = float(strategy_config.get("min_prob_profit", 0.0) or 0.0)
        picks: list[dict] = []
        for short in scored:
            if short.contract.option_type != option_type or short.contract.strike is None or short.contract.expiration is None:
                continue
            if option_type == "put" and short.contract.strike >= spot:
                continue
            if option_type == "call" and short.contract.strike <= spot:
                continue
            for width in widths:
                long_strike = short.contract.strike - width if option_type == "put" else short.contract.strike + width
                long = by_key.get((short.contract.expiration, option_type, _money(long_strike)))
                if long is None or short.bid is None or long.ask is None:
                    continue
                credit = round(short.bid - long.ask, 4)
                if credit < min_credit:
                    continue
                prob_win = _probability_from_delta(short.row.get("option_delta"))
                if prob_win < min_prob:
                    continue
                actual_width = abs(float(short.contract.strike) - float(long.contract.strike))
                max_loss = max(0.01, actual_width - credit)
                dte = int(short.row.get("dte") or max(1, (short.contract.expiration - timestamp.date()).days))
                roi = credit / max_loss
                pick = {
                    "strategy": strategy,
                    "symbol": underlying,
                    "expiry": short.contract.expiration.isoformat(),
                    "current_price": round(float(spot), 2),
                    "short_strike": _money(short.contract.strike),
                    "long_strike": _money(long.contract.strike),
                    "width": round(actual_width, 4),
                    "premium": credit,
                    "max_loss": round(max_loss, 4),
                    "prob_win": round(prob_win, 4),
                    "roi": round(roi, 4),
                    "annualized_roi": round(roi * (365 / max(1, dte)), 4),
                    "model_score": round(float(short.score), 6),
                    "score": round(float(short.score), 6),
                    "mispricing_score": round(float(short.score), 6),
                    "mispricing_score_basis": "Champion ML expected-P&L score on the short leg; deterministic risk gates still apply.",
                    "source": "ml_model",
                    "quantity": 1,
                    "short_option_symbol": short.contract.symbol,
                    "long_option_symbol": long.contract.symbol,
                    "feature_version": self.model.feature_version if self.model else None,
                    "label_version": self.model.label_version if self.model else None,
                    "model_type": self.model.model_type if self.model else None,
                    "model_version": self.registry_entry.model_id if self.registry_entry else self.model.model_type if self.model else None,
                    "model_artifact_path": (
                        self.registry_entry.artifact_manifest.artifact_path if self.registry_entry else self.scanner_config.get("artifact_path")
                    ),
                    "model_id": self.registry_entry.model_id if self.registry_entry else None,
                    "features_hash": _features_hash(short.row, self.model.feature_columns if self.model else []),
                    "features": _feature_subset(short.row, self.model.feature_columns if self.model else []),
                    "score_components": {
                        "short_leg_score": round(float(short.score), 6),
                        "short_option_price": short.option_price,
                        "short_bid": short.bid,
                        "long_ask": long.ask,
                        "net_credit": credit,
                        "prob_win_from_delta": round(prob_win, 4),
                    },
                }
                if option_type == "put":
                    pick["short_put"] = pick["short_strike"]
                    pick["long_put"] = pick["long_strike"]
                else:
                    pick["short_call"] = pick["short_strike"]
                    pick["long_call"] = pick["long_strike"]
                picks.append(pick)
        return picks

    def _earnings_events(self, underlying: str, start: date, end: date) -> list[EarningsEvent]:
        if hasattr(self.provider, "get_earnings_calendar"):
            return self.provider.get_earnings_calendar([underlying], start, end).get(underlying.upper(), [])
        return []

    def _dividend_events(self, underlying: str, start: date, end: date) -> list[DividendEvent]:
        if hasattr(self.provider, "get_dividends"):
            return self.provider.get_dividends([underlying], start, end).get(underlying.upper(), [])
        return []

    def _macro_events(self, start: date, end: date) -> list[EconomicEvent]:
        events = list(fomc_events(start, end))
        if hasattr(self.provider, "get_economic_calendar"):
            events.extend(self.provider.get_economic_calendar(start, end))
        return sorted({(event.event_name, event.event_date): event for event in events}.values(), key=lambda event: event.event_date)


class _ChampionModel:
    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.model_type = str(artifact.get("model_type", "unknown"))
        self.feature_columns = list(artifact.get("feature_columns") or [])
        self.fill_values = {key: float(value) for key, value in dict(artifact.get("fill_values") or {}).items()}
        self.feature_version = artifact.get("feature_version")
        self.label_version = artifact.get("label_version")
        self._booster = None
        if self.model_type.startswith("xgboost"):
            self._booster = self._load_xgboost_booster(artifact)

    def score_rows(self, rows: list[dict[str, Any]]) -> list[float]:
        if not rows:
            return []
        if self.model_type.startswith("linear"):
            return [self._score_linear(row) for row in rows]
        if self.model_type.startswith("xgboost"):
            return self._score_xgboost(rows)
        raise ValueError(f"Unsupported champion model type: {self.model_type}")

    def _score_linear(self, row: dict[str, Any]) -> float:
        coefficients = dict(self.artifact.get("coefficients") or {})
        score = float(self.artifact.get("intercept") or 0.0)
        for column in self.feature_columns:
            score += float(coefficients.get(column, 0.0)) * self._value(row, column)
        return float(score)

    def _score_xgboost(self, rows: list[dict[str, Any]]) -> list[float]:
        try:
            import pandas as pd
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover - depends on optional native runtime
            raise ImportError("xgboost and pandas are required to score xgboost champion artifacts") from exc
        frame = pd.DataFrame(
            [{column: self._value(row, column) for column in self.feature_columns} for row in rows],
            columns=self.feature_columns,
        )
        matrix = xgb.DMatrix(frame, feature_names=self.feature_columns)
        return [float(value) for value in self._booster.predict(matrix)]

    def _value(self, row: dict[str, Any], column: str) -> float:
        value = row.get(column)
        try:
            if value is None:
                raise TypeError
            return float(value)
        except (TypeError, ValueError):
            return float(self.fill_values.get(column, 0.0))

    def _load_xgboost_booster(self, artifact: dict[str, Any]):
        try:
            import xgboost as xgb
        except Exception as exc:  # pragma: no cover - depends on optional native runtime
            raise ImportError("xgboost is required to load an xgboost champion artifact") from exc
        model_path = artifact.get("model_path")
        if not model_path:
            raise ValueError("XGBoost artifact is missing model_path")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        return booster


def _live_feature_row(
    *,
    contract: OptionContract,
    snapshot: OptionChainSnapshot,
    entry_timestamp: datetime,
    option_price: float,
    underlying_history: list[PriceBar],
    market_history: list[PriceBar],
    market_symbol: str,
    vix_bars: list[PriceBar],
    earnings_events: list[EarningsEvent],
    dividend_events: list[DividendEvent],
    macro_events: list[EconomicEvent],
    forward_days: int,
    risk_free_rate: float,
) -> dict[str, Any]:
    entry_bar = PriceBar(
        symbol=contract.symbol,
        timestamp=entry_timestamp,
        open=option_price,
        high=option_price,
        low=option_price,
        close=option_price,
        volume=None,
        trade_count=None,
        vwap=option_price,
        source=snapshot.source,
    )
    dte = _dte(entry_timestamp, contract.expiration)
    underlying_features = _underlying_features(underlying_history, entry_timestamp)
    market_features = _market_regime_features(market_history, entry_timestamp, market_symbol)
    greeks_features = _option_greeks_features(
        entry_bar,
        underlying_features["underlying_close"],
        contract.strike,
        contract.option_type,
        dte,
        risk_free_rate,
        underlying_features["underlying_realized_vol_5d"],
        underlying_features["underlying_realized_vol_20d"],
    )
    vix_features = _vix_features(vix_bars, entry_timestamp)
    event_features = _event_features(earnings_events, entry_timestamp, forward_days)
    dividend_features = _dividend_features(dividend_events, entry_timestamp, forward_days)
    fomc_only = [event for event in macro_events if "fomc" in event.event_name.lower()]
    macro_features = _macro_event_features(fomc_only, macro_events, entry_timestamp, forward_days)
    return {
        "entry_timestamp": entry_timestamp,
        "underlying": contract.underlying,
        "option_symbol": contract.symbol,
        "option_type": contract.option_type,
        "strike": contract.strike,
        "expiration": contract.expiration,
        "dte": dte,
        "source": snapshot.source,
        **underlying_features,
        "strike_distance_pct": _strike_distance_pct(contract.strike, underlying_features["underlying_close"]),
        "moneyness": _moneyness(contract.strike, underlying_features["underlying_close"]),
        **market_features,
        "option_entry_open": option_price,
        "option_entry_high": option_price,
        "option_entry_low": option_price,
        "option_entry_price": option_price,
        "option_entry_range_pct": _range_pct(option_price, option_price, option_price),
        "option_entry_volume": None,
        "option_entry_trade_count": None,
        "option_entry_vwap": option_price,
        **greeks_features,
        "option_volume_5d_avg": None,
        "option_trade_count_5d_avg": None,
        **vix_features,
        **event_features,
        **dividend_features,
        **macro_features,
    }


def _contract_from_snapshot(symbol: str, underlying: str, snapshot: OptionChainSnapshot) -> OptionContract:
    parsed = _parse_osi(symbol)
    raw = snapshot.raw or {}
    return OptionContract(
        symbol=symbol,
        underlying=underlying.upper(),
        expiration=_date_or_none(raw.get("expiration_date") or raw.get("expiration") or (parsed or {}).get("expiration")),
        strike=_float_or_none(raw.get("strike_price") or raw.get("strike") or (parsed or {}).get("strike")),
        option_type=_option_type_or_none(raw.get("type") or raw.get("option_type") or (parsed or {}).get("option_type")),
        status=str(raw.get("status") or "active"),
        source=snapshot.source,
        raw=raw,
    )


def _parse_osi(symbol: str) -> dict[str, Any] | None:
    import re

    match = re.match(r"^([A-Z./]+)(\d{6})([CP])(\d{8})$", symbol)
    if not match:
        return None
    yy, mm, dd = int(match.group(2)[:2]), int(match.group(2)[2:4]), int(match.group(2)[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy
    return {
        "underlying": match.group(1),
        "expiration": date(year, mm, dd),
        "option_type": "call" if match.group(3) == "C" else "put",
        "strike": int(match.group(4)) / 1000.0,
    }


def _feature_option_price(snapshot: OptionChainSnapshot) -> float | None:
    if snapshot.bid is not None and snapshot.ask is not None and snapshot.bid > 0 and snapshot.ask > 0:
        return round((snapshot.bid + snapshot.ask) / 2.0, 8)
    return snapshot.last or snapshot.bid or snapshot.ask


def _probability_from_delta(delta: Any) -> float:
    try:
        value = abs(float(delta))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, 1.0 - value))


def _latest_close(rows: list[dict[str, Any]], underlying: str) -> float | None:
    for row in rows:
        if row.get("underlying") == underlying and row.get("underlying_close") is not None:
            return float(row["underlying_close"])
    return None


def _money(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _spread_widths(strategy_config: dict[str, Any], scanner_config: dict[str, Any]) -> list[float]:
    raw = strategy_config.get("spread_widths", scanner_config.get("spread_widths"))
    if raw is None:
        raw = [strategy_config.get("strike_width", scanner_config.get("strike_width", 5)) or 5]
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [raw]
    widths = sorted({round(float(value), 4) for value in values if float(value) > 0})
    return widths or [5.0]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date_or_none(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _option_type_or_none(value: Any) -> str | None:
    raw = str(getattr(value, "value", value) or "").lower()
    if raw in {"call", "c"}:
        return "call"
    if raw in {"put", "p"}:
        return "put"
    return None


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = value.upper()
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _feature_subset(row: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    if not feature_columns:
        return dict(row)
    return {column: row.get(column) for column in feature_columns}


def _features_hash(row: dict[str, Any], feature_columns: list[str]) -> str:
    payload = json.dumps(_feature_subset(row, feature_columns), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
