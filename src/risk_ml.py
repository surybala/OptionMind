from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ml.models.registry import ModelRegistryEntry, load_registry

_log = logging.getLogger("optionwheel")


class MlExitRiskService:
    """Loads and scores an optional ML early-exit model for open positions."""

    def __init__(self, config: dict):
        risk = config.get("risk_parameters", {})
        self.stop_loss_multiplier = float(risk.get("stop_loss_multiplier", 2.0))
        self.stop_loss_max_loss_pct = self._optional_float(risk.get("stop_loss_max_loss_pct"))
        self.profit_take_enabled = bool(risk.get("profit_take_enabled", True))
        self.profit_take_pct = float(risk.get("profit_take_pct", 0.75))
        self.config = dict(risk.get("ml_exit_risk", {}) or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.threshold = float(self.config.get("threshold", 0.70))
        self.confirmations_required = max(1, int(self.config.get("confirmations_required", 1)))
        self.min_dte = int(self.config.get("min_dte", 0))
        self.max_dte = self._optional_int(self.config.get("max_dte"))
        self.min_age_minutes = float(self.config.get("min_age_minutes", 0.0))
        self.reason_tag = str(self.config.get("reason_tag", "ML_RISK_EXIT") or "ML_RISK_EXIT")

        self.model = None
        self.registry_entry: ModelRegistryEntry | None = None
        self.artifact_path: str | None = None
        self.model_id: str | None = None
        self.model_type: str | None = None
        self.feature_version: str | None = None
        self.label_version: str | None = None

        if self.enabled:
            self._load_model()

    def is_active(self) -> bool:
        return bool(self.enabled and self.model is not None)

    def score_position(
        self,
        pos: dict,
        *,
        current_mark: float | None = None,
        spot: float | None = None,
        risk: dict[str, Any] | None = None,
        chain=None,
    ) -> dict[str, Any] | None:
        if not self.is_active():
            return None

        row = self._build_feature_row(
            pos,
            current_mark=current_mark,
            spot=spot,
            risk=risk,
            chain=chain,
        )
        guard_reason = self._guard_reason(row)
        score = float(self.model.score_rows([row])[0])
        should_trigger = guard_reason is None and score >= self.threshold
        return {
            "ml_exit_risk_score": round(score, 6),
            "ml_exit_risk_threshold": self.threshold,
            "ml_exit_risk_should_trigger": should_trigger,
            "ml_exit_risk_guard_reason": guard_reason,
            "ml_exit_risk_model_id": self.model_id,
            "ml_exit_risk_model_type": self.model_type,
            "ml_exit_risk_feature_version": self.feature_version,
            "ml_exit_risk_label_version": self.label_version,
            "ml_exit_risk_artifact_path": self.artifact_path,
        }

    def annotate_position(
        self,
        pos: dict,
        *,
        current_mark: float | None = None,
        spot: float | None = None,
        risk: dict[str, Any] | None = None,
        chain=None,
    ) -> dict:
        payload = self.score_position(
            pos,
            current_mark=current_mark,
            spot=spot,
            risk=risk,
            chain=chain,
        )
        if payload:
            pos.update(payload)
        return pos

    def _load_model(self) -> None:
        try:
            entry, artifact, artifact_path = self._load_artifact()
            if entry is not None and entry.artifact_manifest.model_path:
                artifact = dict(artifact)
                artifact["model_path"] = entry.artifact_manifest.model_path
            from src.model_scanner import _ChampionModel

            self.model = _ChampionModel(artifact)
            self.registry_entry = entry
            self.artifact_path = artifact_path
            self.model_id = entry.model_id if entry is not None else str(self.config.get("model_id") or "artifact")
            self.model_type = str(artifact.get("model_type") or "unknown")
            self.feature_version = artifact.get("feature_version")
            self.label_version = artifact.get("label_version")
            _log.info(
                "[ml_exit_risk] Loaded model %s from %s (threshold=%.2f, confirmations=%d)",
                self.model_id,
                self.artifact_path or self.model_type,
                self.threshold,
                self.confirmations_required,
            )
        except Exception as exc:
            self.model = None
            _log.warning("[ml_exit_risk] Failed to load exit-risk model: %s", exc)

    def _load_artifact(self) -> tuple[ModelRegistryEntry | None, dict[str, Any], str | None]:
        artifact_path = self.config.get("artifact_path")
        if artifact_path:
            resolved_artifact = self._resolve_path(str(artifact_path))
            artifact = json.loads(resolved_artifact.read_text(encoding="utf-8"))
            return None, artifact, str(resolved_artifact)

        registry_path = self.config.get("registry_path")
        if registry_path:
            registry_file = self._resolve_path(str(registry_path))
            registry = load_registry(str(registry_file))
            model_id = self.config.get("model_id")
            if model_id:
                entry = registry.get(str(model_id))
                if entry is None:
                    raise ValueError(f"No model found in registry {registry_path!r} for {model_id!r}")
                artifact_path = self._resolve_registry_artifact_path(
                    registry_file,
                    entry.artifact_manifest.artifact_path,
                )
                artifact = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
                return entry, artifact, str(artifact_path)

            entry = registry.champion
            if entry is None:
                raise ValueError(f"No champion model is configured in {registry_file}")
            artifact_path = self._resolve_registry_artifact_path(
                registry_file,
                entry.artifact_manifest.artifact_path,
            )
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            return entry, artifact, str(artifact_path)

        raise ValueError("ml_exit_risk requires artifact_path or registry_path")

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        return path if path.is_absolute() else Path.cwd() / path

    @staticmethod
    def _resolve_registry_artifact_path(registry_file: Path, artifact_path: str) -> Path:
        path = Path(artifact_path)
        if path.is_absolute():
            return path
        cwd_resolved = Path.cwd() / path
        if cwd_resolved.exists():
            return cwd_resolved
        return registry_file.parent / path

    def _build_feature_row(
        self,
        pos: dict,
        *,
        current_mark: float | None,
        spot: float | None,
        risk: dict[str, Any] | None,
        chain,
    ) -> dict[str, Any]:
        entry_premium = float(pos.get("entry_premium") or pos.get("premium") or 0.0)
        current_mark = self._optional_float(current_mark if current_mark is not None else pos.get("current_mark"))
        pnl_per_share = self._optional_float(
            pos.get("pnl_per_share") if pos.get("pnl_per_share") is not None else (
                entry_premium - current_mark if current_mark is not None else None
            )
        )
        loss_per_share = max(0.0, -(pnl_per_share or 0.0))
        profit_captured_pct = (
            (pnl_per_share / entry_premium) if pnl_per_share is not None and entry_premium > 0 else None
        )
        dte = self._dte(pos.get("expiry"))
        age_minutes = self._position_age_minutes(pos.get("timestamp"))
        contracts = int(pos.get("contracts") or 1)
        spot = self._optional_float(spot if spot is not None else pos.get("spot"))
        legs = self._parse_legs(pos)
        strategy = str(pos.get("type") or "").upper()
        short_put = self._optional_float(legs.get("short_put"))
        long_put = self._optional_float(legs.get("long_put"))
        short_call = self._optional_float(legs.get("short_call"))
        long_call = self._optional_float(legs.get("long_call"))
        if strategy in {"PCS", "CSP"} and short_put is None:
            short_put = self._optional_float(legs.get("short_strike") or pos.get("strike"))
        if strategy in {"CCS", "CC"} and short_call is None:
            short_call = self._optional_float(legs.get("short_strike") or pos.get("strike"))
        if strategy in {"PCS", "CCS"}:
            if long_put is None:
                long_put = self._optional_float(legs.get("long_strike")) if strategy == "PCS" else long_put
            if long_call is None:
                long_call = self._optional_float(legs.get("long_strike")) if strategy == "CCS" else long_call

        spread_width = self._spread_width(pos, short_put, long_put, short_call, long_call)
        max_loss_per_share = (
            max(0.0, spread_width - entry_premium) if spread_width is not None and spread_width > 0 else None
        )
        stop_trigger_mark = self._stop_trigger_mark(
            entry_premium=entry_premium,
            spread_width=spread_width,
        )
        if stop_trigger_mark <= 0 and entry_premium > 0:
            stop_trigger_mark = None
        profit_take_debit = (
            max(0.0, entry_premium * (1.0 - self.profit_take_pct))
            if self.profit_take_enabled and entry_premium > 0
            else None
        )
        stop_proximity = (
            (current_mark - entry_premium) / max(1e-9, (stop_trigger_mark - entry_premium))
            if current_mark is not None and stop_trigger_mark is not None and stop_trigger_mark > entry_premium
            else None
        )
        stop_distance_pct = (
            ((stop_trigger_mark - current_mark) / stop_trigger_mark) * 100.0
            if current_mark is not None and stop_trigger_mark is not None and stop_trigger_mark > 0
            else None
        )
        minutes_to_expiry = float(dte * 390) if dte is not None else None
        pnl_per_contract = (pnl_per_share * 100.0) if pnl_per_share is not None else None
        current_debit_to_stop = (
            current_mark / stop_trigger_mark
            if current_mark is not None and stop_trigger_mark not in (None, 0.0)
            else None
        )
        current_debit_to_profit_take = (
            current_mark / profit_take_debit
            if current_mark is not None and profit_take_debit not in (None, 0.0)
            else None
        )
        debit_to_width = (
            current_mark / spread_width
            if current_mark is not None and spread_width not in (None, 0.0)
            else None
        )
        loss_pct_of_max_loss = (
            loss_per_share / max_loss_per_share
            if max_loss_per_share not in (None, 0.0)
            else None
        )
        credit_retained_pct = (
            current_mark / entry_premium
            if current_mark is not None and entry_premium > 0
            else None
        )

        primary_distance_pct = None
        distances = []
        if spot and spot > 0:
            if short_put is not None:
                distances.append((spot - short_put) / spot)
            if short_call is not None:
                distances.append((short_call - spot) / spot)
        if distances:
            primary_distance_pct = min(distances, key=lambda value: abs(value))

        quote_metrics = self._quote_metrics(chain, short_put, long_put, short_call, long_call)
        risk = risk or self._risk_from_position(pos)
        market_trend_regime = str(risk.get("market_trend_regime") or pos.get("market_trend_regime") or "").lower()
        market_volatility_regime = str(
            risk.get("market_volatility_regime") or pos.get("market_volatility_regime") or ""
        ).lower()

        row = {
            "entry_premium": entry_premium,
            "current_mark": current_mark,
            "pnl_per_share": pnl_per_share,
            "pnl_per_contract": pnl_per_contract,
            "loss_per_share": loss_per_share,
            "profit_per_share": max(0.0, pnl_per_share or 0.0),
            "profit_captured_pct": profit_captured_pct,
            "dte": dte,
            "minutes_since_entry": age_minutes,
            "minutes_to_expiry": minutes_to_expiry,
            "spot": spot,
            "underlying_close": spot,
            "contracts": contracts,
            "short_put_strike": short_put,
            "long_put_strike": long_put,
            "short_call_strike": short_call,
            "long_call_strike": long_call,
            "spread_width": spread_width,
            "entry_credit": entry_premium,
            "max_loss_per_share": max_loss_per_share,
            "max_loss": max_loss_per_share,
            "current_debit": current_mark,
            "stop_trigger_mark": stop_trigger_mark,
            "stop_debit": stop_trigger_mark,
            "profit_take_debit": profit_take_debit,
            "stop_proximity": stop_proximity,
            "stop_distance_pct": stop_distance_pct,
            "current_debit_to_stop": current_debit_to_stop,
            "current_debit_to_profit_take": current_debit_to_profit_take,
            "debit_to_width": debit_to_width,
            "loss_pct_of_max_loss": loss_pct_of_max_loss,
            "credit_retained_pct": credit_retained_pct,
            "short_strike_distance_pct": primary_distance_pct,
            "has_broker_greeks": 1.0 if bool(getattr(chain, "has_broker_greeks", False) or pos.get("has_broker_greeks")) else 0.0,
            "is_pcs": 1.0 if strategy == "PCS" else 0.0,
            "is_ccs": 1.0 if strategy == "CCS" else 0.0,
            "is_csp": 1.0 if strategy == "CSP" else 0.0,
            "is_cc": 1.0 if strategy == "CC" else 0.0,
            "is_ic": 1.0 if strategy == "IC" else 0.0,
            "is_ifly": 1.0 if strategy == "IFLY" else 0.0,
            "risk_score": self._optional_float(risk.get("risk_score") if risk else None),
            "gamma_theta_ratio": self._optional_float(risk.get("gamma_theta_ratio") if risk else None),
            "net_short_delta": self._optional_float(risk.get("net_short_delta") if risk else None),
            "net_delta": self._optional_float(risk.get("net_delta") if risk else None),
            "net_gamma": self._optional_float(risk.get("net_gamma") if risk else None),
            "net_theta": self._optional_float(risk.get("net_theta") if risk else None),
            "net_vega": self._optional_float(risk.get("net_vega") if risk else None),
            "underlying_return_5m": self._optional_float(risk.get("underlying_return_5m") if risk else pos.get("underlying_return_5m")),
            "underlying_return_15m": self._optional_float(risk.get("underlying_return_15m") if risk else pos.get("underlying_return_15m")),
            "underlying_return_30m": self._optional_float(risk.get("underlying_return_30m") if risk else pos.get("underlying_return_30m")),
            "abs_underlying_return_5m": self._optional_float(risk.get("abs_underlying_return_5m") if risk else pos.get("abs_underlying_return_5m")),
            "abs_underlying_return_15m": self._optional_float(risk.get("abs_underlying_return_15m") if risk else pos.get("abs_underlying_return_15m")),
            "abs_underlying_return_30m": self._optional_float(risk.get("abs_underlying_return_30m") if risk else pos.get("abs_underlying_return_30m")),
            "underlying_realized_vol_15m": self._optional_float(risk.get("underlying_realized_vol_15m") if risk else pos.get("underlying_realized_vol_15m")),
            "underlying_realized_vol_30m": self._optional_float(risk.get("underlying_realized_vol_30m") if risk else pos.get("underlying_realized_vol_30m")),
            "underlying_vol_ratio_15m_30m": self._optional_float(risk.get("underlying_vol_ratio_15m_30m") if risk else pos.get("underlying_vol_ratio_15m_30m")),
            "short_leg_close": self._optional_float(risk.get("short_leg_close") if risk else pos.get("short_leg_close")),
            "long_leg_close": self._optional_float(risk.get("long_leg_close") if risk else pos.get("long_leg_close")),
            "short_leg_share_of_debit": self._optional_float(risk.get("short_leg_share_of_debit") if risk else pos.get("short_leg_share_of_debit")),
            "long_leg_share_of_debit": self._optional_float(risk.get("long_leg_share_of_debit") if risk else pos.get("long_leg_share_of_debit")),
            "short_leg_volume": self._optional_float(risk.get("short_leg_volume") if risk else pos.get("short_leg_volume")),
            "long_leg_volume": self._optional_float(risk.get("long_leg_volume") if risk else pos.get("long_leg_volume")),
            "short_leg_trade_count": self._optional_float(risk.get("short_leg_trade_count") if risk else pos.get("short_leg_trade_count")),
            "long_leg_trade_count": self._optional_float(risk.get("long_leg_trade_count") if risk else pos.get("long_leg_trade_count")),
            "leg_volume_imbalance": self._optional_float(risk.get("leg_volume_imbalance") if risk else pos.get("leg_volume_imbalance")),
            "leg_trade_count_imbalance": self._optional_float(risk.get("leg_trade_count_imbalance") if risk else pos.get("leg_trade_count_imbalance")),
            "market_trend_uptrend": 1.0 if market_trend_regime == "uptrend" else 0.0,
            "market_trend_sideways": 1.0 if market_trend_regime == "sideways" else 0.0,
            "market_trend_downtrend": 1.0 if market_trend_regime == "downtrend" else 0.0,
            "market_volatility_low": 1.0 if market_volatility_regime == "low" else 0.0,
            "market_volatility_medium": 1.0 if market_volatility_regime == "medium" else 0.0,
            "market_volatility_high": 1.0 if market_volatility_regime == "high" else 0.0,
            **quote_metrics,
        }
        return row

    def _stop_trigger_mark(
        self,
        *,
        entry_premium: float,
        spread_width: float | None,
    ) -> float:
        if (
            self.stop_loss_max_loss_pct is not None
            and spread_width is not None
            and spread_width > entry_premium
        ):
            return entry_premium + (spread_width - entry_premium) * self.stop_loss_max_loss_pct
        return (1.0 + self.stop_loss_multiplier) * entry_premium

    def _guard_reason(self, row: dict[str, Any]) -> str | None:
        dte = self._optional_int(row.get("dte"))
        if dte is None:
            return "missing_dte"
        if dte < self.min_dte:
            return "below_min_dte"
        if self.max_dte is not None and dte > self.max_dte:
            return "above_max_dte"

        age_minutes = self._optional_float(row.get("minutes_since_entry"))
        if age_minutes is None:
            return None
        if age_minutes < self.min_age_minutes:
            return "below_min_age_minutes"
        return None

    @staticmethod
    def _parse_legs(pos: dict) -> dict:
        from src.risk_rules.leg_specs import parse_legs

        return parse_legs(pos)

    @staticmethod
    def _risk_from_position(pos: dict) -> dict[str, Any]:
        return {
            "risk_score": pos.get("risk_score"),
            "gamma_theta_ratio": pos.get("gamma_theta_ratio") or pos.get("ratio"),
            "net_short_delta": pos.get("net_short_delta") or pos.get("short_delta"),
            "net_delta": pos.get("net_delta"),
            "net_gamma": pos.get("net_gamma"),
            "net_theta": pos.get("net_theta"),
            "net_vega": pos.get("net_vega"),
        }

    @staticmethod
    def _spread_width(
        pos: dict,
        short_put: float | None,
        long_put: float | None,
        short_call: float | None,
        long_call: float | None,
    ) -> float | None:
        configured = MlExitRiskService._optional_float(pos.get("spread_width"))
        if configured and configured > 0:
            return configured
        widths = []
        if short_put is not None and long_put is not None:
            widths.append(abs(short_put - long_put))
        if short_call is not None and long_call is not None:
            widths.append(abs(short_call - long_call))
        if widths:
            return max(widths)
        return None

    @staticmethod
    def _quote_metrics(
        chain,
        short_put: float | None,
        long_put: float | None,
        short_call: float | None,
        long_call: float | None,
    ) -> dict[str, float | None]:
        if chain is None or getattr(chain, "put_map", None) is None:
            return {
                "short_leg_bid_ask_spread_pct": None,
                "long_leg_bid_ask_spread_pct": None,
                "avg_leg_bid_ask_spread_pct": None,
            }

        metrics = []
        for strike, option_type in (
            (short_put, "put"),
            (short_call, "call"),
            (long_put, "put"),
            (long_call, "call"),
        ):
            if strike is None:
                metrics.append(None)
                continue
            row = chain.put_map.get(float(strike)) if option_type == "put" else chain.call_map.get(float(strike))
            if row is None:
                metrics.append(None)
                continue
            bid = MlExitRiskService._optional_float(row.get("bid") if hasattr(row, "get") else getattr(row, "bid", None))
            ask = MlExitRiskService._optional_float(row.get("ask") if hasattr(row, "get") else getattr(row, "ask", None))
            if bid is None or ask is None or ask <= 0:
                metrics.append(None)
                continue
            mid = (bid + ask) / 2.0
            metrics.append((ask - bid) / mid if mid > 0 else None)

        short_values = [value for value in metrics[:2] if value is not None]
        long_values = [value for value in metrics[2:] if value is not None]
        all_values = [value for value in metrics if value is not None]
        return {
            "short_leg_bid_ask_spread_pct": sum(short_values) / len(short_values) if short_values else None,
            "long_leg_bid_ask_spread_pct": sum(long_values) / len(long_values) if long_values else None,
            "avg_leg_bid_ask_spread_pct": sum(all_values) / len(all_values) if all_values else None,
        }

    @staticmethod
    def _dte(expiry: Any) -> int | None:
        if not expiry:
            return None
        try:
            if isinstance(expiry, datetime.date):
                exp_date = expiry
            else:
                exp_date = datetime.date.fromisoformat(str(expiry))
            return (exp_date - datetime.date.today()).days
        except Exception:
            return None

    @staticmethod
    def _position_age_minutes(timestamp: Any) -> float | None:
        if not timestamp:
            return None
        try:
            ts = datetime.datetime.fromisoformat(str(timestamp))
        except Exception:
            return None
        if ts.tzinfo is None:
            return max(0.0, (datetime.datetime.now() - ts).total_seconds() / 60.0)
        now = datetime.datetime.now(datetime.timezone.utc).astimezone(ts.tzinfo)
        return max(0.0, (now - ts).total_seconds() / 60.0)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            if value is None:
                raise TypeError
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            if value is None:
                raise TypeError
            return int(value)
        except (TypeError, ValueError):
            return None
