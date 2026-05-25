"""Prototype historical option-candidate dataset builder.

This module is intentionally provider-agnostic. It consumes normalized
provider protocols and emits training-row shaped records that can later be
written to parquet/csv and used by feature, label, and model code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from ml.labels import ShortOptionLabelConfig, label_short_option_path
from ml.providers.models import OptionContract, PriceBar
from ml.providers.protocols import MarketDataProvider, OptionContractProvider, OptionPriceProvider


@dataclass(frozen=True)
class CandidateDatasetConfig:
    underlyings: list[str]
    entry_start: datetime
    entry_end: datetime
    min_dte: int = 7
    max_dte: int = 45
    contract_status: str = "inactive"
    option_limit: int | None = 100
    max_contracts_per_underlying: int = 25
    max_rows_per_underlying: int | None = None
    max_abs_strike_distance_pct: float | None = 0.30
    min_forward_bars: int = 2
    sample_every_n_bars: int = 1
    stock_lookback_days: int = 60
    market_regime_symbol: str = "SPY"
    forward_days: int = 30
    profit_take_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    option_timeframe: str = "1Day"
    stock_timeframe: str = "1Day"
    large_loss_multiple: float = 1.0
    label_version: str = "short_option_labels_v001"
    build_window_days: int = 45


@dataclass(frozen=True)
class CandidateDatasetRow:
    entry_timestamp: datetime
    underlying: str
    option_symbol: str
    option_type: str | None
    strike: float | None
    expiration: date | None
    dte: int | None
    source: str

    underlying_close: float | None
    underlying_return_1d: float | None
    underlying_return_5d: float | None
    underlying_return_20d: float | None
    underlying_range_pct: float | None
    underlying_realized_vol_5d: float | None
    underlying_realized_vol_20d: float | None
    underlying_sma_20_distance_pct: float | None
    underlying_above_sma_20: int | None
    underlying_volatility_ratio_5d_20d: float | None
    underlying_volume: float | None
    strike_distance_pct: float | None
    moneyness: float | None

    market_regime_symbol: str | None
    market_return_5d: float | None
    market_return_20d: float | None
    market_realized_vol_5d: float | None
    market_realized_vol_20d: float | None
    market_sma_20_distance_pct: float | None
    market_above_sma_20: int | None
    market_volatility_ratio_5d_20d: float | None
    market_trend_regime: str | None
    market_volatility_regime: str | None

    option_entry_open: float
    option_entry_high: float
    option_entry_low: float
    option_entry_price: float
    option_entry_range_pct: float | None
    option_entry_volume: float | None
    option_entry_trade_count: int | None
    option_entry_vwap: float | None
    option_exit_price: float
    exit_timestamp: datetime
    exit_reason: str

    expected_pnl: float
    realized_pnl_per_contract: float
    profit_label: int
    stop_loss_hit: int
    large_loss_label: int
    max_adverse_excursion: float
    max_favorable_excursion: float
    days_to_exit: float
    label_version: str
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoricalCandidateDatasetBuilder:
    """Build prototype candidate rows from provider protocols."""

    def __init__(
        self,
        market_provider: MarketDataProvider,
        contract_provider: OptionContractProvider,
        price_provider: OptionPriceProvider,
    ) -> None:
        self.market_provider = market_provider
        self.contract_provider = contract_provider
        self.price_provider = price_provider

    def build(self, config: CandidateDatasetConfig) -> list[CandidateDatasetRow]:
        rows: list[CandidateDatasetRow] = []
        underlying_row_counts: dict[str, int] = {u: 0 for u in config.underlyings}

        stock_start = config.entry_start - timedelta(days=config.stock_lookback_days)
        stock_symbols = _stock_symbols(config)
        stock_bars = self.market_provider.get_stock_bars(
            stock_symbols,
            stock_start,
            config.entry_end,
            config.stock_timeframe,
        )
        market_history = stock_bars.get(config.market_regime_symbol.upper(), [])

        for window_start, window_end, is_last_window in _date_windows(
            config.entry_start, config.entry_end, config.build_window_days
        ):
            # Stop early if all underlyings have hit their row cap.
            if config.max_rows_per_underlying is not None and all(
                underlying_row_counts[u] >= config.max_rows_per_underlying for u in config.underlyings
            ):
                break

            expiration_gte = _datetime_to_date(window_start) + timedelta(days=config.min_dte)
            expiration_lte = _datetime_to_date(window_end) + timedelta(days=config.max_dte)

            for underlying in config.underlyings:
                if (
                    config.max_rows_per_underlying is not None
                    and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                ):
                    continue

                contracts = self.contract_provider.get_option_contracts(
                    [underlying],
                    expiration_gte=expiration_gte,
                    expiration_lte=expiration_lte,
                    status=config.contract_status,
                    limit=config.option_limit,
                )
                underlying_history = stock_bars.get(underlying, [])
                reference_features = _underlying_features(underlying_history, window_start)
                contracts = _candidate_contracts(
                    contracts,
                    underlying,
                    config,
                    reference_close=reference_features["underlying_close"],
                )
                if not contracts:
                    continue

                option_bars = self.price_provider.get_option_bars(
                    [contract.symbol for contract in contracts],
                    window_start,
                    window_end + timedelta(days=config.forward_days),
                    config.option_timeframe,
                    limit=None,
                )

                for contract in contracts:
                    path = sorted(option_bars.get(contract.symbol, []), key=lambda bar: bar.timestamp)
                    # Use exclusive upper bound except on the last window to avoid
                    # emitting duplicate rows when windows share a boundary timestamp.
                    if is_last_window:
                        window_bars = [b for b in path if window_start <= b.timestamp <= config.entry_end]
                    else:
                        window_bars = [b for b in path if window_start <= b.timestamp < window_end]
                    for entry_bar in window_bars[:: config.sample_every_n_bars]:
                        entry_dte = _dte(entry_bar.timestamp, contract.expiration)
                        if entry_dte is None or entry_dte < config.min_dte or entry_dte > config.max_dte:
                            continue
                        future_path = [
                            bar
                            for bar in path
                            if entry_bar.timestamp <= bar.timestamp <= entry_bar.timestamp + timedelta(days=config.forward_days)
                        ]
                        if len(future_path) < config.min_forward_bars:
                            continue
                        underlying_features = _underlying_features(underlying_history, entry_bar.timestamp)
                        market_features = _market_regime_features(
                            market_history,
                            entry_bar.timestamp,
                            config.market_regime_symbol,
                        )
                        label = label_short_option_path(
                            entry_bar,
                            future_path,
                            ShortOptionLabelConfig(
                                profit_take_pct=config.profit_take_pct,
                                stop_loss_multiple=config.stop_loss_multiple,
                                large_loss_multiple=config.large_loss_multiple,
                                label_version=config.label_version,
                            ),
                        )
                        rows.append(
                            _row_from_label(
                                contract,
                                underlying,
                                entry_bar,
                                entry_dte,
                                underlying_features,
                                market_features,
                                label,
                            )
                        )
                        underlying_row_counts[underlying] += 1
                        if (
                            config.max_rows_per_underlying is not None
                            and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                        ):
                            break
                    if (
                        config.max_rows_per_underlying is not None
                        and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                    ):
                        break
        return rows


def _candidate_contracts(
    contracts: list[OptionContract],
    underlying: str,
    config: CandidateDatasetConfig,
    reference_close: float | None = None,
) -> list[OptionContract]:
    filtered: list[OptionContract] = [
        contract
        for contract in contracts
        if contract.underlying.upper() == underlying.upper()
        and contract.expiration is not None
        and contract.strike is not None
        and contract.option_type in {"call", "put"}
    ]
    if reference_close and config.max_abs_strike_distance_pct is not None:
        filtered = [
            contract
            for contract in filtered
            if abs((float(contract.strike) - reference_close) / reference_close) <= config.max_abs_strike_distance_pct
        ]
    if reference_close:
        filtered = sorted(filtered, key=lambda c: abs(float(c.strike or 0.0) - reference_close))

    buckets: dict[tuple[date, str], list[OptionContract]] = {}
    for contract in filtered:
        buckets.setdefault((contract.expiration, contract.option_type), []).append(contract)

    ordered: list[OptionContract] = []
    bucket_values = list(buckets.values())
    while bucket_values and len(ordered) < config.max_contracts_per_underlying:
        next_values: list[list[OptionContract]] = []
        for bucket in bucket_values:
            if bucket and len(ordered) < config.max_contracts_per_underlying:
                ordered.append(bucket.pop(0))
            if bucket:
                next_values.append(bucket)
        bucket_values = next_values
    return ordered


def _stock_symbols(config: CandidateDatasetConfig) -> list[str]:
    symbols: list[str] = []
    for symbol in [*config.underlyings, config.market_regime_symbol]:
        normalized = symbol.upper()
        if normalized not in symbols:
            symbols.append(normalized)
    return symbols


def _date_windows(
    start: datetime,
    end: datetime,
    window_days: int,
) -> list[tuple[datetime, datetime, bool]]:
    """Non-overlapping entry windows covering [start, end].

    Returns (window_start, window_end, is_last) triples. Windows are contiguous
    with no overlap. The last window's is_last flag signals callers to use an
    inclusive upper-bound filter so the final entry_end timestamp is included.
    """
    windows: list[tuple[datetime, datetime, bool]] = []
    current = start
    while current < end:
        window_end = min(current + timedelta(days=window_days), end)
        is_last = window_end >= end
        windows.append((current, window_end, is_last))
        if is_last:
            break
        current = window_end
    return windows


def _row_from_label(
    contract: OptionContract,
    underlying: str,
    entry_bar: PriceBar,
    entry_dte: int,
    underlying_features: dict[str, Any],
    market_features: dict[str, Any],
    label: Any,
) -> CandidateDatasetRow:
    features = {**underlying_features, **market_features}
    missing = _missing_fields(contract, features)
    return CandidateDatasetRow(
        entry_timestamp=entry_bar.timestamp,
        underlying=underlying,
        option_symbol=contract.symbol,
        option_type=contract.option_type,
        strike=contract.strike,
        expiration=contract.expiration,
        dte=entry_dte,
        source=contract.source,
        underlying_close=underlying_features["underlying_close"],
        underlying_return_1d=underlying_features["underlying_return_1d"],
        underlying_return_5d=underlying_features["underlying_return_5d"],
        underlying_return_20d=underlying_features["underlying_return_20d"],
        underlying_range_pct=underlying_features["underlying_range_pct"],
        underlying_realized_vol_5d=underlying_features["underlying_realized_vol_5d"],
        underlying_realized_vol_20d=underlying_features["underlying_realized_vol_20d"],
        underlying_sma_20_distance_pct=underlying_features["underlying_sma_20_distance_pct"],
        underlying_above_sma_20=underlying_features["underlying_above_sma_20"],
        underlying_volatility_ratio_5d_20d=underlying_features["underlying_volatility_ratio_5d_20d"],
        underlying_volume=underlying_features["underlying_volume"],
        strike_distance_pct=_strike_distance_pct(contract.strike, underlying_features["underlying_close"]),
        moneyness=_moneyness(contract.strike, underlying_features["underlying_close"]),
        market_regime_symbol=market_features["market_regime_symbol"],
        market_return_5d=market_features["market_return_5d"],
        market_return_20d=market_features["market_return_20d"],
        market_realized_vol_5d=market_features["market_realized_vol_5d"],
        market_realized_vol_20d=market_features["market_realized_vol_20d"],
        market_sma_20_distance_pct=market_features["market_sma_20_distance_pct"],
        market_above_sma_20=market_features["market_above_sma_20"],
        market_volatility_ratio_5d_20d=market_features["market_volatility_ratio_5d_20d"],
        market_trend_regime=market_features["market_trend_regime"],
        market_volatility_regime=market_features["market_volatility_regime"],
        option_entry_open=float(entry_bar.open),
        option_entry_high=float(entry_bar.high),
        option_entry_low=float(entry_bar.low),
        option_entry_price=label.entry_price,
        option_entry_range_pct=_range_pct(entry_bar.high, entry_bar.low, entry_bar.close),
        option_entry_volume=entry_bar.volume,
        option_entry_trade_count=entry_bar.trade_count,
        option_entry_vwap=entry_bar.vwap,
        option_exit_price=label.exit_price,
        exit_timestamp=label.exit_timestamp,
        exit_reason=label.exit_reason,
        expected_pnl=label.expected_pnl,
        realized_pnl_per_contract=label.realized_pnl_per_contract,
        profit_label=label.profit_label,
        stop_loss_hit=label.stop_loss_hit,
        large_loss_label=label.large_loss_label,
        max_adverse_excursion=label.max_adverse_excursion,
        max_favorable_excursion=label.max_favorable_excursion,
        days_to_exit=label.days_to_exit,
        label_version=label.label_version,
        missing_fields=tuple(missing),
    )


def _underlying_features(bars: list[PriceBar], entry_timestamp: datetime) -> dict[str, Any]:
    history = [bar for bar in sorted(bars, key=lambda b: b.timestamp) if bar.timestamp <= entry_timestamp]
    if not history:
        return {
            "underlying_close": None,
            "underlying_return_1d": None,
            "underlying_return_5d": None,
            "underlying_return_20d": None,
            "underlying_range_pct": None,
            "underlying_realized_vol_5d": None,
            "underlying_realized_vol_20d": None,
            "underlying_sma_20_distance_pct": None,
            "underlying_above_sma_20": None,
            "underlying_volatility_ratio_5d_20d": None,
            "underlying_volume": None,
        }

    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    prev_close = previous.close if previous else None
    return_1d = ((current.close / prev_close) - 1.0) if prev_close else None
    return_5d = _window_return(history, periods=5)
    return_20d = _window_return(history, periods=20)
    returns = _close_returns(history)
    realized_vol_5d = _realized_vol(returns[-5:])
    realized_vol_20d = _realized_vol(returns[-20:])
    sma_20_distance_pct = _sma_distance_pct(history, periods=20)
    return {
        "underlying_close": current.close,
        "underlying_return_1d": round(return_1d, 8) if return_1d is not None else None,
        "underlying_return_5d": round(return_5d, 8) if return_5d is not None else None,
        "underlying_return_20d": round(return_20d, 8) if return_20d is not None else None,
        "underlying_range_pct": _range_pct(current.high, current.low, current.close),
        "underlying_realized_vol_5d": realized_vol_5d,
        "underlying_realized_vol_20d": realized_vol_20d,
        "underlying_sma_20_distance_pct": sma_20_distance_pct,
        "underlying_above_sma_20": _above_sma(sma_20_distance_pct),
        "underlying_volatility_ratio_5d_20d": _volatility_ratio(realized_vol_5d, realized_vol_20d),
        "underlying_volume": current.volume,
    }


def _market_regime_features(
    bars: list[PriceBar],
    entry_timestamp: datetime,
    market_regime_symbol: str,
) -> dict[str, Any]:
    features = _underlying_features(bars, entry_timestamp)
    market_sma_20_distance_pct = features["underlying_sma_20_distance_pct"]
    market_realized_vol_20d = features["underlying_realized_vol_20d"]
    return {
        "market_regime_symbol": market_regime_symbol.upper(),
        "market_return_5d": features["underlying_return_5d"],
        "market_return_20d": features["underlying_return_20d"],
        "market_realized_vol_5d": features["underlying_realized_vol_5d"],
        "market_realized_vol_20d": market_realized_vol_20d,
        "market_sma_20_distance_pct": market_sma_20_distance_pct,
        "market_above_sma_20": features["underlying_above_sma_20"],
        "market_volatility_ratio_5d_20d": features["underlying_volatility_ratio_5d_20d"],
        "market_trend_regime": _trend_regime(market_sma_20_distance_pct),
        "market_volatility_regime": _volatility_regime(market_realized_vol_20d),
    }


def _window_return(history: list[PriceBar], periods: int) -> float | None:
    if len(history) <= periods:
        return None
    current = history[-1].close
    previous = history[-(periods + 1)].close
    if not previous:
        return None
    return (current / previous) - 1.0


def _close_returns(history: list[PriceBar]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(history, history[1:]):
        if previous.close:
            returns.append((current.close / previous.close) - 1.0)
    return returns


def _realized_vol(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return round((variance ** 0.5) * (252 ** 0.5), 8)


def _sma_distance_pct(history: list[PriceBar], periods: int) -> float | None:
    if len(history) < periods:
        return None
    closes = [bar.close for bar in history[-periods:]]
    sma = sum(closes) / periods
    if not sma:
        return None
    return round((history[-1].close / sma) - 1.0, 8)


def _above_sma(sma_distance_pct: float | None) -> int | None:
    if sma_distance_pct is None:
        return None
    return int(sma_distance_pct > 0.0)


def _volatility_ratio(short_vol: float | None, long_vol: float | None) -> float | None:
    if short_vol is None or not long_vol:
        return None
    return round(short_vol / long_vol, 8)


def _trend_regime(sma_distance_pct: float | None) -> str | None:
    if sma_distance_pct is None:
        return None
    if sma_distance_pct >= 0.02:
        return "uptrend"
    if sma_distance_pct <= -0.02:
        return "downtrend"
    return "sideways"


def _volatility_regime(realized_vol_20d: float | None) -> str | None:
    if realized_vol_20d is None:
        return None
    if realized_vol_20d >= 0.30:
        return "high"
    if realized_vol_20d <= 0.15:
        return "low"
    return "normal"


def _range_pct(high: float | None, low: float | None, close: float | None) -> float | None:
    if high is None or low is None or not close:
        return None
    return round((float(high) - float(low)) / float(close), 8)


def _strike_distance_pct(strike: float | None, underlying_close: float | None) -> float | None:
    if strike is None or not underlying_close:
        return None
    return round((float(strike) - float(underlying_close)) / float(underlying_close), 8)


def _moneyness(strike: float | None, underlying_close: float | None) -> float | None:
    if strike is None or not underlying_close:
        return None
    return round(float(underlying_close) / float(strike), 8)


def _first_bar_at_or_after(path: list[PriceBar], timestamp: datetime) -> PriceBar | None:
    for bar in path:
        if bar.timestamp >= timestamp:
            return bar
    return None


def _dte(entry_timestamp: datetime, expiration: date | None) -> int | None:
    if expiration is None:
        return None
    return (expiration - entry_timestamp.date()).days


def _missing_fields(contract: OptionContract, features: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if contract.expiration is None:
        missing.append("expiration")
    if contract.strike is None:
        missing.append("strike")
    if contract.option_type is None:
        missing.append("option_type")
    for key, value in features.items():
        if value is None:
            missing.append(key)
    return missing


def _datetime_to_date(value: datetime) -> date:
    return value.date()


def market_open_utc(day: date) -> datetime:
    """Convenience helper for tests and scripts using regular U.S. market open.

    This is intentionally simple for the prototype. Later we should make market
    calendars explicit so holidays and daylight-saving shifts are handled by a
    real calendar provider.
    """
    return datetime.combine(day, time(13, 30), tzinfo=UTC)
