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
    stock_lookback_days: int = 30
    forward_days: int = 30
    profit_take_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    option_timeframe: str = "1Day"
    stock_timeframe: str = "1Day"
    large_loss_multiple: float = 1.0
    label_version: str = "short_option_labels_v001"


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
    underlying_range_pct: float | None
    underlying_realized_vol_5d: float | None
    underlying_realized_vol_20d: float | None
    underlying_volume: float | None
    strike_distance_pct: float | None
    moneyness: float | None

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
        stock_start = config.entry_start - timedelta(days=config.stock_lookback_days)
        stock_bars = self.market_provider.get_stock_bars(
            config.underlyings,
            stock_start,
            config.entry_end,
            config.stock_timeframe,
        )

        for underlying in config.underlyings:
            underlying_row_count = 0
            expiration_gte = _datetime_to_date(config.entry_start) + timedelta(days=config.min_dte)
            expiration_lte = _datetime_to_date(config.entry_end) + timedelta(days=config.max_dte)
            contracts = self.contract_provider.get_option_contracts(
                [underlying],
                expiration_gte=expiration_gte,
                expiration_lte=expiration_lte,
                status=config.contract_status,
                limit=config.option_limit,
            )
            underlying_history = stock_bars.get(underlying, [])
            reference_features = _underlying_features(underlying_history, config.entry_start)
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
                config.entry_start,
                config.entry_end + timedelta(days=config.forward_days),
                config.option_timeframe,
                limit=None,
            )

            for contract in contracts:
                path = sorted(option_bars.get(contract.symbol, []), key=lambda bar: bar.timestamp)
                for entry_bar in _entry_bars(path, config):
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
                    rows.append(_row_from_label(contract, underlying, entry_bar, entry_dte, underlying_features, label))
                    underlying_row_count += 1
                    if (
                        config.max_rows_per_underlying is not None
                        and underlying_row_count >= config.max_rows_per_underlying
                    ):
                        break
                if (
                    config.max_rows_per_underlying is not None
                    and underlying_row_count >= config.max_rows_per_underlying
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


def _entry_bars(path: list[PriceBar], config: CandidateDatasetConfig) -> list[PriceBar]:
    if config.sample_every_n_bars <= 0:
        raise ValueError("sample_every_n_bars must be positive")
    bars = [bar for bar in path if config.entry_start <= bar.timestamp <= config.entry_end]
    return bars[:: config.sample_every_n_bars]


def _row_from_label(
    contract: OptionContract,
    underlying: str,
    entry_bar: PriceBar,
    entry_dte: int,
    underlying_features: dict[str, float | None],
    label: Any,
) -> CandidateDatasetRow:
    missing = _missing_fields(contract, underlying_features)
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
        underlying_range_pct=underlying_features["underlying_range_pct"],
        underlying_realized_vol_5d=underlying_features["underlying_realized_vol_5d"],
        underlying_realized_vol_20d=underlying_features["underlying_realized_vol_20d"],
        underlying_volume=underlying_features["underlying_volume"],
        strike_distance_pct=_strike_distance_pct(contract.strike, underlying_features["underlying_close"]),
        moneyness=_moneyness(contract.strike, underlying_features["underlying_close"]),
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


def _underlying_features(bars: list[PriceBar], entry_timestamp: datetime) -> dict[str, float | None]:
    history = [bar for bar in sorted(bars, key=lambda b: b.timestamp) if bar.timestamp <= entry_timestamp]
    if not history:
        return {
            "underlying_close": None,
            "underlying_return_1d": None,
            "underlying_return_5d": None,
            "underlying_range_pct": None,
            "underlying_realized_vol_5d": None,
            "underlying_realized_vol_20d": None,
            "underlying_volume": None,
        }

    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    prev_close = previous.close if previous else None
    return_1d = ((current.close / prev_close) - 1.0) if prev_close else None
    return_5d = _window_return(history, periods=5)
    returns = _close_returns(history)
    return {
        "underlying_close": current.close,
        "underlying_return_1d": round(return_1d, 8) if return_1d is not None else None,
        "underlying_return_5d": round(return_5d, 8) if return_5d is not None else None,
        "underlying_range_pct": _range_pct(current.high, current.low, current.close),
        "underlying_realized_vol_5d": _realized_vol(returns[-5:]),
        "underlying_realized_vol_20d": _realized_vol(returns[-20:]),
        "underlying_volume": current.volume,
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


def _missing_fields(contract: OptionContract, features: dict[str, float | None]) -> list[str]:
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
