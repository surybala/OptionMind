"""Prototype historical option-candidate dataset builder.

This module is intentionally provider-agnostic. It consumes normalized
provider protocols and emits training-row shaped records that can later be
written to parquet/csv and used by feature, label, and model code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

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
    stock_lookback_days: int = 30
    forward_days: int = 30
    profit_take_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    option_timeframe: str = "1Day"
    stock_timeframe: str = "1Day"


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
    underlying_range_pct: float | None
    underlying_volume: float | None

    option_entry_price: float
    option_entry_volume: float | None
    option_exit_price: float
    exit_timestamp: datetime
    exit_reason: str

    realized_pnl_per_contract: float
    profit_label: int
    stop_loss_hit: int
    large_loss_label: int
    max_adverse_excursion: float
    max_favorable_excursion: float
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
            expiration_gte = _datetime_to_date(config.entry_start) + timedelta(days=config.min_dte)
            expiration_lte = _datetime_to_date(config.entry_start) + timedelta(days=config.max_dte)
            contracts = self.contract_provider.get_option_contracts(
                [underlying],
                expiration_gte=expiration_gte,
                expiration_lte=expiration_lte,
                status=config.contract_status,
                limit=config.option_limit,
            )
            contracts = _candidate_contracts(contracts, underlying, config)
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
                entry_bar = _first_bar_at_or_after(path, config.entry_start)
                if entry_bar is None or entry_bar.timestamp > config.entry_end:
                    continue
                entry_dte = _dte(entry_bar.timestamp, contract.expiration)
                if entry_dte is None or entry_dte < config.min_dte or entry_dte > config.max_dte:
                    continue
                future_path = [
                    bar
                    for bar in path
                    if entry_bar.timestamp <= bar.timestamp <= entry_bar.timestamp + timedelta(days=config.forward_days)
                ]
                if not future_path:
                    continue
                underlying_features = _underlying_features(stock_bars.get(underlying, []), entry_bar.timestamp)
                label = _simulate_short_option_path(
                    entry_bar,
                    future_path,
                    profit_take_pct=config.profit_take_pct,
                    stop_loss_multiple=config.stop_loss_multiple,
                )
                missing = _missing_fields(contract, underlying_features)
                rows.append(
                    CandidateDatasetRow(
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
                        underlying_range_pct=underlying_features["underlying_range_pct"],
                        underlying_volume=underlying_features["underlying_volume"],
                        option_entry_price=label["entry_price"],
                        option_entry_volume=entry_bar.volume,
                        option_exit_price=label["exit_price"],
                        exit_timestamp=label["exit_timestamp"],
                        exit_reason=label["exit_reason"],
                        realized_pnl_per_contract=label["realized_pnl_per_contract"],
                        profit_label=label["profit_label"],
                        stop_loss_hit=label["stop_loss_hit"],
                        large_loss_label=label["large_loss_label"],
                        max_adverse_excursion=label["max_adverse_excursion"],
                        max_favorable_excursion=label["max_favorable_excursion"],
                        missing_fields=tuple(missing),
                    )
                )
        return rows


def _candidate_contracts(
    contracts: list[OptionContract],
    underlying: str,
    config: CandidateDatasetConfig,
) -> list[OptionContract]:
    filtered = [
        contract
        for contract in contracts
        if contract.underlying.upper() == underlying.upper()
        and contract.expiration is not None
        and contract.strike is not None
        and contract.option_type in {"call", "put"}
    ]
    return filtered[: config.max_contracts_per_underlying]


def _simulate_short_option_path(
    entry_bar: PriceBar,
    path: list[PriceBar],
    profit_take_pct: float,
    stop_loss_multiple: float,
) -> dict[str, Any]:
    entry_price = max(float(entry_bar.close), 0.0)
    profit_take_price = entry_price * (1.0 - profit_take_pct)
    stop_price = entry_price * stop_loss_multiple

    exit_bar = path[-1]
    exit_reason = "horizon"
    max_cost = entry_price
    min_cost = entry_price

    for bar in path[1:]:
        cost = float(bar.close)
        max_cost = max(max_cost, cost)
        min_cost = min(min_cost, cost)
        if cost >= stop_price:
            exit_bar = bar
            exit_reason = "stop_loss"
            break
        if cost <= profit_take_price:
            exit_bar = bar
            exit_reason = "profit_take"
            break

    exit_price = float(exit_bar.close)
    realized = round((entry_price - exit_price) * 100, 4)
    max_adverse = round(max(0.0, (max_cost - entry_price) * 100), 4)
    max_favorable = round(max(0.0, (entry_price - min_cost) * 100), 4)
    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_timestamp": exit_bar.timestamp,
        "exit_reason": exit_reason,
        "realized_pnl_per_contract": realized,
        "profit_label": 1 if realized > 0 else 0,
        "stop_loss_hit": 1 if exit_reason == "stop_loss" else 0,
        "large_loss_label": 1 if realized <= -(entry_price * 100) else 0,
        "max_adverse_excursion": max_adverse,
        "max_favorable_excursion": max_favorable,
    }


def _underlying_features(bars: list[PriceBar], entry_timestamp: datetime) -> dict[str, float | None]:
    history = [bar for bar in sorted(bars, key=lambda b: b.timestamp) if bar.timestamp <= entry_timestamp]
    if not history:
        return {
            "underlying_close": None,
            "underlying_return_1d": None,
            "underlying_range_pct": None,
            "underlying_volume": None,
        }

    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    prev_close = previous.close if previous else None
    return_1d = ((current.close / prev_close) - 1.0) if prev_close else None
    range_pct = ((current.high - current.low) / current.close) if current.close else None
    return {
        "underlying_close": current.close,
        "underlying_return_1d": round(return_1d, 8) if return_1d is not None else None,
        "underlying_range_pct": round(range_pct, 8) if range_pct is not None else None,
        "underlying_volume": current.volume,
    }


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
