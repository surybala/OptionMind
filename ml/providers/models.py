"""Normalized provider data models for ML dataset generation.

These models are intentionally vendor-neutral. Feature builders and labelers
should depend on these shapes instead of Alpaca, OptionMetrics, Cboe, or any
other provider-specific response object.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal


OptionType = Literal["call", "put"]
ContractStatus = Literal["active", "inactive", "unknown"]


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    trade_count: int | None = None
    vwap: float | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class OptionTrade:
    symbol: str
    timestamp: datetime
    price: float
    size: int | None = None
    exchange: str | None = None
    conditions: tuple[str, ...] = ()
    source: str = "unknown"


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    underlying: str
    expiration: date | None
    strike: float | None
    option_type: OptionType | None
    status: ContractStatus = "unknown"
    exercise_style: str | None = None
    root_symbol: str | None = None
    source: str = "unknown"
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class Greeks:
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None


@dataclass(frozen=True)
class EarningsEvent:
    """Normalized earnings calendar event."""

    symbol: str
    report_date: date
    fiscal_period: str | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class OptionChainSnapshot:
    symbol: str
    underlying: str
    timestamp: datetime | None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    implied_volatility: float | None = None
    greeks: Greeks | None = None
    source: str = "unknown"
    raw: dict[str, Any] | None = None
