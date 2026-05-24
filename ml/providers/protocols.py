"""Provider protocols for the OptionMind ML data layer.

Concrete providers can be brokers, paid historical datasets, local parquet
files, or synthetic fixtures. The dataset builder should depend on these
protocols instead of provider-specific SDKs.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Protocol

from ml.providers.models import OptionChainSnapshot, OptionContract, OptionTrade, PriceBar


class MarketDataProvider(Protocol):
    def get_stock_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        """Return normalized underlying bars keyed by symbol."""


class OptionContractProvider(Protocol):
    def get_option_contracts(
        self,
        underlyings: list[str],
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        status: str = "active",
        limit: int | None = None,
    ) -> list[OptionContract]:
        """Return normalized option contract metadata."""


class OptionChainProvider(Protocol):
    def get_current_option_chain(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        limit: int | None = None,
    ) -> dict[str, OptionChainSnapshot]:
        """Return current chain snapshots keyed by option symbol."""


class OptionPriceProvider(Protocol):
    def get_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None = None,
    ) -> dict[str, list[PriceBar]]:
        """Return normalized historical option bars keyed by option symbol."""

    def get_option_trades(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> dict[str, list[OptionTrade]]:
        """Return normalized historical option trades keyed by option symbol."""


class VolatilityDataProvider(Protocol):
    def get_volatility_series(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[PriceBar]]:
        """Return volatility index or volatility-derived series."""


class EventDataProvider(Protocol):
    def get_events(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Return earnings, macro, dividend, or other event records."""
