"""Alpaca provider adapter for the OptionMind ML data layer."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any

from ml.providers.models import Greeks, OptionChainSnapshot, OptionContract, OptionTrade, PriceBar


_OSI_RE = re.compile(r"^([A-Z./]+)(\d{6})([CP])(\d{8})$")


@dataclass
class AlpacaProvider:
    """Normalized Alpaca adapter for market, option, chain, and contract data."""

    api_key: str
    api_secret: str
    paper: bool = True
    stock_feed: str = "sip"
    option_feed: str = "opra"
    stock_client: Any | None = None
    option_client: Any | None = None
    trading_client: Any | None = None

    source: str = "alpaca"

    @classmethod
    def from_env(cls, paper: bool = True) -> "AlpacaProvider":
        api_key = os.getenv("ALPACA_API_KEY")
        api_secret = os.getenv("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET are required")
        return cls(api_key=api_key, api_secret=api_secret, paper=paper)

    def get_stock_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        try:
            from alpaca.data.enums import DataFeed
            from alpaca.data.requests import StockBarsRequest
            feed = _enum_value(DataFeed, self.stock_feed)
        except Exception:
            StockBarsRequest = SimpleNamespace
            feed = self.stock_feed

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            feed=feed,
        )
        return _normalize_bars(self._stock().get_stock_bars(req), source=self.source)

    def get_option_contracts(
        self,
        underlyings: list[str],
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        status: str = "active",
        limit: int | None = None,
    ) -> list[OptionContract]:
        try:
            from alpaca.trading.enums import AssetStatus
            from alpaca.trading.requests import GetOptionContractsRequest
            status_value = _enum_value(AssetStatus, status)
        except Exception:
            GetOptionContractsRequest = SimpleNamespace
            status_value = status

        req = GetOptionContractsRequest(
            underlying_symbols=underlyings,
            status=status_value,
            expiration_date_gte=expiration_gte,
            expiration_date_lte=expiration_lte,
            limit=limit,
        )
        response = self._trading().get_option_contracts(req)
        return [_normalize_contract(contract, source=self.source) for contract in _contracts_from_response(response)]

    def get_current_option_chain(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        limit: int | None = None,
    ) -> dict[str, OptionChainSnapshot]:
        try:
            from alpaca.data.enums import OptionsFeed
            from alpaca.data.requests import OptionChainRequest
            feed = _enum_value(OptionsFeed, self.option_feed)
        except Exception:
            OptionChainRequest = SimpleNamespace
            feed = self.option_feed

        req = OptionChainRequest(
            underlying_symbol=underlying,
            feed=feed,
            expiration_date_gte=expiration_gte,
            expiration_date_lte=expiration_lte,
        )
        data = _mapping_data(self._option().get_option_chain(req))
        items = list(data.items())[:limit] if limit else data.items()
        return {
            symbol: _normalize_chain_snapshot(symbol, underlying, snapshot, source=self.source)
            for symbol, snapshot in items
        }

    def get_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None = None,
    ) -> dict[str, list[PriceBar]]:
        try:
            from alpaca.data.requests import OptionBarsRequest
        except Exception:
            OptionBarsRequest = SimpleNamespace

        req = OptionBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=_alpaca_timeframe(timeframe),
            start=start,
            end=end,
            limit=limit,
        )
        return _normalize_bars(self._option().get_option_bars(req), source=self.source)

    def get_option_trades(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> dict[str, list[OptionTrade]]:
        try:
            from alpaca.data.requests import OptionTradesRequest
        except Exception:
            OptionTradesRequest = SimpleNamespace

        req = OptionTradesRequest(
            symbol_or_symbols=symbols,
            start=start,
            end=end,
            limit=limit,
        )
        return _normalize_trades(self._option().get_option_trades(req), source=self.source)

    def _stock(self) -> Any:
        if self.stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self.stock_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        return self.stock_client

    def _option(self) -> Any:
        if self.option_client is None:
            from alpaca.data.historical import OptionHistoricalDataClient

            self.option_client = OptionHistoricalDataClient(self.api_key, self.api_secret)
        return self.option_client

    def _trading(self) -> Any:
        if self.trading_client is None:
            from alpaca.trading.client import TradingClient

            self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
        return self.trading_client


def _normalize_bars(result: Any, source: str) -> dict[str, list[PriceBar]]:
    normalized: dict[str, list[PriceBar]] = {}
    for symbol, rows in _mapping_data(result).items():
        normalized[str(symbol)] = [
            PriceBar(
                symbol=str(getattr(row, "symbol", symbol)),
                timestamp=getattr(row, "timestamp"),
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=_float_or_none(getattr(row, "volume", None)),
                trade_count=_int_or_none(getattr(row, "trade_count", None)),
                vwap=_float_or_none(getattr(row, "vwap", None)),
                source=source,
            )
            for row in rows
        ]
    return normalized


def _normalize_trades(result: Any, source: str) -> dict[str, list[OptionTrade]]:
    normalized: dict[str, list[OptionTrade]] = {}
    for symbol, rows in _mapping_data(result).items():
        normalized[str(symbol)] = [
            OptionTrade(
                symbol=str(getattr(row, "symbol", symbol)),
                timestamp=getattr(row, "timestamp"),
                price=float(getattr(row, "price")),
                size=_int_or_none(getattr(row, "size", None)),
                exchange=getattr(row, "exchange", None),
                conditions=tuple(getattr(row, "conditions", None) or ()),
                source=source,
            )
            for row in rows
        ]
    return normalized


def _normalize_contract(contract: Any, source: str) -> OptionContract:
    symbol = str(getattr(contract, "symbol", ""))
    parsed = _parse_osi(symbol)
    underlying = (
        getattr(contract, "underlying_symbol", None)
        or getattr(contract, "underlying", None)
        or (parsed["underlying"] if parsed else "")
    )
    expiration = getattr(contract, "expiration_date", None) or (parsed["expiration"] if parsed else None)
    strike = getattr(contract, "strike_price", None) or (parsed["strike"] if parsed else None)
    option_type = getattr(contract, "type", None) or (parsed["option_type"] if parsed else None)
    status = getattr(contract, "status", "unknown")

    return OptionContract(
        symbol=symbol,
        underlying=str(underlying),
        expiration=_date_or_none(expiration),
        strike=_float_or_none(strike),
        option_type=_option_type_or_none(option_type),
        status=str(getattr(status, "value", status) or "unknown"),
        exercise_style=_value_or_none(getattr(contract, "style", None)),
        root_symbol=_value_or_none(getattr(contract, "root_symbol", None)),
        source=source,
        raw=_public_attrs(contract),
    )


def _normalize_chain_snapshot(
    symbol: str,
    underlying: str,
    snapshot: Any,
    source: str,
) -> OptionChainSnapshot:
    quote = getattr(snapshot, "latest_quote", None)
    trade = getattr(snapshot, "latest_trade", None)
    greeks = getattr(snapshot, "greeks", None)
    timestamp = (
        getattr(quote, "timestamp", None)
        or getattr(trade, "timestamp", None)
        or getattr(snapshot, "timestamp", None)
    )
    return OptionChainSnapshot(
        symbol=symbol,
        underlying=underlying,
        timestamp=timestamp,
        bid=_float_or_none(getattr(quote, "bid_price", None)),
        ask=_float_or_none(getattr(quote, "ask_price", None)),
        last=_float_or_none(getattr(trade, "price", None)),
        implied_volatility=_float_or_none(getattr(snapshot, "implied_volatility", None)),
        greeks=(
            Greeks(
                delta=_float_or_none(getattr(greeks, "delta", None)),
                gamma=_float_or_none(getattr(greeks, "gamma", None)),
                theta=_float_or_none(getattr(greeks, "theta", None)),
                vega=_float_or_none(getattr(greeks, "vega", None)),
                rho=_float_or_none(getattr(greeks, "rho", None)),
            )
            if greeks is not None
            else None
        ),
        source=source,
        raw=_public_attrs(snapshot),
    )


def _alpaca_timeframe(value: str) -> Any:
    normalized = value.lower()
    try:
        from alpaca.data.timeframe import TimeFrame
    except Exception:
        return normalized

    if normalized in {"1min", "minute", "min"}:
        return TimeFrame.Minute
    if normalized in {"1day", "day", "daily"}:
        return TimeFrame.Day
    if normalized in {"1hour", "hour", "hourly"}:
        return TimeFrame.Hour
    raise ValueError(f"Unsupported Alpaca timeframe: {value}")


def _enum_value(enum_cls: Any, raw: str) -> Any:
    raw_lower = str(raw).lower()
    for item in enum_cls:
        if item.value.lower() == raw_lower or item.name.lower() == raw_lower:
            return item
    raise ValueError(f"Unsupported {enum_cls.__name__} value: {raw}")


def _mapping_data(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    data = getattr(result, "data", result)
    if hasattr(data, "items"):
        return dict(data.items())
    return {}


def _contracts_from_response(response: Any) -> list[Any]:
    if response is None:
        return []
    contracts = getattr(response, "option_contracts", None)
    if contracts is not None:
        return list(contracts)
    if isinstance(response, list):
        return response
    return []


def _parse_osi(symbol: str) -> dict[str, Any] | None:
    match = _OSI_RE.match(symbol)
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


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _date_or_none(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _option_type_or_none(value: Any) -> str | None:
    raw = str(getattr(value, "value", value) or "").lower()
    if raw in {"call", "put"}:
        return raw
    return None


def _value_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _public_attrs(value: Any) -> dict[str, Any]:
    raw = getattr(value, "__dict__", {})
    return {k: v for k, v in raw.items() if not k.startswith("_")} or None
