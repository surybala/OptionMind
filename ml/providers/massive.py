"""Massive/Polygon provider adapter for the OptionMind ML data layer."""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import requests

from ml.providers.models import Greeks, OptionChainSnapshot, OptionContract, OptionTrade, PriceBar


class MassiveApiError(RuntimeError):
    """Redacted Massive API failure."""

    def __init__(self, safe_url: str, status_code: int | None = None, detail: str | None = None) -> None:
        self.safe_url = safe_url
        self.status_code = status_code
        message = f"Massive API request failed for {safe_url}"
        if status_code is not None:
            message += f" status={status_code}"
        if detail:
            message += f" detail={detail[:300]}"
        super().__init__(message)


@dataclass
class MassiveProvider:
    """Normalized Massive.com adapter for historical market/options data.

    Massive is the current Polygon.io API brand. The adapter accepts either
    ``MASSIVE_API_KEY`` or the legacy ``POLYGON_API_KEY`` env var.
    """

    api_key: str
    base_url: str = "https://api.massive.com"
    timeout: float = 30.0
    adjusted: bool = False
    cache_dir: Path | str | None = None
    session: Any = field(default_factory=requests.Session)

    source: str = "massive"

    @classmethod
    def from_env(cls) -> "MassiveProvider":
        api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY or POLYGON_API_KEY is required")
        return cls(api_key=api_key, cache_dir=os.getenv("MASSIVE_CACHE_DIR", "artifacts/cache/massive"))

    def get_stock_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        multiplier, timespan = _parse_timeframe(timeframe)
        return {
            symbol: self._get_aggregate_bars(
                ticker=symbol,
                start=start,
                end=end,
                multiplier=multiplier,
                timespan=timespan,
                normalized_symbol=symbol,
            )
            for symbol in symbols
        }

    def get_option_contracts(
        self,
        underlyings: list[str],
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        status: str = "active",
        limit: int | None = None,
    ) -> list[OptionContract]:
        expired = str(status).lower() in {"inactive", "expired"}
        contracts: list[OptionContract] = []
        page_limit = min(limit or 1000, 1000)

        for underlying in underlyings:
            params: dict[str, Any] = {
                "underlying_ticker": underlying,
                "expired": str(expired).lower(),
                "limit": page_limit,
                "sort": "expiration_date",
                "order": "asc",
            }
            if expiration_gte is not None:
                params["expiration_date.gte"] = expiration_gte.isoformat()
            if expiration_lte is not None:
                params["expiration_date.lte"] = expiration_lte.isoformat()

            for item in self._get_paginated("/v3/reference/options/contracts", params, limit=limit):
                contracts.append(_normalize_contract(item, source=self.source))
                if limit is not None and len(contracts) >= limit:
                    return contracts

        return contracts

    def get_current_option_chain(
        self,
        underlying: str,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        limit: int | None = None,
    ) -> dict[str, OptionChainSnapshot]:
        params: dict[str, Any] = {
            "limit": min(limit or 250, 250),
            "sort": "details.expiration_date",
            "order": "asc",
        }
        if expiration_gte is not None:
            params["expiration_date.gte"] = expiration_gte.isoformat()
        if expiration_lte is not None:
            params["expiration_date.lte"] = expiration_lte.isoformat()

        path = f"/v3/snapshot/options/{quote(underlying, safe='')}"
        rows = self._get_paginated(path, params, limit=limit)
        return {
            snapshot.symbol: snapshot
            for snapshot in (_normalize_chain_snapshot(underlying, item, source=self.source) for item in rows)
        }

    def get_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None = None,
    ) -> dict[str, list[PriceBar]]:
        multiplier, timespan = _parse_timeframe(timeframe)
        return {
            _normalize_option_symbol(symbol): self._get_aggregate_bars(
                ticker=_api_option_ticker(symbol),
                start=start,
                end=end,
                multiplier=multiplier,
                timespan=timespan,
                normalized_symbol=_normalize_option_symbol(symbol),
                limit=limit,
            )
            for symbol in symbols
        }

    def get_earnings_calendar(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> dict[str, list]:
        """Stub: Massive/Polygon earnings calendar not yet implemented.

        Returns an empty dict so callers fall back to None earnings features
        gracefully. Implement by hitting /vX/reference/financials or a dedicated
        earnings endpoint when that data becomes available.
        """
        return {}

    def get_option_trades(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int | None = None,
    ) -> dict[str, list[OptionTrade]]:
        result: dict[str, list[OptionTrade]] = {}
        for symbol in symbols:
            normalized = _normalize_option_symbol(symbol)
            params = {
                "timestamp.gte": _to_nanoseconds(start),
                "timestamp.lte": _to_nanoseconds(end),
                "sort": "timestamp",
                "order": "asc",
                "limit": min(limit or 50_000, 50_000),
            }
            path = f"/v3/trades/{quote(_api_option_ticker(symbol), safe='')}"
            rows = self._get_paginated(path, params, limit=limit)
            result[normalized] = [_normalize_trade(normalized, item, source=self.source) for item in rows]
        return result

    def _get_aggregate_bars(
        self,
        *,
        ticker: str,
        start: datetime,
        end: datetime,
        multiplier: int,
        timespan: str,
        normalized_symbol: str,
        limit: int | None = None,
    ) -> list[PriceBar]:
        path = (
            f"/v2/aggs/ticker/{quote(ticker, safe='')}/range/"
            f"{multiplier}/{timespan}/{_date_or_ms(start)}/{_date_or_ms(end)}"
        )
        params = {
            "adjusted": str(self.adjusted).lower(),
            "sort": "asc",
            "limit": min(limit or 50_000, 50_000),
        }
        try:
            payload = self._get_json(path, params)
        except MassiveApiError as exc:
            if exc.status_code == 404:
                return []
            raise
        rows = payload.get("results") or []
        if limit is not None:
            rows = rows[:limit]
        return [_normalize_bar(normalized_symbol, item, source=self.source) for item in rows]

    def _get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        next_url: str | None = path
        next_params = dict(params or {})

        while next_url:
            payload = self._get_json(next_url, next_params)
            items.extend(payload.get("results") or [])
            if limit is not None and len(items) >= limit:
                return items[:limit]
            next_url = payload.get("next_url")
            next_params = {}

        return items

    def _get_json(self, path_or_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url.rstrip('/')}{path_or_url}"
        query = dict(params or {})
        query["apiKey"] = self.api_key
        cache_path = self._cache_path(url, query)
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        try:
            response = self.session.get(url, params=query, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            safe_url = _redact_url(getattr(getattr(exc, "request", None), "url", None) or url)
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            detail = _redact_text(getattr(response, "text", "") or "")
            raise MassiveApiError(safe_url, status_code=status_code, detail=detail) from None
        payload = response.json()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_params = {key: value for key, value in params.items() if key.lower() != "apikey"}
        key = json.dumps({"url": url, "params": safe_params}, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"


def _parse_timeframe(value: str) -> tuple[int, str]:
    normalized = value.strip().lower()
    aliases = {
        "1day": (1, "day"),
        "day": (1, "day"),
        "daily": (1, "day"),
        "1d": (1, "day"),
        "1min": (1, "minute"),
        "minute": (1, "minute"),
        "min": (1, "minute"),
        "1m": (1, "minute"),
        "1hour": (1, "hour"),
        "hour": (1, "hour"),
        "hourly": (1, "hour"),
        "1h": (1, "hour"),
    }
    if normalized in aliases:
        return aliases[normalized]
    for suffix, timespan in (("min", "minute"), ("m", "minute"), ("hour", "hour"), ("h", "hour")):
        if normalized.endswith(suffix):
            raw = normalized[: -len(suffix)]
            if raw.isdigit():
                return int(raw), timespan
    raise ValueError(f"Unsupported Massive timeframe: {value}")


def _normalize_contract(item: dict[str, Any], source: str) -> OptionContract:
    return OptionContract(
        symbol=_normalize_option_symbol(str(item.get("ticker") or "")),
        underlying=str(item.get("underlying_ticker") or ""),
        expiration=_date_or_none(item.get("expiration_date")),
        strike=_float_or_none(item.get("strike_price")),
        option_type=_option_type_or_none(item.get("contract_type")),
        status="inactive" if item.get("expired") is True else "active",
        exercise_style=_str_or_none(item.get("exercise_style")),
        root_symbol=_str_or_none(item.get("root_symbol")),
        source=source,
        raw=item,
    )


def _normalize_bar(symbol: str, item: dict[str, Any], source: str) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        timestamp=_from_milliseconds(item["t"]),
        open=float(item["o"]),
        high=float(item["h"]),
        low=float(item["l"]),
        close=float(item["c"]),
        volume=_float_or_none(item.get("v")),
        trade_count=_int_or_none(item.get("n")),
        vwap=_float_or_none(item.get("vw")),
        source=source,
    )


def _normalize_trade(symbol: str, item: dict[str, Any], source: str) -> OptionTrade:
    return OptionTrade(
        symbol=symbol,
        timestamp=_from_nanoseconds(item.get("sip_timestamp") or item.get("participant_timestamp")),
        price=float(item["price"]),
        size=_int_or_none(item.get("size")),
        exchange=_str_or_none(item.get("exchange")),
        conditions=tuple(str(condition) for condition in item.get("conditions") or ()),
        source=source,
    )


def _normalize_chain_snapshot(
    underlying: str,
    item: dict[str, Any],
    source: str,
) -> OptionChainSnapshot:
    details = item.get("details") or {}
    quote = item.get("last_quote") or {}
    trade = item.get("last_trade") or {}
    greeks = item.get("greeks") or {}
    timestamp = (
        quote.get("sip_timestamp")
        or quote.get("participant_timestamp")
        or trade.get("sip_timestamp")
        or trade.get("participant_timestamp")
        or item.get("fmv_last_updated")
    )
    symbol = _normalize_option_symbol(str(details.get("ticker") or item.get("ticker") or ""))
    return OptionChainSnapshot(
        symbol=symbol,
        underlying=underlying,
        timestamp=_from_nanoseconds(timestamp) if timestamp is not None else None,
        bid=_float_or_none(quote.get("bid_price")),
        ask=_float_or_none(quote.get("ask_price")),
        last=_float_or_none(trade.get("price")),
        implied_volatility=_float_or_none(item.get("implied_volatility")),
        greeks=Greeks(
            delta=_float_or_none(greeks.get("delta")),
            gamma=_float_or_none(greeks.get("gamma")),
            theta=_float_or_none(greeks.get("theta")),
            vega=_float_or_none(greeks.get("vega")),
            rho=_float_or_none(greeks.get("rho")),
        )
        if greeks
        else None,
        source=source,
        raw=item,
    )


def _api_option_ticker(symbol: str) -> str:
    return symbol if symbol.startswith("O:") else f"O:{symbol}"


def _normalize_option_symbol(symbol: str) -> str:
    return symbol[2:] if symbol.startswith("O:") else symbol


def _date_or_ms(value: datetime) -> str:
    if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
        return value.date().isoformat()
    return str(int(value.timestamp() * 1000))


def _to_nanoseconds(value: datetime) -> int:
    return int(value.timestamp() * 1_000_000_000)


def _from_milliseconds(value: int) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _from_nanoseconds(value: Any) -> datetime:
    return datetime.fromtimestamp(int(value) / 1_000_000_000, tz=UTC)


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
    raw = str(value or "").lower()
    if raw in {"call", "put"}:
        return raw
    return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "***" if key.lower() == "apikey" else value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _redact_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("apiKey", "apiKey").replace("apikey", "apikey")
