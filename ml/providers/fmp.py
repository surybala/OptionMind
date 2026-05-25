"""Financial Modeling Prep (FMP) provider for calendar event data.

FMP covers three event types in one API:
  - Earnings calendar   GET /stable/earnings-calendar?from=…&to=…
  - Dividend calendar   GET /stable/dividends-calendar?from=…&to=…
  - Economic calendar   GET /stable/economic-calendar?from=…&to=…

Sign up for a free key at https://financialmodelingprep.com/developer/docs
Set the environment variable FMP_API_KEY before use.

Free-tier limits (as of 2025): 250 API calls/day, daily data only.
Starter plan ($14/mo): 300 calls/min, 5-year history.
"""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from ml.providers.models import DividendEvent, EarningsEvent, EconomicEvent

# Economic events we consider "high-impact" for options Greeks/risk.
# These map onto the event_name strings returned by FMP's economic calendar.
_HIGH_IMPACT_KEYWORDS = {
    "fomc", "fed", "interest rate", "cpi", "consumer price",
    "ppi", "producer price", "nonfarm", "payroll", "unemployment",
    "gdp", "gross domestic", "pce", "core inflation",
}


class FMPApiError(RuntimeError):
    """Redacted FMP API failure."""

    def __init__(self, safe_url: str, status_code: int | None = None, detail: str | None = None) -> None:
        message = f"FMP API request failed for {safe_url}"
        if status_code is not None:
            message += f" status={status_code}"
        if detail:
            message += f" detail={detail[:300]}"
        super().__init__(message)


@dataclass
class FMPProvider:
    """FMP adapter for earnings, dividend, and economic calendar data.

    Implements EventDataProvider, DividendDataProvider, and
    EconomicCalendarProvider protocols — one instance covers all three.
    """

    api_key: str
    base_url: str = "https://financialmodelingprep.com"
    timeout: float = 30.0
    cache_dir: Path | str | None = None
    session: Any = field(default_factory=requests.Session)
    source: str = "fmp"

    @classmethod
    def from_env(cls) -> "FMPProvider":
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        return cls(api_key=api_key, cache_dir=os.getenv("FMP_CACHE_DIR", "artifacts/cache/fmp"))

    # ------------------------------------------------------------------
    # EventDataProvider
    # ------------------------------------------------------------------

    def get_earnings_calendar(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> dict[str, list[EarningsEvent]]:
        """Return earnings events keyed by symbol.

        FMP returns all earnings in the date range; we filter to requested
        symbols client-side to avoid one call per symbol.
        """
        result: dict[str, list[EarningsEvent]] = {s.upper(): [] for s in symbols}
        symbol_set = {s.upper() for s in symbols}
        for item in self._get_date_range_chunks("/stable/earnings-calendar", start, end):
            sym = (item.get("symbol") or "").upper()
            if sym not in symbol_set:
                continue
            report_date = _parse_date(item.get("date"))
            if report_date is None:
                continue
            result[sym].append(
                EarningsEvent(
                    symbol=sym,
                    report_date=report_date,
                    fiscal_period=item.get("period"),  # e.g. "Q1", "Q2"
                    source=self.source,
                )
            )
        return result

    # ------------------------------------------------------------------
    # DividendDataProvider
    # ------------------------------------------------------------------

    def get_dividends(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> dict[str, list[DividendEvent]]:
        """Return ex-dividend events keyed by symbol.

        FMP's /stable/dividends-calendar returns all symbols in the range;
        we filter client-side.
        """
        result: dict[str, list[DividendEvent]] = {s.upper(): [] for s in symbols}
        symbol_set = {s.upper() for s in symbols}
        for item in self._get_date_range_chunks("/stable/dividends-calendar", start, end):
            sym = (item.get("symbol") or "").upper()
            if sym not in symbol_set:
                continue
            # FMP field "date" is the ex-dividend date
            ex_date = _parse_date(item.get("date"))
            if ex_date is None:
                continue
            result[sym].append(
                DividendEvent(
                    symbol=sym,
                    ex_date=ex_date,
                    pay_date=_parse_date(item.get("paymentDate")),
                    declaration_date=_parse_date(item.get("declarationDate")),
                    record_date=_parse_date(item.get("recordDate")),
                    cash_amount=_float_or_none(item.get("dividend") or item.get("adjDividend")),
                    source=self.source,
                )
            )
        return result

    # ------------------------------------------------------------------
    # EconomicCalendarProvider
    # ------------------------------------------------------------------

    def get_economic_calendar(
        self,
        start: date,
        end: date,
    ) -> list[EconomicEvent]:
        """Return high-impact macro events (CPI, NFP, GDP, FOMC decisions, …).

        FMP's economic calendar endpoint is hard-capped at a **3-month window**
        per request.  This method automatically chunks multi-year date ranges
        into sequential 90-day windows so the full requested span is always
        covered — at the cost of one API call per chunk.

        Results are filtered to US High-impact events whose names match the
        known high-volatility keyword set.
        """
        events: list[EconomicEvent] = []
        seen: set[tuple[str, date]] = set()  # deduplicate across chunk edges

        for item in self._get_date_range_chunks("/stable/economic-calendar", start, end):
            if (item.get("country") or "").upper() != "US":
                continue
            if (item.get("impact") or "").lower() != "high":
                continue
            event_date = _parse_date((item.get("date") or "")[:10])
            if event_date is None:
                continue
            name = item.get("event", "")
            name_lower = name.lower()
            if not any(kw in name_lower for kw in _HIGH_IMPACT_KEYWORDS):
                continue
            key = (name, event_date)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                EconomicEvent(
                    event_name=name,
                    event_date=event_date,
                    country="US",
                    impact="High",
                    actual=_float_or_none(item.get("actual")),
                    estimate=_float_or_none(item.get("estimate")),
                    source=self.source,
                )
            )

        events.sort(key=lambda e: e.event_date)
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_date_range_chunks(
        self,
        path: str,
        start: date,
        end: date,
        *,
        chunk_days: int = 90,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end)
            rows.extend(
                self._get(
                    path,
                    {
                        "from": chunk_start.isoformat(),
                        "to": chunk_end.isoformat(),
                        "apikey": self.api_key,
                    },
                )
            )
            chunk_start = chunk_end + timedelta(days=1)
        return rows

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        cache_path = self._cache_path(url, params)
        if cache_path is not None and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            safe_url = _redact_url(getattr(getattr(exc, "request", None), "url", None) or url)
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            detail = _redact_text(getattr(response, "text", "") or "")
            raise FMPApiError(safe_url, status_code=status_code, detail=detail) from None
        data = resp.json()
        rows = data if isinstance(data, list) else []
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(rows, sort_keys=True) + "\n", encoding="utf-8")
        return rows

    def _cache_path(self, url: str, params: dict[str, Any]) -> Path | None:
        if self.cache_dir is None:
            return None
        safe_params = {key: value for key, value in params.items() if key.lower() != "apikey"}
        key = json.dumps({"url": url, "params": safe_params}, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redact_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    query = urlencode(
        [(key, "<redacted>" if key.lower() == "apikey" else value) for key, value in parse_qsl(parts.query, keep_blank_values=True)]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _redact_text(text: str) -> str:
    if not text:
        return ""
    return text
