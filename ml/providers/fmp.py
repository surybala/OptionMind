"""Financial Modeling Prep (FMP) provider for calendar event data.

FMP covers three event types in one API:
  - Earnings calendar   GET /v3/earning_calendar?from=…&to=…
  - Dividend calendar   GET /v3/stock_dividend_calendar?from=…&to=…
  - Economic calendar   GET /v3/economic_calendar?from=…&to=…

Sign up for a free key at https://financialmodelingprep.com/developer/docs
Set the environment variable FMP_API_KEY before use.

Free-tier limits (as of 2025): 250 API calls/day, daily data only.
Starter plan ($14/mo): 300 calls/min, 5-year history.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import requests

from ml.providers.models import DividendEvent, EarningsEvent, EconomicEvent

# Economic events we consider "high-impact" for options Greeks/risk.
# These map onto the event_name strings returned by FMP's economic calendar.
_HIGH_IMPACT_KEYWORDS = {
    "fomc", "fed", "interest rate", "cpi", "consumer price",
    "ppi", "producer price", "nonfarm", "payroll", "unemployment",
    "gdp", "gross domestic", "pce", "core inflation",
}


@dataclass
class FMPProvider:
    """FMP adapter for earnings, dividend, and economic calendar data.

    Implements EventDataProvider, DividendDataProvider, and
    EconomicCalendarProvider protocols — one instance covers all three.
    """

    api_key: str
    base_url: str = "https://financialmodelingprep.com/api"
    timeout: float = 30.0
    session: Any = field(default_factory=requests.Session)
    source: str = "fmp"

    @classmethod
    def from_env(cls) -> "FMPProvider":
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            raise ValueError("FMP_API_KEY environment variable is required")
        return cls(api_key=api_key)

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
        params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": self.api_key}
        for item in self._get("/v3/earning_calendar", params):
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

        FMP's /v3/stock_dividend_calendar returns all symbols in the range;
        we filter client-side.
        """
        result: dict[str, list[DividendEvent]] = {s.upper(): [] for s in symbols}
        symbol_set = {s.upper() for s in symbols}
        params = {"from": start.isoformat(), "to": end.isoformat(), "apikey": self.api_key}
        for item in self._get("/v3/stock_dividend_calendar", params):
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
                    cash_amount=_float_or_none(item.get("dividend")),
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

        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(
                date(chunk_start.year + (chunk_start.month // 10),
                     (chunk_start.month % 10 + 1) if chunk_start.month < 10 else
                     (chunk_start.month - 9), 1) - timedelta(days=1),
                end,
            )
            # Simpler 90-day chunks that avoid month-arithmetic edge cases.
            chunk_end = min(chunk_start + timedelta(days=89), end)

            params = {
                "from": chunk_start.isoformat(),
                "to": chunk_end.isoformat(),
                "apikey": self.api_key,
            }
            for item in self._get("/v3/economic_calendar", params):
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

            chunk_start = chunk_end + timedelta(days=1)

        events.sort(key=lambda e: e.event_date)
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []


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
