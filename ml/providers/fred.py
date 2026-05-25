"""FRED (Federal Reserve Economic Data) provider for historical macro event dates.

FRED is operated by the St. Louis Fed and is free to use with an API key.
Sign up at https://fred.stlouisfed.org/docs/api/api_key.html — keys are
issued instantly.

Set environment variable ``FRED_API_KEY`` before use.

Why FRED instead of (or in addition to) FMP for historical data
----------------------------------------------------------------
FMP's ``/v3/economic_calendar`` endpoint has a hard **3-month window limit**
per request.  For a training dataset spanning 2020–present that means 20+
sequential API calls under the free tier's 250-call/day budget.

FRED's ``/fred/release/dates`` endpoint returns **all** historical release
dates for a given data series in one call, going back to the 1990s.  We then
filter to the requested date range client-side — single call per series,
zero pagination issues, no window arithmetic.

Supported series (release_id → name)
-------------------------------------
  10  → CPI (Consumer Price Index for All Urban Consumers)
  50  → Employment Situation (Nonfarm Payrolls / NFP)
  53  → GDP (Gross Domestic Product, advance estimate)
  46  → PPI (Producer Price Index)

Additional IDs can be added to ``_RELEASE_CONFIG`` without code changes.

Usage
-----
    from ml.providers.fred import FREDProvider

    provider = FREDProvider.from_env()
    events = provider.get_economic_calendar(date(2020, 1, 1), date(2024, 12, 31))
    # → list[EconomicEvent] with all CPI/NFP/GDP/PPI release dates in range
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any

import requests

from ml.providers.models import EconomicEvent, PriceBar

# Release IDs and human-readable names as published on FRED.
# Add more entries here to expand coverage — no other code changes needed.
_RELEASE_CONFIG: dict[int, str] = {
    10: "CPI Release",           # Consumer Price Index
    50: "Nonfarm Payrolls (NFP)",  # Employment Situation
    53: "GDP Release",            # Gross Domestic Product
    46: "PPI Release",            # Producer Price Index
}

_BASE_URL = "https://api.stlouisfed.org/fred"


@dataclass
class FREDProvider:
    """EconomicCalendarProvider and VolatilityDataProvider backed by FRED.

    Fetches all historical release dates for CPI, NFP, GDP, and PPI in a
    single call per series, then filters to the requested date range locally.
    This makes it suitable for multi-year training-data builds without
    exhausting per-day API quotas.

    Implements the ``EconomicCalendarProvider`` protocol.
    """

    api_key: str
    base_url: str = _BASE_URL
    timeout: float = 30.0
    session: Any = field(default_factory=requests.Session)
    source: str = "fred"

    # Cache release dates so repeated calls within the same builder run
    # (different date ranges, same instance) don't hit the network twice.
    _cache: dict[int, list[date]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(cls) -> "FREDProvider":
        """Construct from the ``FRED_API_KEY`` environment variable."""
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise ValueError(
                "FRED_API_KEY environment variable is required. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        return cls(api_key=api_key)

    # ------------------------------------------------------------------
    # EconomicCalendarProvider
    # ------------------------------------------------------------------

    def get_economic_calendar(
        self,
        start: date,
        end: date,
    ) -> list[EconomicEvent]:
        """Return CPI, NFP, GDP, and PPI release dates between *start* and *end*.

        The full history is fetched once per series per provider instance and
        cached; subsequent calls with different date ranges are served from the
        in-memory cache — no extra network requests.
        """
        events: list[EconomicEvent] = []
        for release_id, event_name in _RELEASE_CONFIG.items():
            for release_date in self._release_dates(release_id):
                if start <= release_date <= end:
                    events.append(
                        EconomicEvent(
                            event_name=event_name,
                            event_date=release_date,
                            country="US",
                            impact="High",
                            source=self.source,
                        )
                    )
        # Stable ordering — earliest events first.
        events.sort(key=lambda e: e.event_date)
        return events

    def get_volatility_series(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[PriceBar]]:
        """Return VIX-like daily bars from FRED observations.

        FRED publishes CBOE VIX close values as the ``VIXCLS`` series.  The
        dataset builder consumes bars, so each observation is normalized with
        open/high/low/close all set to the published close value.
        """
        result: dict[str, list[PriceBar]] = {}
        for symbol in symbols:
            normalized = symbol.upper()
            series_id = _volatility_series_id(normalized)
            if series_id is None:
                result[normalized] = []
                continue
            result[normalized] = self._series_observations(
                series_id,
                normalized,
                start.date(),
                end.date(),
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _release_dates(self, release_id: int) -> list[date]:
        """Return all historical release dates for *release_id* (cached)."""
        if release_id in self._cache:
            return self._cache[release_id]

        url = f"{self.base_url}/release/dates"
        params = {
            "release_id": release_id,
            "api_key": self.api_key,
            "file_type": "json",
            # Include release dates from the very beginning of FRED coverage.
            "include_release_dates_with_no_data": "true",
            "limit": 10000,  # well above the number of releases in any series
        }
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        dates: list[date] = []
        for item in data.get("release_dates", []):
            d = _parse_date(item.get("date"))
            if d is not None:
                dates.append(d)

        self._cache[release_id] = dates
        return dates

    def _series_observations(
        self,
        series_id: str,
        symbol: str,
        start: date,
        end: date,
    ) -> list[PriceBar]:
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        bars: list[PriceBar] = []
        for item in data.get("observations", []):
            observation_date = _parse_date(item.get("date"))
            value = _float_or_none(item.get("value"))
            if observation_date is None or value is None:
                continue
            timestamp = datetime.combine(observation_date, time.min, tzinfo=UTC)
            bars.append(
                PriceBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=value,
                    high=value,
                    low=value,
                    close=value,
                    source=self.source,
                )
            )
        return bars


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _float_or_none(value: Any) -> float | None:
    if value in (None, "."):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _volatility_series_id(symbol: str) -> str | None:
    aliases = {
        "I:VIX": "VIXCLS",
        "VIX": "VIXCLS",
        "VIXCLS": "VIXCLS",
        "^VIX": "VIXCLS",
    }
    return aliases.get(symbol.upper())
