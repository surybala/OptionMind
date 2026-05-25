"""Hardcoded FOMC meeting dates and a stub for future BLS release-date scraping.

FOMC meeting dates are published roughly a year in advance on:
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

This module stores the decision-day (second day of each 2-day meeting) for
2024–2026 and exposes them as ``EconomicEvent`` objects so the builder can
treat FOMC identically to FMP-sourced CPI/NFP events.

Update procedure: once the Fed publishes the following year's calendar
(typically in November), add the new dates to ``_FOMC_DECISION_DATES`` and
bump ``_FOMC_LAST_UPDATED``.
"""
from __future__ import annotations

from datetime import date

from ml.providers.models import EconomicEvent

_FOMC_LAST_UPDATED = "2025-05-24"

# Decision day = second day of each 2-day FOMC meeting.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DECISION_DATES: list[date] = [
    # 2024
    date(2024, 1, 31),
    date(2024, 3, 20),
    date(2024, 5, 1),
    date(2024, 6, 12),
    date(2024, 7, 31),
    date(2024, 9, 18),
    date(2024, 11, 7),
    date(2024, 12, 18),
    # 2025
    date(2025, 1, 29),
    date(2025, 3, 19),
    date(2025, 5, 7),
    date(2025, 6, 18),
    date(2025, 7, 30),
    date(2025, 9, 17),
    date(2025, 10, 29),
    date(2025, 12, 10),
    # 2026
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def fomc_events(start: date, end: date) -> list[EconomicEvent]:
    """Return FOMC decision-day events between *start* and *end* (inclusive)."""
    return [
        EconomicEvent(
            event_name="FOMC Rate Decision",
            event_date=d,
            country="US",
            impact="High",
            source="fomc_hardcoded",
        )
        for d in _FOMC_DECISION_DATES
        if start <= d <= end
    ]
