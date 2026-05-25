"""Hardcoded FOMC meeting dates covering 2020–2026.

FOMC meeting dates are published roughly a year in advance on:
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

This module stores the decision-day (second day of each 2-day meeting) and
exposes them as ``EconomicEvent`` objects so the builder can treat FOMC
identically to FRED-sourced CPI/NFP events.

Update procedure: once the Fed publishes the following year's calendar
(typically in November), append the new dates to ``_FOMC_DECISION_DATES``
and bump ``_FOMC_LAST_UPDATED``.

Historical coverage: 2020-01-01 onward (sufficient for any training window
built from data starting in 2020).  Emergency 2020 inter-meeting actions
(March 3, 15, 19, 23, 31 notation votes) are excluded — they were not
full FOMC press-conference events and do not drive the same IV dynamics.
"""
from __future__ import annotations

from datetime import date

from ml.providers.models import EconomicEvent

_FOMC_LAST_UPDATED = "2025-05-24"

# Decision day = second day of each 2-day FOMC meeting.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_DECISION_DATES: list[date] = [
    # 2020
    date(2020, 1, 29),
    date(2020, 3, 15),   # unscheduled emergency cut (COVID)
    date(2020, 4, 29),
    date(2020, 6, 10),
    date(2020, 7, 29),
    date(2020, 9, 16),
    date(2020, 11, 5),
    date(2020, 12, 16),
    # 2021
    date(2021, 1, 27),
    date(2021, 3, 17),
    date(2021, 4, 28),
    date(2021, 6, 16),
    date(2021, 7, 28),
    date(2021, 9, 22),
    date(2021, 11, 3),
    date(2021, 12, 15),
    # 2022
    date(2022, 1, 26),
    date(2022, 3, 16),
    date(2022, 5, 4),
    date(2022, 6, 15),
    date(2022, 7, 27),
    date(2022, 9, 21),
    date(2022, 11, 2),
    date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),
    date(2023, 3, 22),
    date(2023, 5, 3),
    date(2023, 6, 14),
    date(2023, 7, 26),
    date(2023, 9, 20),
    date(2023, 11, 1),
    date(2023, 12, 13),
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
