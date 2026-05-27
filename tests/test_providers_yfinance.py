"""Tests for YFinanceProvider.get_earnings_calendar.

Stock bar functionality is exercised via integration paths elsewhere;
these tests focus on the earnings calendar addition.

Coverage
--------
- Returns EarningsEvent objects filtered to the requested date range.
- Events outside the window are excluded.
- Symbols with no earnings data return an empty list.
- Raw download result is cached to disk on first call.
- Cache is read on subsequent calls (download not invoked again).
- Gracefully handles tickers where get_earnings_dates() raises an exception
  by falling back to the earnings_dates property.
- Returns an empty list when both methods return None/empty.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.providers.yfinance_provider import YFinanceProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_earnings_df(*date_strs: str) -> pd.DataFrame:
    """Build a minimal DataFrame that mimics yfinance earnings_dates."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz="America/New_York") for d in date_strs])
    return pd.DataFrame(
        {"EPS Estimate": [None] * len(date_strs)},
        index=idx,
    )


def _make_ticker(earnings_df: pd.DataFrame | None, raises: bool = False):
    ticker = MagicMock()
    if raises:
        ticker.get_earnings_dates.side_effect = AttributeError("not available")
    else:
        ticker.get_earnings_dates.return_value = earnings_df
    ticker.earnings_dates = earnings_df
    return ticker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetEarningsCalendarFiltering:
    def test_returns_events_in_window(self, tmp_path):
        df = _make_earnings_df("2024-02-01", "2024-05-01", "2024-08-01")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 6, 30)
            )

        events = result["AAPL"]
        assert len(events) == 2
        assert events[0].report_date == date(2024, 2, 1)
        assert events[1].report_date == date(2024, 5, 1)

    def test_excludes_events_outside_window(self, tmp_path):
        df = _make_earnings_df("2023-12-15", "2024-03-15", "2024-09-15")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["MSFT"], date(2024, 1, 1), date(2024, 6, 30)
            )

        events = result["MSFT"]
        assert len(events) == 1
        assert events[0].report_date == date(2024, 3, 15)

    def test_empty_when_no_earnings_in_window(self, tmp_path):
        df = _make_earnings_df("2023-01-15", "2023-04-15")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["SPY"], date(2024, 1, 1), date(2024, 12, 31)
            )

        assert result["SPY"] == []

    def test_events_sorted_ascending(self, tmp_path):
        # yfinance often returns earnings newest-first
        df = _make_earnings_df("2024-08-01", "2024-02-01", "2024-05-01")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
            )

        dates = [e.report_date for e in result["AAPL"]]
        assert dates == sorted(dates)

    def test_event_fields(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
            )

        ev = result["AAPL"][0]
        assert ev.symbol == "AAPL"
        assert ev.report_date == date(2024, 4, 25)
        assert ev.source == "yfinance"

    def test_symbol_uppercased(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["aapl"], date(2024, 1, 1), date(2024, 12, 31)
            )

        assert "AAPL" in result


class TestGetEarningsCalendarCaching:
    def test_result_cached_to_disk(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker) as mock_yt:
            provider.get_earnings_calendar(["AAPL"], date(2024, 1, 1), date(2024, 12, 31))

        earnings_cache = list((tmp_path / "earnings").glob("*.json"))
        assert len(earnings_cache) == 1

    def test_cache_hit_skips_download(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker) as mock_yt:
            provider.get_earnings_calendar(["AAPL"], date(2024, 1, 1), date(2024, 12, 31))
            provider.get_earnings_calendar(["AAPL"], date(2024, 1, 1), date(2024, 12, 31))

        assert ticker.get_earnings_dates.call_count == 1

    def test_different_symbols_separate_cache_entries(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            provider.get_earnings_calendar(["AAPL"], date(2024, 1, 1), date(2024, 12, 31))
            provider.get_earnings_calendar(["MSFT"], date(2024, 1, 1), date(2024, 12, 31))

        entries = list((tmp_path / "earnings").glob("*.json"))
        assert len(entries) == 2


class TestGetEarningsCalendarFallback:
    def test_falls_back_to_earnings_dates_property(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df, raises=True)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
            )

        assert len(result["AAPL"]) == 1

    def test_empty_when_none_df(self, tmp_path):
        ticker = _make_ticker(None)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
            )

        assert result["AAPL"] == []

    def test_empty_when_empty_df(self, tmp_path):
        empty_df = pd.DataFrame()
        ticker = _make_ticker(empty_df)

        provider = YFinanceProvider(cache_dir=str(tmp_path))
        with patch("yfinance.Ticker", return_value=ticker):
            result = provider.get_earnings_calendar(
                ["AAPL"], date(2024, 1, 1), date(2024, 12, 31)
            )

        assert result["AAPL"] == []


class TestEarningsLookback:
    def test_lookback_passed_to_get_earnings_dates(self, tmp_path):
        df = _make_earnings_df("2024-04-25")
        ticker = _make_ticker(df)

        provider = YFinanceProvider(cache_dir=str(tmp_path), earnings_lookback=12)
        with patch("yfinance.Ticker", return_value=ticker):
            provider.get_earnings_calendar(["AAPL"], date(2024, 1, 1), date(2024, 12, 31))

        ticker.get_earnings_dates.assert_called_once_with(limit=12)
