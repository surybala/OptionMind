"""Tests for FREDProvider and FMP economic-calendar chunking.

FREDProvider
------------
- Returns EconomicEvent objects filtered to the requested date range.
- Caches release dates so the network is only hit once per series per instance.
- Correct event_name strings for each release ID.
- Zero events returned when the date range contains no releases.

FMPProvider.get_economic_calendar
----------------------------------
- Splits requests longer than 90 days into sequential 90-day chunks.
- Deduplicates events that appear in multiple chunks.
- Filters to US High-impact events matching _HIGH_IMPACT_KEYWORDS.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from ml.providers.fred import FREDProvider, _RELEASE_CONFIG
from ml.providers.fmp import FMPApiError, FMPProvider
from ml.providers.models import EconomicEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fred_response(release_id: int, dates: list[str]) -> dict:
    return {
        "release_dates": [
            {"release_id": release_id, "date": d} for d in dates
        ]
    }


def _mock_fred_session(release_dates_by_id: dict[int, list[str]]):
    """Return a mock requests.Session whose .get() returns FRED responses."""
    session = MagicMock()

    def fake_get(url, params=None, timeout=None):
        release_id = int((params or {}).get("release_id", 0))
        dates = release_dates_by_id.get(release_id, [])
        mock_resp = MagicMock()
        mock_resp.json.return_value = _make_fred_response(release_id, dates)
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    session.get.side_effect = fake_get
    return session


def _make_fmp_event(name: str, dt: str, country: str = "US", impact: str = "High"):
    return {"event": name, "date": dt, "country": country, "impact": impact}


# ---------------------------------------------------------------------------
# FREDProvider tests
# ---------------------------------------------------------------------------

class TestFREDProviderFiltering:
    """Date-range filtering is applied client-side from the full history."""

    def test_events_within_range_are_returned(self):
        session = _mock_fred_session({
            10: ["2022-01-12", "2022-02-10", "2022-03-10"],
            50: ["2022-01-07", "2022-02-04"],
            53: ["2022-01-27"],
            46: ["2022-01-13"],
        })
        provider = FREDProvider(api_key="test", session=session)
        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 31))

        dates = {e.event_date for e in events}
        assert date(2022, 1, 12) in dates   # CPI
        assert date(2022, 1, 7) in dates    # NFP
        assert date(2022, 1, 27) in dates   # GDP
        assert date(2022, 1, 13) in dates   # PPI
        # Feb dates must not leak through
        assert date(2022, 2, 10) not in dates
        assert date(2022, 2, 4) not in dates

    def test_no_events_when_range_is_empty(self):
        session = _mock_fred_session({
            10: ["2021-01-13"],
            50: ["2021-01-08"],
            53: [],
            46: [],
        })
        provider = FREDProvider(api_key="test", session=session)
        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 12, 31))
        assert events == []

    def test_inclusive_boundary_dates(self):
        session = _mock_fred_session({10: ["2022-06-10"], 50: [], 53: [], 46: []})
        provider = FREDProvider(api_key="test", session=session)

        # Exact start boundary
        events = provider.get_economic_calendar(date(2022, 6, 10), date(2022, 6, 30))
        assert len(events) == 1

        # Exact end boundary
        events2 = provider.get_economic_calendar(date(2022, 6, 1), date(2022, 6, 10))
        assert len(events2) == 1

        # Just outside the range
        events3 = provider.get_economic_calendar(date(2022, 6, 11), date(2022, 6, 30))
        assert events3 == []


class TestFREDProviderEventNames:
    """Each release ID maps to the correct human-readable event_name."""

    def _single_date_provider(self, release_id: int) -> tuple[FREDProvider, EconomicEvent]:
        session = _mock_fred_session({rid: [] for rid in _RELEASE_CONFIG})
        # Inject one date only for the target release
        session2 = _mock_fred_session({
            **{rid: [] for rid in _RELEASE_CONFIG},
            release_id: ["2023-06-15"],
        })
        provider = FREDProvider(api_key="test", session=session2)
        events = provider.get_economic_calendar(date(2023, 1, 1), date(2023, 12, 31))
        matches = [e for e in events if e.event_date == date(2023, 6, 15)]
        assert matches, f"No event found for release_id={release_id}"
        return provider, matches[0]

    def test_cpi_event_name(self):
        _, event = self._single_date_provider(10)
        assert "CPI" in event.event_name

    def test_nfp_event_name(self):
        _, event = self._single_date_provider(50)
        assert "Nonfarm" in event.event_name or "NFP" in event.event_name

    def test_gdp_event_name(self):
        _, event = self._single_date_provider(53)
        assert "GDP" in event.event_name

    def test_ppi_event_name(self):
        _, event = self._single_date_provider(46)
        assert "PPI" in event.event_name

    def test_source_is_fred(self):
        session = _mock_fred_session({10: ["2023-06-13"], 50: [], 53: [], 46: []})
        provider = FREDProvider(api_key="test", session=session)
        events = provider.get_economic_calendar(date(2023, 6, 1), date(2023, 6, 30))
        assert all(e.source == "fred" for e in events)

    def test_country_and_impact(self):
        session = _mock_fred_session({10: ["2023-06-13"], 50: [], 53: [], 46: []})
        provider = FREDProvider(api_key="test", session=session)
        events = provider.get_economic_calendar(date(2023, 6, 1), date(2023, 6, 30))
        assert all(e.country == "US" for e in events)
        assert all(e.impact == "High" for e in events)


class TestFREDProviderCaching:
    """Network is hit only once per release series per provider instance."""

    def test_cache_avoids_duplicate_requests(self):
        session = _mock_fred_session({10: ["2022-01-12", "2022-02-10"], 50: [], 53: [], 46: []})
        provider = FREDProvider(api_key="test", session=session)

        # First call — populates cache
        provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 31))
        call_count_after_first = session.get.call_count

        # Second call with different range — should not make new requests
        provider.get_economic_calendar(date(2022, 2, 1), date(2022, 2, 28))
        assert session.get.call_count == call_count_after_first, (
            "Second call should be served from cache with no extra network requests"
        )

    def test_cache_populated_per_release(self):
        session = _mock_fred_session({10: [], 50: [], 53: [], 46: []})
        provider = FREDProvider(api_key="test", session=session)
        provider.get_economic_calendar(date(2022, 1, 1), date(2022, 3, 31))
        # Four release IDs → four GET calls
        assert session.get.call_count == len(_RELEASE_CONFIG)

    def test_results_sorted_by_date(self):
        # Provide unsorted dates across two series
        session = _mock_fred_session({
            10: ["2022-03-10", "2022-01-12"],
            50: ["2022-02-04"],
            53: [], 46: [],
        })
        provider = FREDProvider(api_key="test", session=session)
        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 3, 31))
        event_dates = [e.event_date for e in events]
        assert event_dates == sorted(event_dates), "Events must be returned in ascending date order"


class TestFREDFromEnv:
    def test_from_env_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(ValueError, match="FRED_API_KEY"):
            FREDProvider.from_env()

    def test_from_env_uses_key(self, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "abc123")
        provider = FREDProvider.from_env()
        assert provider.api_key == "abc123"


class TestFREDProviderVolatilitySeries:
    def test_vix_observations_are_normalized_to_price_bars(self):
        session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "observations": [
                {"date": "2026-01-02", "value": "16.24"},
                {"date": "2026-01-05", "value": "."},
                {"date": "2026-01-06", "value": "17.10"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        session.get.return_value = mock_resp
        provider = FREDProvider(api_key="test", session=session)

        series = provider.get_volatility_series(
            ["I:VIX"],
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 31, tzinfo=UTC),
        )

        bars = series["I:VIX"]
        assert [bar.close for bar in bars] == [16.24, 17.10]
        assert bars[0].timestamp == datetime(2026, 1, 2, tzinfo=UTC)
        assert bars[0].open == bars[0].high == bars[0].low == bars[0].close
        assert bars[0].source == "fred"
        params = session.get.call_args.kwargs["params"]
        assert params["series_id"] == "VIXCLS"
        assert params["observation_start"] == "2026-01-01"
        assert params["observation_end"] == "2026-01-31"

    def test_unknown_volatility_symbol_returns_empty_series(self):
        provider = FREDProvider(api_key="test", session=MagicMock())
        series = provider.get_volatility_series(
            ["VVIX"],
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 31, tzinfo=UTC),
        )

        assert series == {"VVIX": []}


# ---------------------------------------------------------------------------
# FMPProvider.get_economic_calendar chunking tests
# ---------------------------------------------------------------------------

class TestFMPEconomicCalendarChunking:
    """Requests spanning > 90 days are split into sequential 90-day chunks."""

    def _make_fmp_provider(self, events_by_chunk: list[list[dict]]) -> FMPProvider:
        """Return an FMPProvider whose _get() returns successive chunks."""
        call_index = {"n": 0}
        chunks = events_by_chunk

        def fake_get(path, params):
            if path != "/stable/economic-calendar":
                return []
            idx = call_index["n"]
            call_index["n"] += 1
            return chunks[idx] if idx < len(chunks) else []

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]
        return provider

    def test_single_chunk_for_short_range(self):
        """A 30-day range should produce exactly one API call."""
        call_count = {"n": 0}

        def fake_get(path, params):
            call_count["n"] += 1
            return [_make_fmp_event("CPI Release", "2022-01-12")]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 30))
        assert call_count["n"] == 1

    def test_multiple_chunks_for_long_range(self):
        """A 200-day range should produce at least 2 API calls (90-day chunks)."""
        call_count = {"n": 0}

        def fake_get(path, params):
            call_count["n"] += 1
            return []

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        provider.get_economic_calendar(date(2022, 1, 1), date(2022, 7, 20))  # 200 days
        # 200 / 90 ≈ 2.2 → 3 chunks (days 0-89, 90-179, 180-199)
        assert call_count["n"] == 3

    def test_deduplication_across_chunks(self):
        """An event appearing at a chunk boundary must not be duplicated."""
        # Same event returned from two adjacent chunks
        dup_event = _make_fmp_event("Nonfarm Payrolls (NFP)", "2022-01-07")

        call_count = {"n": 0}
        def fake_get(path, params):
            call_count["n"] += 1
            return [dup_event]  # returned in every chunk

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 4, 30))
        # Despite multiple chunks each returning the duplicate, output should be unique
        dates = [e.event_date for e in events]
        assert dates.count(date(2022, 1, 7)) == 1, "Duplicate events must be deduplicated"

    def test_filters_non_us_events(self):
        def fake_get(path, params):
            return [
                _make_fmp_event("CPI Release", "2022-01-12", country="US"),
                _make_fmp_event("CPI Release", "2022-01-13", country="UK"),  # foreign
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 31))
        assert all(e.country == "US" for e in events)
        assert len(events) == 1

    def test_filters_low_impact_events(self):
        def fake_get(path, params):
            return [
                _make_fmp_event("Nonfarm Payrolls (NFP)", "2022-01-07", impact="High"),
                _make_fmp_event("Consumer Confidence", "2022-01-25", impact="Medium"),
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 31))
        assert len(events) == 1
        assert events[0].event_date == date(2022, 1, 7)

    def test_filters_unknown_keywords(self):
        """Events not matching _HIGH_IMPACT_KEYWORDS are excluded even if High-impact."""
        def fake_get(path, params):
            return [
                _make_fmp_event("CPI Release", "2022-01-12", impact="High"),
                _make_fmp_event("Housing Starts", "2022-01-19", impact="High"),
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 1, 31))
        assert len(events) == 1
        assert "CPI" in events[0].event_name

    def test_results_sorted_ascending(self):
        """Events collected across chunks are sorted chronologically."""
        def fake_get(path, params):
            # Deliberately return events out of order
            return [
                _make_fmp_event("GDP Release", "2022-03-30"),
                _make_fmp_event("CPI Release", "2022-01-12"),
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_economic_calendar(date(2022, 1, 1), date(2022, 4, 30))
        dates = [e.event_date for e in events]
        assert dates == sorted(set(dates))


class TestFMPEnrichmentCalendars:
    def test_earnings_calendar_chunks_and_filters_symbols(self):
        calls: list[dict] = []

        def fake_get(path, params):
            calls.append({"path": path, "params": params})
            return [
                {"symbol": "SPY", "date": params["from"], "period": "Q1"},
                {"symbol": "AAPL", "date": params["from"], "period": "Q1"},
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_earnings_calendar(["SPY"], date(2022, 1, 1), date(2022, 7, 20))

        assert len(calls) == 3
        assert all(call["path"] == "/stable/earnings-calendar" for call in calls)
        assert set(events) == {"SPY"}
        assert len(events["SPY"]) == 3
        assert all(event.source == "fmp" for event in events["SPY"])

    def test_dividend_calendar_chunks_and_filters_symbols(self):
        calls: list[dict] = []

        def fake_get(path, params):
            calls.append({"path": path, "params": params})
            return [
                {"symbol": "SPY", "date": params["from"], "paymentDate": params["to"], "dividend": "1.23"},
                {"symbol": "AAPL", "date": params["from"], "dividend": "0.25"},
            ]

        provider = FMPProvider(api_key="test")
        provider._get = fake_get  # type: ignore[method-assign]

        events = provider.get_dividends(["SPY"], date(2022, 1, 1), date(2022, 4, 30))

        assert len(calls) == 2
        assert all(call["path"] == "/stable/dividends-calendar" for call in calls)
        assert len(events["SPY"]) == 2
        assert events["SPY"][0].cash_amount == 1.23

    def test_from_env_sets_default_cache_dir(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "abc123")
        monkeypatch.delenv("FMP_CACHE_DIR", raising=False)

        provider = FMPProvider.from_env()

        assert provider.api_key == "abc123"
        assert str(provider.cache_dir) == "artifacts/cache/fmp"

    def test_get_uses_cache_without_api_key_in_cache_key(self, tmp_path):
        session = MagicMock()
        response = MagicMock()
        response.json.return_value = [{"symbol": "SPY", "date": "2022-01-01"}]
        response.raise_for_status = MagicMock()
        session.get.return_value = response
        provider = FMPProvider(api_key="secret", cache_dir=tmp_path, session=session)

        params = {"from": "2022-01-01", "to": "2022-01-31", "apikey": "secret"}
        first = provider._get("/stable/earnings-calendar", params)
        second = provider._get("/stable/earnings-calendar", params)

        assert first == second
        assert session.get.call_count == 1
        cache_file = next(tmp_path.glob("*.json"))
        assert "secret" not in cache_file.read_text(encoding="utf-8")

    def test_request_errors_redact_api_key(self):
        import requests

        class FailingSession:
            def get(self, url, params=None, timeout=None):
                request = requests.Request("GET", url, params=params).prepare()
                error = requests.ConnectionError("network unavailable")
                error.request = request
                raise error

        provider = FMPProvider(api_key="secret", session=FailingSession())

        with pytest.raises(FMPApiError) as exc:
            provider._get("/stable/earnings-calendar", {"from": "2022-01-01", "apikey": "secret"})

        message = str(exc.value)
        assert "secret" not in message
        assert "apikey=%3Credacted%3E" in message or "apikey=<redacted>" in message
