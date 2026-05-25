"""Feature source documentation and completeness validation.

Each feature group in CandidateDatasetRow comes from a distinct data source:

┌─────────────────────────────────────┬──────────────────────────────────────────────────┐
│ Feature group                       │ Data source                                      │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ strike, dte, option_type,           │ OptionContractProvider.get_option_contracts()     │
│ expiration, moneyness,              │ Polygon: /v3/reference/options/contracts          │
│ strike_distance_pct                 │                                                  │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ underlying_close,                   │ MarketDataProvider.get_stock_bars()              │
│ underlying_return_{1d,3d,5d,20d},   │ for the underlying symbol (SPY, QQQ …)           │
│ underlying_realized_vol_{5d,10d,20d}│ Polygon: /v2/aggs/ticker/{sym}/range/…           │
│ underlying_sma_20_distance_pct,     │                                                  │
│ underlying_above_sma_20,            │                                                  │
│ underlying_volatility_ratio_5d_20d, │                                                  │
│ underlying_volume, underlying_skew_5d│                                                 │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ market_return_{5d,20d},             │ Same get_stock_bars() call — market_regime_symbol│
│ market_realized_vol_{5d,20d},       │ (SPY by default). Reuses the underlying bar      │
│ market_sma_20_distance_pct,         │ series when underlying == market_regime_symbol.  │
│ market_above_sma_20,                │                                                  │
│ market_volatility_ratio_5d_20d,     │                                                  │
│ market_trend_regime,                │                                                  │
│ market_volatility_regime            │                                                  │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ option_entry_open/high/low,         │ OptionPriceProvider.get_option_bars()            │
│ option_entry_price,                 │ for the specific option symbol                   │
│ option_entry_range_pct,             │ Polygon: /v2/aggs/ticker/O:{sym}/range/…         │
│ option_entry_volume,                │                                                  │
│ option_entry_trade_count,           │                                                  │
│ option_entry_vwap                   │                                                  │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ option_volume_5d_avg,               │ Same get_option_bars() call — bars in the        │
│ option_trade_count_5d_avg           │ 5-day window BEFORE the entry timestamp.         │
│                                     │ Fetch window extended by option_lookback_days.   │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ implied_volatility,                 │ COMPUTED internally — no API call.               │
│ option_delta, option_gamma,         │ Inputs: option entry price, underlying close,    │
│ option_theta, option_vega,          │ strike, DTE, option_type, risk_free_rate config. │
│ iv_vs_hv5d, iv_vs_hv20d            │ Algorithm: Newton-Raphson B-S IV + analytic Greeks│
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ vix_close, vix_return_5d,           │ MarketDataProvider.get_stock_bars()              │
│ vix_realized_vol_5d,                │ for vix_symbol (default "VIX").                 │
│ vix_above_20, vix_above_30          │ ⚠ Polygon/Massive uses ticker "I:VIX" for the    │
│                                     │   CBOE index — set vix_symbol="I:VIX" in prod.  │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ days_to_earnings,                   │ EventDataProvider.get_earnings_calendar()        │
│ has_earnings_in_forward_days        │ ⚠ STUBBED — MassiveProvider always returns {}.  │
│                                     │   Will be None/0 until wired to real source.     │
└─────────────────────────────────────┴──────────────────────────────────────────────────┘
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.providers.models import EarningsEvent, OptionContract, PriceBar


# ---------------------------------------------------------------------------
# Full-coverage fake providers — each source returns enough data for every
# feature computation to succeed.
# ---------------------------------------------------------------------------

_ENTRY = datetime(2026, 5, 14, tzinfo=UTC)
_EXPIRY = date(2026, 6, 26)   # 43 DTE from entry
_STRIKE = 500.0
_UNDERLYING_CLOSE = 504.0
_VIX_CLOSE = 22.0             # intentionally > 20 so vix_above_20 = 1


def _make_spy_bars() -> list[PriceBar]:
    """25 daily bars (24 pre-entry history + entry bar).

    Prices increase slightly with small irregularity so return-series std
    is non-trivial and skewness can be computed.
    """
    bars = []
    # alternating bumps create genuine return variance and non-zero skewness
    closes = [480.0 + i * 1.0 + (2.0 if i % 4 == 0 else 0.0) for i in range(25)]
    closes.append(_UNDERLYING_CLOSE)   # entry bar close = 504
    for i, close in enumerate(closes):
        day = _ENTRY - timedelta(days=len(closes) - 1 - i)
        bars.append(PriceBar("SPY", day, close - 1, close + 1, close - 2, close, volume=1_000_000))
    return bars


def _make_vix_bars() -> list[PriceBar]:
    """25 VIX bars ending at 22 (above 20 threshold) with enough history for
    5d return and 5d realized vol.
    """
    bars = []
    for offset in range(25, 0, -1):
        day = _ENTRY - timedelta(days=offset)
        close = 18.0 + offset * 0.1  # 20.5 → 18.1 as we approach entry
        bars.append(PriceBar("VIX", day, close - 0.5, close + 0.5, close - 1, close))
    bars.append(PriceBar("VIX", _ENTRY, 21.0, 23.0, 20.5, _VIX_CLOSE))
    return bars


def _make_option_bars(symbol: str) -> list[PriceBar]:
    """6 pre-entry bars (for lookback) + 1 entry bar + 3 forward bars (for labeling).

    Pre-entry bars: 6 days before entry → 5 fall inside the 5d lookback window
    Entry bar: close 4.0, realistic spread and volume
    Forward bars: declining toward profit-take
    """
    bars = []
    # pre-entry lookback bars (entry-6 to entry-1)
    for offset in range(6, 0, -1):
        day = _ENTRY - timedelta(days=offset)
        bars.append(PriceBar(symbol, day, 3.8, 4.2, 3.6, 4.0, volume=80.0, trade_count=8, vwap=3.95))
    # entry bar
    bars.append(PriceBar(symbol, _ENTRY, 3.9, 4.2, 3.8, 4.0, volume=150.0, trade_count=15, vwap=4.05))
    # forward bars — option value declines (profitable short)
    for offset in range(1, 4):
        close = 4.0 - offset * 0.8
        day = _ENTRY + timedelta(days=offset)
        bars.append(PriceBar(symbol, day, close - 0.1, close + 0.1, close - 0.2, close, volume=60.0, trade_count=6))
    return bars


class FullDataProvider:
    """Covers all three provider protocols with complete data."""

    _SYMBOL = "SPY260626P00500000"

    def get_stock_bars(self, symbols, start, end, timeframe):
        spy = _make_spy_bars()
        vix = _make_vix_bars()
        result = {}
        for sym in symbols:
            if sym == "SPY":
                result[sym] = spy
            elif sym == "VIX":
                result[sym] = vix
        return result

    def get_option_contracts(self, underlyings, expiration_gte=None, expiration_lte=None,
                             status="inactive", limit=None):
        return [
            OptionContract(
                symbol=self._SYMBOL,
                underlying="SPY",
                expiration=_EXPIRY,
                strike=_STRIKE,
                option_type="put",
                status=status,
                source="test",
            )
        ]

    def get_option_bars(self, symbols, start, end, timeframe, limit=None):
        return {sym: _make_option_bars(sym) for sym in symbols}


class FullEventProvider:
    """Returns one upcoming earnings event within the 30-day forward window."""

    def get_earnings_calendar(self, symbols, start, end):
        return {
            "SPY": [
                EarningsEvent(
                    symbol="SPY",
                    report_date=date(2026, 6, 1),   # 18 calendar days from entry
                    fiscal_period="Q1-2026",
                    source="test",
                )
            ]
        }


# ---------------------------------------------------------------------------
# Completeness validation test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def complete_row():
    """Build one row with every data source fully populated."""
    provider = FullDataProvider()
    builder = HistoricalCandidateDatasetBuilder(
        provider, provider, provider,
        event_provider=FullEventProvider(),
    )
    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=_ENTRY,
            entry_end=_ENTRY + timedelta(days=1),
            max_contracts_per_underlying=1,
            min_dte=7,
            max_dte=60,
            forward_days=30,
        )
    )
    assert rows, "Builder produced no rows — check provider data or DTE filter"
    return rows[0]


class TestContractMetadataSource:
    """Source: OptionContractProvider.get_option_contracts()"""

    def test_option_symbol(self, complete_row):
        assert complete_row.option_symbol == FullDataProvider._SYMBOL

    def test_strike(self, complete_row):
        assert complete_row.strike == _STRIKE

    def test_option_type(self, complete_row):
        assert complete_row.option_type == "put"

    def test_expiration(self, complete_row):
        assert complete_row.expiration == _EXPIRY

    def test_dte(self, complete_row):
        assert complete_row.dte == 43

    def test_moneyness(self, complete_row):
        # moneyness = underlying_close / strike
        assert complete_row.moneyness is not None
        assert complete_row.moneyness == pytest.approx(_UNDERLYING_CLOSE / _STRIKE, rel=1e-5)

    def test_strike_distance_pct(self, complete_row):
        # strike_distance_pct = (strike - underlying_close) / underlying_close
        expected = (_STRIKE - _UNDERLYING_CLOSE) / _UNDERLYING_CLOSE
        assert complete_row.strike_distance_pct == pytest.approx(expected, rel=1e-5)


class TestUnderlyingPriceSource:
    """Source: MarketDataProvider.get_stock_bars() for the underlying symbol"""

    def test_underlying_close(self, complete_row):
        assert complete_row.underlying_close == _UNDERLYING_CLOSE

    def test_return_1d(self, complete_row):
        assert complete_row.underlying_return_1d is not None

    def test_return_3d(self, complete_row):
        assert complete_row.underlying_return_3d is not None

    def test_return_5d(self, complete_row):
        assert complete_row.underlying_return_5d is not None

    def test_return_20d(self, complete_row):
        assert complete_row.underlying_return_20d is not None

    def test_realized_vol_5d(self, complete_row):
        assert complete_row.underlying_realized_vol_5d is not None
        assert complete_row.underlying_realized_vol_5d > 0

    def test_realized_vol_10d(self, complete_row):
        assert complete_row.underlying_realized_vol_10d is not None
        assert complete_row.underlying_realized_vol_10d > 0

    def test_realized_vol_20d(self, complete_row):
        assert complete_row.underlying_realized_vol_20d is not None
        assert complete_row.underlying_realized_vol_20d > 0

    def test_sma_20_distance_pct(self, complete_row):
        assert complete_row.underlying_sma_20_distance_pct is not None

    def test_above_sma_20(self, complete_row):
        assert complete_row.underlying_above_sma_20 in (0, 1)

    def test_volatility_ratio(self, complete_row):
        assert complete_row.underlying_volatility_ratio_5d_20d is not None
        assert complete_row.underlying_volatility_ratio_5d_20d > 0

    def test_skew_5d(self, complete_row):
        assert complete_row.underlying_skew_5d is not None


class TestMarketRegimeSource:
    """Source: same get_stock_bars() call, market_regime_symbol key (SPY)"""

    def test_regime_symbol(self, complete_row):
        assert complete_row.market_regime_symbol == "SPY"

    def test_market_return_5d(self, complete_row):
        assert complete_row.market_return_5d is not None

    def test_market_return_20d(self, complete_row):
        assert complete_row.market_return_20d is not None

    def test_market_realized_vol_5d(self, complete_row):
        assert complete_row.market_realized_vol_5d is not None

    def test_market_realized_vol_20d(self, complete_row):
        assert complete_row.market_realized_vol_20d is not None

    def test_market_trend_regime(self, complete_row):
        assert complete_row.market_trend_regime in ("uptrend", "downtrend", "sideways")

    def test_market_volatility_regime(self, complete_row):
        assert complete_row.market_volatility_regime in ("low", "normal", "high")


class TestOptionEntryBarSource:
    """Source: OptionPriceProvider.get_option_bars() — the entry bar"""

    def test_entry_price(self, complete_row):
        assert complete_row.option_entry_price == pytest.approx(4.0, rel=1e-4)

    def test_entry_volume(self, complete_row):
        assert complete_row.option_entry_volume == 150.0

    def test_entry_trade_count(self, complete_row):
        assert complete_row.option_entry_trade_count == 15

    def test_entry_vwap(self, complete_row):
        assert complete_row.option_entry_vwap == pytest.approx(4.05, rel=1e-4)

    def test_entry_range_pct(self, complete_row):
        # (4.2 - 3.8) / 4.0 = 0.10
        assert complete_row.option_entry_range_pct == pytest.approx(0.10, rel=1e-4)


class TestOptionLookbackSource:
    """Source: same get_option_bars() — bars in the 5-day window BEFORE entry"""

    def test_volume_5d_avg(self, complete_row):
        # 5 bars at entry-5 to entry-1, each volume=80
        assert complete_row.option_volume_5d_avg == pytest.approx(80.0, rel=1e-4)

    def test_trade_count_5d_avg(self, complete_row):
        # 5 bars at entry-5 to entry-1, each trade_count=8
        assert complete_row.option_trade_count_5d_avg == pytest.approx(8.0, rel=1e-4)


class TestBlackScholesSource:
    """Source: COMPUTED — Newton-Raphson IV + analytic B-S Greeks.
    No API call; inputs come from option entry price, underlying close,
    contract strike/DTE/type, and risk_free_rate config.
    """

    def test_implied_volatility(self, complete_row):
        assert complete_row.implied_volatility is not None
        assert 0.01 < complete_row.implied_volatility < 5.0

    def test_put_delta_range(self, complete_row):
        # Put delta in (-1, 0)
        assert complete_row.option_delta is not None
        assert -1.0 < complete_row.option_delta < 0.0

    def test_gamma_positive(self, complete_row):
        assert complete_row.option_gamma is not None
        assert complete_row.option_gamma > 0

    def test_theta_negative(self, complete_row):
        # Theta ($/day) is negative for long options — short sellers receive it
        assert complete_row.option_theta is not None
        assert complete_row.option_theta < 0

    def test_vega_positive(self, complete_row):
        # Vega (per 1% IV move) is positive
        assert complete_row.option_vega is not None
        assert complete_row.option_vega > 0

    def test_iv_vs_hv5d(self, complete_row):
        assert complete_row.iv_vs_hv5d is not None
        assert complete_row.iv_vs_hv5d > 0

    def test_iv_vs_hv20d(self, complete_row):
        assert complete_row.iv_vs_hv20d is not None
        assert complete_row.iv_vs_hv20d > 0


class TestVixSource:
    """Source: MarketDataProvider.get_stock_bars() for vix_symbol.
    NOTE: Polygon/Massive exposes the CBOE VIX index as ticker 'I:VIX',
    not bare 'VIX'. Set vix_symbol='I:VIX' in CandidateDatasetConfig
    when using MassiveProvider in production.
    """

    def test_vix_close(self, complete_row):
        assert complete_row.vix_close == _VIX_CLOSE

    def test_vix_return_5d(self, complete_row):
        assert complete_row.vix_return_5d is not None

    def test_vix_realized_vol_5d(self, complete_row):
        assert complete_row.vix_realized_vol_5d is not None
        assert complete_row.vix_realized_vol_5d > 0

    def test_vix_above_20(self, complete_row):
        # VIX close is 22.0, so above_20 = 1
        assert complete_row.vix_above_20 == 1

    def test_vix_above_30(self, complete_row):
        # VIX close is 22.0, so above_30 = 0
        assert complete_row.vix_above_30 == 0


class TestEventSource:
    """Source: EventDataProvider.get_earnings_calendar().
    WARNING: MassiveProvider.get_earnings_calendar() is currently a stub
    that always returns {}. These fields will be None/0 in real builds
    until an earnings data source is wired in.
    """

    def test_days_to_earnings(self, complete_row):
        # EarningsEvent on 2026-06-01, entry 2026-05-14 → 18 days
        assert complete_row.days_to_earnings == 18

    def test_has_earnings_in_forward_days(self, complete_row):
        # 18 days < 30-day forward window → in window
        assert complete_row.has_earnings_in_forward_days == 1


class TestMissingFieldsCompleteness:
    """When all data sources are wired, missing_fields should be empty."""

    def test_no_missing_fields(self, complete_row):
        assert complete_row.missing_fields == (), (
            f"Unexpected missing fields: {complete_row.missing_fields}"
        )
