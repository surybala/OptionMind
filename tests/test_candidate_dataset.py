from datetime import UTC, date, datetime, timedelta

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.providers.models import OptionContract, PriceBar


class FakeProvider:
    def __init__(self):
        self.entry = datetime(2026, 5, 14, tzinfo=UTC)

    def get_stock_bars(self, symbols, start, end, timeframe):
        spy_bars = []
        vix_bars = []
        for offset in range(21, 0, -1):
            day = self.entry - timedelta(days=offset)
            close = 501 - offset
            spy_bars.append(PriceBar("SPY", day, close - 1, close + 1, close - 2, close, volume=1000 + offset))
            vix_bars.append(PriceBar("I:VIX", day, 17.0, 18.0, 16.0, 17.0 + offset * 0.05))
        spy_bars.append(PriceBar("SPY", self.entry, 500, 505, 499, 504, volume=1200))
        vix_bars.append(PriceBar("I:VIX", self.entry, 18.0, 19.0, 17.0, 18.5))
        result = {}
        for symbol in symbols:
            if symbol == "SPY":
                result[symbol] = spy_bars
            elif symbol == "I:VIX":
                result[symbol] = vix_bars
        return result

    def get_option_contracts(
        self,
        underlyings,
        expiration_gte=None,
        expiration_lte=None,
        status="inactive",
        limit=None,
    ):
        return [
            OptionContract(
                symbol="SPY260626P00500000",
                underlying="SPY",
                expiration=date(2026, 6, 26),
                strike=500.0,
                option_type="put",
                status=status,
                source="fake",
            ),
            OptionContract(
                symbol="SPY260626C00550000",
                underlying="SPY",
                expiration=date(2026, 6, 26),
                strike=550.0,
                option_type="call",
                status=status,
                source="fake",
            ),
        ]

    def get_option_bars(self, symbols, start, end, timeframe, limit=None):
        data = {}
        for symbol in symbols:
            data[symbol] = [
                PriceBar(symbol, self.entry, 4.00, 4.20, 3.90, 4.00, volume=100, trade_count=12, vwap=4.05),
                PriceBar(symbol, self.entry + timedelta(days=1), 2.40, 2.50, 2.00, 2.20, volume=90, trade_count=10, vwap=2.25),
                PriceBar(symbol, self.entry + timedelta(days=2), 1.90, 2.00, 1.80, 1.90, volume=80, trade_count=8, vwap=1.92),
            ]
        return data

    def get_option_trades(self, symbols, start, end, limit=None):
        return {}


class StopLossProvider(FakeProvider):
    def get_option_bars(self, symbols, start, end, timeframe, limit=None):
        return {
            symbol: [
                PriceBar(symbol, self.entry, 4.00, 4.20, 3.90, 4.00, volume=100, trade_count=12, vwap=4.05),
                PriceBar(symbol, self.entry + timedelta(days=1), 8.30, 8.50, 8.10, 8.30, volume=90, trade_count=9, vwap=8.32),
            ]
            for symbol in symbols
        }


def test_candidate_dataset_builder_emits_profit_take_rows():
    provider = FakeProvider()
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=1),
            max_contracts_per_underlying=1,
        )
    )

    assert len(rows) == 2
    row = rows[0]
    assert row.option_symbol == "SPY260626P00500000"
    assert row.dte == 43
    assert row.underlying_close == 504
    assert row.underlying_return_1d == 0.008
    assert row.underlying_return_5d == 0.01612903
    assert row.underlying_return_20d == 0.04781705
    assert row.underlying_realized_vol_5d is not None
    assert row.underlying_sma_20_distance_pct == 0.0251195
    assert row.underlying_above_sma_20 == 1
    assert row.underlying_volatility_ratio_5d_20d is not None
    assert row.strike_distance_pct == -0.00793651
    assert row.moneyness == 1.008
    assert row.market_regime_symbol == "SPY"
    assert row.market_return_5d == 0.01612903
    assert row.market_return_20d == 0.04781705
    assert row.market_sma_20_distance_pct == 0.0251195
    assert row.market_above_sma_20 == 1
    assert row.market_trend_regime == "uptrend"
    assert row.market_volatility_regime is not None
    assert row.option_entry_range_pct == 0.075
    assert row.option_entry_trade_count == 12
    assert row.option_entry_vwap == 4.05
    assert row.exit_reason == "profit_take"
    assert row.expected_pnl == 210.0
    assert row.realized_pnl_per_contract == 210.0
    assert row.days_to_exit == 2.0
    assert row.label_version == "short_option_labels_v001"
    assert row.profit_label == 1
    assert row.stop_loss_hit == 0
    # FakeProvider has no pre-entry option bars (lookback) and no event provider,
    # so those features are legitimately absent; VIX is now provided.
    assert "option_volume_5d_avg" in row.missing_fields
    assert "option_trade_count_5d_avg" in row.missing_fields
    assert "days_to_earnings" in row.missing_fields
    assert "underlying_close" not in row.missing_fields
    assert "implied_volatility" not in row.missing_fields
    assert "vix_close" not in row.missing_fields
    # New feature spot-checks
    assert row.implied_volatility is not None
    assert row.option_delta is not None
    assert row.option_gamma is not None
    assert row.option_theta is not None
    assert row.option_vega is not None
    assert row.iv_vs_hv5d is not None
    assert row.iv_vs_hv20d is not None
    assert row.vix_close == 18.5
    assert row.vix_above_20 == 0
    assert row.vix_above_30 == 0
    assert row.vix_return_5d is not None
    assert row.underlying_realized_vol_10d is not None
    assert row.underlying_return_3d is not None
    assert row.underlying_skew_5d is not None
    assert row.has_earnings_in_forward_days == 0
    assert rows[1].entry_timestamp == provider.entry + timedelta(days=1)
    assert rows[1].exit_reason == "horizon"


def test_candidate_dataset_builder_labels_stop_loss_rows():
    provider = StopLossProvider()
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=1),
            max_contracts_per_underlying=1,
        )
    )

    assert rows[0].exit_reason == "stop_loss"
    assert rows[0].stop_loss_hit == 1
    assert rows[0].profit_label == 0
    assert rows[0].realized_pnl_per_contract == -430.0
    assert rows[0].expected_pnl == -430.0
    assert rows[0].days_to_exit == 1.0


def test_candidate_dataset_builder_skips_contracts_without_option_bars():
    provider = FakeProvider()
    provider.get_option_bars = lambda symbols, start, end, timeframe, limit=None: {}
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=1),
        )
    )

    assert rows == []


def test_candidate_dataset_builder_enforces_actual_entry_window_and_dte():
    provider = FakeProvider()
    provider.get_option_bars = lambda symbols, start, end, timeframe, limit=None: {
        symbol: [
            PriceBar(symbol, provider.entry + timedelta(days=10), 4.0, 4.2, 3.9, 4.0, volume=100)
        ]
        for symbol in symbols
    }
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=1),
            max_contracts_per_underlying=1,
        )
    )

    assert rows == []


def test_candidate_dataset_builder_uses_separate_contracts_per_window():
    """Windows use their own reference close and expiration range."""
    calls: list[dict] = []
    provider = FakeProvider()
    original_get_contracts = provider.get_option_contracts

    def tracking_get_contracts(underlyings, expiration_gte=None, expiration_lte=None, status="inactive", limit=None):
        calls.append({"expiration_gte": expiration_gte, "expiration_lte": expiration_lte})
        return original_get_contracts(underlyings, expiration_gte=expiration_gte, expiration_lte=expiration_lte, status=status, limit=limit)

    provider.get_option_contracts = tracking_get_contracts
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    # Two windows: [entry, entry+10days] and [entry+10days, entry+20days]
    builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=20),
            build_window_days=10,
            max_contracts_per_underlying=1,
        )
    )

    assert len(calls) == 2
    assert calls[0]["expiration_lte"] < calls[1]["expiration_lte"]


def test_candidate_dataset_builder_no_duplicate_rows_across_windows():
    """Entry bars are not double-counted when windows share a boundary."""
    provider = FakeProvider()
    builder = HistoricalCandidateDatasetBuilder(provider, provider, provider)

    rows = builder.build(
        CandidateDatasetConfig(
            underlyings=["SPY"],
            entry_start=provider.entry,
            entry_end=provider.entry + timedelta(days=1),
            build_window_days=1,
            max_contracts_per_underlying=2,
        )
    )

    timestamps = [r.entry_timestamp for r in rows]
    assert len(timestamps) == len(set(str(t) + r.option_symbol for t, r in zip(timestamps, rows)))


# ---------------------------------------------------------------------------
# New feature tests
# ---------------------------------------------------------------------------

def test_implied_volatility_and_greeks_computed():
    """Black-Scholes Greeks and IV are non-None when data is available."""
    from ml.datasets.candidate_dataset import (
        _implied_volatility,
        _black_scholes_greeks,
        _option_greeks_features,
    )
    from ml.providers.models import PriceBar
    from datetime import UTC, datetime

    # Simple sanity check on the B-S solver
    iv = _implied_volatility(
        market_price=4.0,
        S=504.0,
        K=500.0,
        T=43 / 365.0,
        r=0.045,
        option_type="put",
    )
    assert iv is not None
    assert 0.0 < iv < 2.0  # should be a reasonable IV

    # Greeks should compute from a valid IV
    greeks = _black_scholes_greeks(504.0, 500.0, 43 / 365.0, 0.045, iv, "put")
    assert greeks["delta"] is not None
    assert -1.0 <= greeks["delta"] <= 0.0  # put delta in (-1, 0)
    assert greeks["gamma"] is not None and greeks["gamma"] > 0
    assert greeks["theta"] is not None and greeks["theta"] < 0  # theta negative for long options
    assert greeks["vega"] is not None and greeks["vega"] > 0

    # Full feature dict
    entry_bar = PriceBar("OPT", datetime(2026, 5, 14, tzinfo=UTC), 3.9, 4.2, 3.8, 4.0, volume=100, trade_count=10)
    feats = _option_greeks_features(entry_bar, 504.0, 500.0, "put", 43, 0.045, 0.15, 0.12)
    assert feats["implied_volatility"] is not None
    assert feats["iv_vs_hv5d"] is not None
    assert feats["iv_vs_hv20d"] is not None


def test_implied_volatility_returns_none_for_invalid_inputs():
    """IV solver gracefully returns None for degenerate inputs."""
    from ml.datasets.candidate_dataset import _implied_volatility

    assert _implied_volatility(0.0, 504.0, 500.0, 0.118, 0.045, "put") is None  # zero price
    assert _implied_volatility(4.0, 504.0, 500.0, 0.0, 0.045, "put") is None   # zero time


def test_vix_features_computed():
    """VIX features compute correctly from a history of bars."""
    from ml.datasets.candidate_dataset import _vix_features
    from ml.providers.models import PriceBar
    from datetime import UTC, datetime, timedelta

    entry = datetime(2026, 5, 14, tzinfo=UTC)
    vix_bars = [
        PriceBar("VIX", entry - timedelta(days=d), 18.0, 19.0, 17.0, 18.0 + d * 0.1)
        for d in range(10, 0, -1)
    ]
    vix_bars.append(PriceBar("VIX", entry, 18.0, 20.0, 17.5, 22.0))  # spike above 20

    feats = _vix_features(vix_bars, entry)
    assert feats["vix_close"] == 22.0
    assert feats["vix_above_20"] == 1
    assert feats["vix_above_30"] == 0
    assert feats["vix_return_5d"] is not None
    assert feats["vix_realized_vol_5d"] is not None


def test_vix_features_empty_bars():
    """VIX features return all None when no bars available."""
    from ml.datasets.candidate_dataset import _vix_features
    from datetime import UTC, datetime

    feats = _vix_features([], datetime(2026, 5, 14, tzinfo=UTC))
    assert all(v is None for v in feats.values())


def test_event_features_with_upcoming_earnings():
    """Event features correctly identify nearest upcoming earnings."""
    from datetime import UTC, date, datetime
    from ml.datasets.candidate_dataset import _event_features
    from ml.providers.models import EarningsEvent

    entry = datetime(2026, 5, 14, tzinfo=UTC)
    events = [
        EarningsEvent(symbol="SPY", report_date=date(2026, 6, 1)),   # 18 days out, in 30d window
        EarningsEvent(symbol="SPY", report_date=date(2026, 7, 1)),   # outside window
    ]
    feats = _event_features(events, entry, forward_days=30)
    assert feats["days_to_earnings"] == 18
    assert feats["has_earnings_in_forward_days"] == 1


def test_event_features_no_events():
    """Event features return 0 flag and None days when no events."""
    from datetime import UTC, datetime
    from ml.datasets.candidate_dataset import _event_features

    feats = _event_features([], datetime(2026, 5, 14, tzinfo=UTC), forward_days=30)
    assert feats["days_to_earnings"] is None
    assert feats["has_earnings_in_forward_days"] == 0


def test_option_lookback_features():
    """Lookback averages computed from pre-entry bars."""
    from datetime import UTC, datetime, timedelta
    from ml.datasets.candidate_dataset import _option_lookback_features
    from ml.providers.models import PriceBar

    entry = datetime(2026, 5, 14, tzinfo=UTC)
    path = [
        PriceBar("OPT", entry - timedelta(days=3), 4.0, 4.2, 3.8, 4.0, volume=100.0, trade_count=10),
        PriceBar("OPT", entry - timedelta(days=2), 4.1, 4.3, 3.9, 4.1, volume=120.0, trade_count=12),
        PriceBar("OPT", entry - timedelta(days=1), 4.2, 4.4, 4.0, 4.2, volume=110.0, trade_count=11),
        PriceBar("OPT", entry, 4.3, 4.5, 4.1, 4.3, volume=200.0, trade_count=20),  # entry bar (excluded)
    ]
    feats = _option_lookback_features(path, entry, lookback_days=5)
    # Average of [100, 120, 110] = 110
    assert abs(feats["option_volume_5d_avg"] - 110.0) < 0.01
    assert abs(feats["option_trade_count_5d_avg"] - 11.0) < 0.01


def test_skewness():
    """Skewness returns None for < 3 obs and a signed value for valid series."""
    from ml.datasets.candidate_dataset import _skewness

    assert _skewness([]) is None
    assert _skewness([0.01, 0.02]) is None  # too few

    # Positively skewed series
    skew = _skewness([0.01, 0.01, 0.01, 0.10])
    assert skew is not None and skew > 0

    # Negatively skewed series
    skew_neg = _skewness([-0.10, -0.01, -0.01, -0.01])
    assert skew_neg is not None and skew_neg < 0
