from datetime import UTC, date, datetime, timedelta

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.providers.models import OptionContract, PriceBar


class FakeProvider:
    def __init__(self):
        self.entry = datetime(2026, 5, 14, tzinfo=UTC)

    def get_stock_bars(self, symbols, start, end, timeframe):
        bars = []
        for offset in range(21, 0, -1):
            day = self.entry - timedelta(days=offset)
            close = 501 - offset
            bars.append(PriceBar("SPY", day, close - 1, close + 1, close - 2, close, volume=1000 + offset))
        bars.append(PriceBar("SPY", self.entry, 500, 505, 499, 504, volume=1200))
        return {
            "SPY": bars
        }

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

    assert len(rows) == 1
    row = rows[0]
    assert row.option_symbol == "SPY260626P00500000"
    assert row.dte == 43
    assert row.underlying_close == 504
    assert row.underlying_return_1d == 0.008
    assert row.underlying_return_5d == 0.01612903
    assert row.underlying_realized_vol_5d is not None
    assert row.strike_distance_pct == -0.00793651
    assert row.moneyness == 1.008
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
    assert row.missing_fields == ()


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
