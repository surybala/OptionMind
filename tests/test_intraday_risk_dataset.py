import json
from datetime import UTC, date, datetime, timedelta

from ml.datasets.intraday_risk_dataset import (
    IntradayRiskDatasetBuilder,
    IntradayRiskDatasetConfig,
)
from ml.providers.models import PriceBar


class FakeIntradayProvider:
    def __init__(self, stock_bars, option_bars):
        self.stock_bars = stock_bars
        self.option_bars = option_bars

    def get_stock_bars(self, symbols, start, end, timeframe):
        return {
            symbol: [bar for bar in self.stock_bars.get(symbol, []) if start <= bar.timestamp <= end]
            for symbol in symbols
        }

    def get_option_bars(self, symbols, start, end, timeframe, limit=None):
        return {
            symbol: [bar for bar in self.option_bars.get(symbol, []) if start <= bar.timestamp <= end]
            for symbol in symbols
        }


def _bar(symbol, ts, close, volume=100, trade_count=10):
    return PriceBar(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        trade_count=trade_count,
        vwap=close,
        source="fake",
    )


def test_intraday_risk_builder_emits_state_rows(tmp_path):
    entry = datetime(2026, 5, 14, 14, 0, tzinfo=UTC)
    timestamps = [entry + timedelta(minutes=offset) for offset in (0, 5, 10, 15)]
    provider = FakeIntradayProvider(
        stock_bars={
            "SPY": [
                _bar("SPY", entry - timedelta(minutes=15), 500.0),
                _bar("SPY", timestamps[0], 501.0),
                _bar("SPY", timestamps[1], 502.0),
                _bar("SPY", timestamps[2], 503.0),
                _bar("SPY", timestamps[3], 501.5),
            ]
        },
        option_bars={
            "SPY260626P00500000": [
                _bar("SPY260626P00500000", timestamps[0], 3.00),
                _bar("SPY260626P00500000", timestamps[1], 3.50),
                _bar("SPY260626P00500000", timestamps[2], 4.20),
                _bar("SPY260626P00500000", timestamps[3], 4.50),
            ],
            "SPY260626P00495000": [
                _bar("SPY260626P00495000", timestamps[0], 1.00),
                _bar("SPY260626P00495000", timestamps[1], 1.05),
                _bar("SPY260626P00495000", timestamps[2], 1.10),
                _bar("SPY260626P00495000", timestamps[3], 1.15),
            ],
        },
    )

    seed_row = {
        "entry_timestamp": entry.isoformat(),
        "exit_timestamp": timestamps[-1].isoformat(),
        "underlying": "SPY",
        "strategy": "PCS",
        "market_regime_symbol": "SPY",
        "market_trend_regime": "downtrend",
        "market_volatility_regime": "high",
        "short_option_symbol": "SPY260626P00500000",
        "long_option_symbol": "SPY260626P00495000",
        "spread_width": 5.0,
        "entry_credit": 2.0,
        "expiration": date(2026, 6, 26).isoformat(),
        "source": "fake",
    }
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(json.dumps(seed_row) + "\n", encoding="utf-8")

    rows = IntradayRiskDatasetBuilder(provider, provider).build(
        seed_path,
        IntradayRiskDatasetConfig(
            sample_every_n_minutes=1,
            max_workers=1,
            stop_loss_multiple=1.2,
            stop_loss_max_loss_pct=None,
            min_state_rows_per_candidate=2,
        ),
    )

    assert len(rows) >= 2
    first = rows[0]
    assert first.underlying == "SPY"
    assert first.current_debit == 2.0
    assert first.stop_loss_hit_5m == 1
    assert first.future_worst_debit_15m is not None
    assert first.intraday_exit_reason == "stop_loss"
    assert first.market_regime_symbol == "SPY"
    assert first.market_trend_regime == "downtrend"
    assert first.market_volatility_regime == "high"


def test_intraday_risk_builder_aligns_sparse_leg_and_stock_timestamps(tmp_path):
    entry = datetime(2026, 5, 14, 4, 0, tzinfo=UTC)
    provider = FakeIntradayProvider(
        stock_bars={
            "SPY": [
                _bar("SPY", datetime(2026, 5, 14, 13, 30, tzinfo=UTC), 500.0),
                _bar("SPY", datetime(2026, 5, 14, 13, 35, tzinfo=UTC), 500.5),
                _bar("SPY", datetime(2026, 5, 14, 13, 40, tzinfo=UTC), 501.0),
            ]
        },
        option_bars={
            "SPY260626P00500000": [
                _bar("SPY260626P00500000", datetime(2026, 5, 14, 13, 31, tzinfo=UTC), 3.00),
                _bar("SPY260626P00500000", datetime(2026, 5, 14, 13, 36, tzinfo=UTC), 3.40),
            ],
            "SPY260626P00495000": [
                _bar("SPY260626P00495000", datetime(2026, 5, 14, 13, 34, tzinfo=UTC), 1.00),
                _bar("SPY260626P00495000", datetime(2026, 5, 14, 13, 39, tzinfo=UTC), 1.05),
            ],
        },
    )
    seed_row = {
        "entry_timestamp": entry.isoformat(),
        "exit_timestamp": datetime(2026, 5, 14, 13, 40, tzinfo=UTC).isoformat(),
        "underlying": "SPY",
        "strategy": "PCS",
        "short_option_symbol": "SPY260626P00500000",
        "long_option_symbol": "SPY260626P00495000",
        "spread_width": 5.0,
        "entry_credit": 2.0,
        "expiration": date(2026, 6, 26).isoformat(),
        "source": "fake",
    }
    seed_path = tmp_path / "seed_sparse.jsonl"
    seed_path.write_text(json.dumps(seed_row) + "\n", encoding="utf-8")

    rows = IntradayRiskDatasetBuilder(provider, provider).build(
        seed_path,
        IntradayRiskDatasetConfig(
            sample_every_n_minutes=1,
            max_workers=1,
            min_state_rows_per_candidate=1,
            stop_loss_multiple=2.0,
            stop_loss_max_loss_pct=None,
        ),
    )

    assert len(rows) >= 1
    assert rows[0].entry_timestamp == datetime(2026, 5, 14, 13, 34, tzinfo=UTC)
