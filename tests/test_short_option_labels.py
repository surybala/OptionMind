from datetime import UTC, datetime, timedelta

import pytest

from ml.labels import CreditSpreadLabelConfig, ShortOptionLabelConfig, label_credit_spread_path, label_short_option_path
from ml.providers.models import PriceBar


def _bar(symbol: str, timestamp: datetime, close: float) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
    )


def test_short_option_labeler_exits_on_profit_take():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    entry_bar = _bar("SPY260626P00500000", entry, 4.0)
    path = [
        entry_bar,
        _bar(entry_bar.symbol, entry + timedelta(days=1), 2.2),
        _bar(entry_bar.symbol, entry + timedelta(days=2), 1.9),
    ]

    label = label_short_option_path(entry_bar, path)

    assert label.exit_reason == "profit_take"
    assert label.exit_timestamp == entry + timedelta(days=2)
    assert label.expected_pnl == 210.0
    assert label.realized_pnl_per_contract == 210.0
    assert label.profit_label == 1
    assert label.stop_loss_hit == 0
    assert label.large_loss_label == 0
    assert label.max_favorable_excursion == 210.0
    assert label.days_to_exit == 2.0


def test_short_option_labeler_exits_on_stop_loss_before_later_recovery():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    entry_bar = _bar("SPY260626P00500000", entry, 4.0)
    path = [
        entry_bar,
        _bar(entry_bar.symbol, entry + timedelta(days=1), 8.3),
        _bar(entry_bar.symbol, entry + timedelta(days=2), 1.0),
    ]

    label = label_short_option_path(entry_bar, path)

    assert label.exit_reason == "stop_loss"
    assert label.exit_timestamp == entry + timedelta(days=1)
    assert label.expected_pnl == -430.0
    assert label.stop_loss_hit == 1
    assert label.large_loss_label == 1
    assert label.max_adverse_excursion == 430.0
    assert label.days_to_exit == 1.0


def test_short_option_labeler_uses_horizon_when_no_exit_rule_hits():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    entry_bar = _bar("SPY260626P00500000", entry, 4.0)
    path = [
        entry_bar,
        _bar(entry_bar.symbol, entry + timedelta(days=1), 3.5),
        _bar(entry_bar.symbol, entry + timedelta(days=2), 3.0),
    ]

    label = label_short_option_path(entry_bar, path)

    assert label.exit_reason == "horizon"
    assert label.exit_timestamp == entry + timedelta(days=2)
    assert label.expected_pnl == 100.0
    assert label.profit_label == 1
    assert label.max_favorable_excursion == 100.0


def test_short_option_labeler_accepts_forward_path_without_entry_bar():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    entry_bar = _bar("SPY260626P00500000", entry, 4.0)
    path = [
        _bar(entry_bar.symbol, entry + timedelta(days=1), 1.9),
    ]

    label = label_short_option_path(entry_bar, path)

    assert label.exit_reason == "profit_take"
    assert label.expected_pnl == 210.0
    assert label.days_to_exit == 1.0


def test_short_option_labeler_rejects_negative_prices():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    entry_bar = _bar("SPY260626P00500000", entry, -1.0)

    with pytest.raises(ValueError, match="entry option price"):
        label_short_option_path(entry_bar, [])


def test_short_option_labeler_validates_exit_config():
    with pytest.raises(ValueError, match="profit_take_pct"):
        ShortOptionLabelConfig(profit_take_pct=1.0)

    with pytest.raises(ValueError, match="stop_loss_multiple"):
        ShortOptionLabelConfig(stop_loss_multiple=1.0)


def test_credit_spread_labeler_exits_pcs_on_profit_take():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    short_entry = _bar("SPY260626P00500000", entry, 4.0)
    long_entry = _bar("SPY260626P00495000", entry, 1.0)

    label = label_credit_spread_path(
        strategy="PCS",
        short_entry_bar=short_entry,
        long_entry_bar=long_entry,
        short_path=[
            short_entry,
            _bar(short_entry.symbol, entry + timedelta(days=1), 3.0),
            _bar(short_entry.symbol, entry + timedelta(days=2), 2.2),
        ],
        long_path=[
            long_entry,
            _bar(long_entry.symbol, entry + timedelta(days=1), 1.0),
            _bar(long_entry.symbol, entry + timedelta(days=2), 0.8),
        ],
    )

    assert label.strategy == "PCS"
    assert label.entry_credit == 3.0
    assert label.exit_debit == 1.4
    assert label.exit_reason == "profit_take"
    assert label.expected_pnl == 160.0
    assert label.profit_label == 1
    assert label.max_favorable_excursion == 160.0


def test_credit_spread_labeler_exits_ccs_on_stop_loss():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    short_entry = _bar("SPY260626C00510000", entry, 3.0)
    long_entry = _bar("SPY260626C00515000", entry, 1.0)

    label = label_credit_spread_path(
        strategy="CCS",
        short_entry_bar=short_entry,
        long_entry_bar=long_entry,
        short_path=[
            short_entry,
            _bar(short_entry.symbol, entry + timedelta(days=1), 6.2),
            _bar(short_entry.symbol, entry + timedelta(days=2), 1.0),
        ],
        long_path=[
            long_entry,
            _bar(long_entry.symbol, entry + timedelta(days=1), 1.8),
            _bar(long_entry.symbol, entry + timedelta(days=2), 0.2),
        ],
    )

    assert label.strategy == "CCS"
    assert label.entry_credit == 2.0
    assert label.exit_debit == 4.4
    assert label.exit_reason == "stop_loss"
    assert label.expected_pnl == -240.0
    assert label.stop_loss_hit == 1
    assert label.spread_width == 5.0
    assert label.max_loss == 300.0
    assert label.large_loss_label == 0


def test_credit_spread_labeler_validates_executable_credit():
    entry = datetime(2026, 5, 14, tzinfo=UTC)
    short_entry = _bar("SPY260626P00500000", entry, 1.0)
    long_entry = _bar("SPY260626P00495000", entry, 1.2)

    with pytest.raises(ValueError, match="entry credit"):
        label_credit_spread_path(
            strategy="PCS",
            short_entry_bar=short_entry,
            long_entry_bar=long_entry,
            short_path=[],
            long_path=[],
        )

    with pytest.raises(ValueError, match="profit_take_pct"):
        CreditSpreadLabelConfig(profit_take_pct=1.0)
