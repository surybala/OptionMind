import datetime
import gc
import os
import sqlite3
import tempfile
import sys
from unittest.mock import MagicMock

_alpaca_mock = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.trading',
    'alpaca.trading.client',
    'alpaca.trading.enums',
    'alpaca.trading.requests',
    'numpy',
    'pandas',
    'scipy',
    'scipy.stats',
    'yfinance',
]:
    sys.modules.setdefault(_mod, _alpaca_mock)

from agent import _reconcile_positions_before_budget
from src.database import TradeDatabase


class _FakeAlpacaClient:
    def get_all_positions(self):
        return []


class _FakeExecutor:
    is_logged_in = True
    client = _FakeAlpacaClient()


def _make_db():
    fd, path = tempfile.mkstemp(
        suffix='.db', prefix='test_agent_reconcile_', dir=tempfile.gettempdir()
    )
    os.close(fd)
    os.remove(path)
    return TradeDatabase(db_path=path), path


def _cleanup(db, db_path):
    del db
    gc.collect()
    if os.path.exists(db_path):
        os.remove(db_path)


def test_agent_reconciles_externally_closed_positions_before_budget_snapshot():
    db, db_path = _make_db()
    try:
        future_expiry = (
            datetime.date.today() + datetime.timedelta(days=14)
        ).isoformat()
        trade_id = db.log_trade(
            'AAPL',
            future_expiry,
            150.0,
            'PCS',
            0.50,
            0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        old_ts = (
            datetime.datetime.now() - datetime.timedelta(minutes=30)
        ).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE trades SET timestamp = ? WHERE id = ?",
                (old_ts, trade_id),
            )
            conn.commit()

        assert len(db.get_open_positions()) == 1

        _reconcile_positions_before_budget(
            db,
            _FakeExecutor(),
            {'monitor_schedule': {'ghost_grace_minutes': 10}},
        )

        assert db.get_open_positions() == []
        row = db.get_history(limit=1)[0]
        assert row['status'] == 'CLOSED'
        assert row['order_id'] == 'EXTERNALLY_CLOSED'
    finally:
        _cleanup(db, db_path)
