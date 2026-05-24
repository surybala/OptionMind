import gc
import os
import sys
import tempfile
from unittest.mock import MagicMock

_alpaca_mock = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.trading',
    'alpaca.trading.client',
    'alpaca.trading.enums',
    'alpaca.trading.requests',
]:
    sys.modules.setdefault(_mod, _alpaca_mock)

from src.database import TradeDatabase
from src.position_lifecycle import PositionLifecycleService


FUTURE_EXPIRY = '2099-12-31'


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix='.db', prefix='test_lifecycle_')
    os.close(fd)
    os.remove(path)
    return TradeDatabase(db_path=path), path


def _cleanup(db, path):
    del db
    gc.collect()
    if os.path.exists(path):
        os.remove(path)


def _pos(trade_id):
    return {
        'id': trade_id,
        'symbol': 'AAPL',
        'expiry': FUTURE_EXPIRY,
        'strike': 150.0,
        'type': 'PCS',
        'premium': 0.50,
        'contracts': 1,
        'status': 'EXECUTED',
        'legs': {'short_strike': 150.0, 'long_strike': 145.0},
    }


def test_live_close_claim_is_atomic_and_submits_once():
    db, path = _tmp_db()
    try:
        tid = db.log_trade(
            'AAPL', FUTURE_EXPIRY, 150.0, 'PCS', 0.50, 0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        executor = MagicMock()
        executor.execute_close_position.return_value = 'CLOSE-1'
        svc = PositionLifecycleService(db, executor)

        first = svc.close_position(
            _pos(tid), limit_price=0.25, pnl=25.0,
            dry_run=False, reason='TEST_CLOSE',
        )
        second = svc.close_position(
            _pos(tid), limit_price=0.20, pnl=30.0,
            dry_run=False, reason='TEST_CLOSE',
        )

        assert first.success is True
        assert first.submitted is True
        assert second.success is True
        assert second.already_pending is True
        executor.execute_close_position.assert_called_once()
        row = db.get_trade(tid)
        assert row['status'] == 'PENDING_CLOSE'
        assert row['order_id'] == 'CLOSE-1'
        assert row['close_order_id'] == 'CLOSE-1'
    finally:
        _cleanup(db, path)


def test_failed_live_close_reopens_for_retry():
    db, path = _tmp_db()
    try:
        tid = db.log_trade(
            'AAPL', FUTURE_EXPIRY, 150.0, 'PCS', 0.50, 0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        executor = MagicMock()
        executor.execute_close_position.return_value = None
        result = PositionLifecycleService(db, executor).close_position(
            _pos(tid), limit_price=0.25, pnl=25.0,
            dry_run=False, reason='TEST_CLOSE',
        )

        assert result.success is False
        assert db.get_trade(tid)['status'] == 'EXECUTED'
    finally:
        _cleanup(db, path)
