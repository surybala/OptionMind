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
        executor.config = {'risk_parameters': {'close_execution': {'market_after_same_day_attempts': 3}}}
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
        executor.execute_close_position.assert_called_once()
        submitted_limit = executor.execute_close_position.call_args.kwargs['limit_price']
        assert submitted_limit > 0.25
        assert executor.execute_close_position.call_args.kwargs['order_type'] == 'limit'
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
        executor.config = {'risk_parameters': {'close_execution': {'market_after_same_day_attempts': 3}}}
        executor.execute_close_position.return_value = None
        result = PositionLifecycleService(db, executor).close_position(
            _pos(tid), limit_price=0.25, pnl=25.0,
            dry_run=False, reason='TEST_CLOSE',
        )

        assert result.success is False
        assert db.get_trade(tid)['status'] == 'EXECUTED'
    finally:
        _cleanup(db, path)


def test_close_position_escalates_to_market_after_same_day_attempts():
    db, path = _tmp_db()
    try:
        tid = db.log_trade(
            'AAPL', FUTURE_EXPIRY, 150.0, 'PCS', 0.50, 0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        db.upsert_trade_order(tid, 'CLOSE-ATTEMPT-1', role='CLOSE', status='canceled')
        db.upsert_trade_order(tid, 'CLOSE-ATTEMPT-2', role='CLOSE', status='canceled')
        executor = MagicMock()
        executor.config = {'risk_parameters': {'close_execution': {'market_after_same_day_attempts': 3}}}
        executor.execute_close_position.return_value = 'CLOSE-3'

        result = PositionLifecycleService(db, executor).close_position(
            _pos(tid), limit_price=0.25, pnl=25.0,
            dry_run=False, reason='TEST_CLOSE',
        )

        assert result.success is True
        assert executor.execute_close_position.call_args.kwargs['order_type'] == 'market'
        assert executor.execute_close_position.call_args.kwargs['limit_price'] is None
    finally:
        _cleanup(db, path)
