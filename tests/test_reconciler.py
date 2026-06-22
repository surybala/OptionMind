"""
Tests for PositionReconciler — Alpaca ↔ Database sync daemon.

Coverage:
  - _pos_to_osi_set         strategy → OSI symbol mapping for all strategies
  - reconcile_pending_opens filled / terminal / still-pending / missing order_id
  - reconcile_pending_closes legs gone (confirm) / legs present (reopen) / no OSI
  - reconcile_executed       expired ghost / pre-expiry ghost / live position ignored
                             grace period skips fresh positions
  - recover_false_ghosts     reopens EXTERNALLY_CLOSED rows whose legs are in Alpaca
  - find_orphans             no orphans / orphan detected
  - run()                    full cycle summary / Alpaca connection failure
  - database helpers         get_false_ghost_positions / reopen_false_ghost
"""
import os
import tempfile
import datetime
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from src.database import TradeDatabase
from src.position_reconciler import PositionReconciler, _pos_to_osi_set, _osi


# ── Helpers ────────────────────────────────────────────────────────────────────

FUTURE_EXPIRY = '2099-12-31'
PAST_EXPIRY   = '2000-01-01'


def _tmp_db() -> TradeDatabase:
    fd, path = tempfile.mkstemp(suffix='.db', prefix='test_recon_')
    os.close(fd)
    return TradeDatabase(db_path=path)


class _EnumLikeStatus:
    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


def _mock_executor(alpaca_symbols: set[str] | None = None) -> MagicMock:
    """Return a mock AlpacaExecutor whose client.get_all_positions() returns
    fake Position objects for each supplied OSI string."""
    executor = MagicMock()
    executor.is_logged_in = True

    positions = []
    for sym in (alpaca_symbols or set()):
        p = MagicMock()
        p.symbol      = sym
        p.asset_class = 'us_option'
        positions.append(p)

    executor.client.get_all_positions.return_value = positions
    return executor


def _make_reconciler(db, alpaca_symbols=None, ghost_grace_minutes=0) -> PositionReconciler:
    # Default grace=0 so existing tests that insert positions right now are not skipped.
    return PositionReconciler(db, _mock_executor(alpaca_symbols),
                              ghost_grace_minutes=ghost_grace_minutes)


# ── _pos_to_osi_set ────────────────────────────────────────────────────────────

class TestPosToOsiSet:
    """Verify OSI symbol derivation for each strategy type."""

    def _pos(self, strat, symbol='AAPL', expiry='2026-04-30', legs=None, strike=None):
        return {'type': strat, 'symbol': symbol, 'expiry': expiry,
                'legs': legs or {}, 'strike': strike}

    def test_csp_uses_short_strike(self):
        pos  = self._pos('CSP', legs={'short_strike': 150.0})
        osis = _pos_to_osi_set(pos)
        assert osis == {_osi('AAPL', '2026-04-30', 150.0, 'PUT')}

    def test_csp_falls_back_to_strike_field(self):
        pos  = self._pos('CSP', strike=200.0)
        osis = _pos_to_osi_set(pos)
        assert osis == {_osi('AAPL', '2026-04-30', 200.0, 'PUT')}

    def test_cc_call(self):
        pos  = self._pos('CC', legs={'short_strike': 160.0})
        osis = _pos_to_osi_set(pos)
        assert osis == {_osi('AAPL', '2026-04-30', 160.0, 'CALL')}

    def test_pcs_two_put_legs(self):
        pos  = self._pos('PCS', legs={'short_strike': 150.0, 'long_strike': 145.0})
        osis = _pos_to_osi_set(pos)
        assert osis == {
            _osi('AAPL', '2026-04-30', 150.0, 'PUT'),
            _osi('AAPL', '2026-04-30', 145.0, 'PUT'),
        }

    def test_ccs_two_call_legs(self):
        pos  = self._pos('CCS', legs={'short_strike': 160.0, 'long_strike': 165.0})
        osis = _pos_to_osi_set(pos)
        assert osis == {
            _osi('AAPL', '2026-04-30', 160.0, 'CALL'),
            _osi('AAPL', '2026-04-30', 165.0, 'CALL'),
        }

    def test_ic_four_legs(self):
        pos  = self._pos('IC', legs={
            'short_put': 140.0, 'long_put': 135.0,
            'short_call': 160.0, 'long_call': 165.0,
        })
        osis = _pos_to_osi_set(pos)
        assert osis == {
            _osi('AAPL', '2026-04-30', 140.0, 'PUT'),
            _osi('AAPL', '2026-04-30', 135.0, 'PUT'),
            _osi('AAPL', '2026-04-30', 160.0, 'CALL'),
            _osi('AAPL', '2026-04-30', 165.0, 'CALL'),
        }

    def test_strangle_two_legs(self):
        pos  = self._pos('STRANGLE', legs={'short_put': 145.0, 'short_call': 155.0})
        osis = _pos_to_osi_set(pos)
        assert osis == {
            _osi('AAPL', '2026-04-30', 145.0, 'PUT'),
            _osi('AAPL', '2026-04-30', 155.0, 'CALL'),
        }

    def test_unknown_strategy_returns_empty(self):
        pos  = self._pos('UNKNOWN')
        osis = _pos_to_osi_set(pos)
        assert osis == set()

    def test_missing_legs_returns_empty(self):
        pos  = {'type': 'PCS', 'symbol': 'AAPL', 'expiry': '2026-04-30', 'legs': {}}
        osis = _pos_to_osi_set(pos)
        assert osis == set()


# ── reconcile_pending_opens ────────────────────────────────────────────────────

class TestReconcilePendingOpens:

    def _insert_pending(self, db, order_id='ORDER-1', legs=None):
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='PENDING', order_id=order_id,
                           legs=legs or {'short_strike': 150.0, 'long_strike': 145.0})
        return tid

    def _order(self, status, fill=None):
        o = MagicMock()
        o.status           = status
        o.filled_avg_price = fill
        return o

    def test_filled_order_confirms_open(self):
        db   = _tmp_db()
        tid  = self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.return_value = self._order('filled', fill=-0.48)

        result = recon.reconcile_pending_opens()

        assert result['confirmed'] == 1
        assert result['voided']    == 0
        rows = db.get_open_positions()
        assert len(rows) == 1
        assert rows[0]['status'] == 'EXECUTED'
        assert abs(rows[0]['premium'] - 0.48) < 1e-6   # abs() normalised

    def test_enum_style_filled_order_confirms_open(self):
        db = _tmp_db()
        self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.return_value = self._order(
            _EnumLikeStatus('OrderStatus.FILLED'),
            fill=-0.48,
        )

        result = recon.reconcile_pending_opens()

        assert result['confirmed'] == 1
        assert db.get_open_positions()[0]['status'] == 'EXECUTED'

    def test_filled_order_updates_premium(self):
        db   = _tmp_db()
        self._insert_pending(db, order_id='ORD-FILL')
        recon = _make_reconciler(db)
        # Fill price differs from scanned 0.50
        recon.executor.client.get_order_by_id.return_value = self._order('filled', fill=0.42)

        recon.reconcile_pending_opens()
        assert abs(db.get_open_positions()[0]['premium'] - 0.42) < 1e-6

    def test_canceled_order_voids_trade(self):
        db  = _tmp_db()
        tid = self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.return_value = self._order('canceled')

        result = recon.reconcile_pending_opens()

        assert result['voided'] == 1
        assert db.get_open_positions() == []   # voided, not in open set

    def test_rejected_order_voids_trade(self):
        db  = _tmp_db()
        self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.return_value = self._order('rejected')

        result = recon.reconcile_pending_opens()
        assert result['voided'] == 1

    def test_still_pending_order_left_alone(self):
        db  = _tmp_db()
        tid = self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.return_value = self._order('new')

        result = recon.reconcile_pending_opens()

        assert result['confirmed']     == 0
        assert result['voided']        == 0
        assert result['still_pending'] == 1
        # Row unchanged
        rows = db.get_pending_open_positions()
        assert len(rows) == 1

    def test_missing_order_id_voids_trade(self):
        db  = _tmp_db()
        # Insert with no order_id
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='PENDING', order_id=None)
        recon = _make_reconciler(db)

        result = recon.reconcile_pending_opens()
        assert result['voided'] == 1

    def test_alpaca_fetch_error_skips_row(self):
        db  = _tmp_db()
        self._insert_pending(db)
        recon = _make_reconciler(db)
        recon.executor.client.get_order_by_id.side_effect = RuntimeError("timeout")

        result = recon.reconcile_pending_opens()
        assert result['still_pending'] == 1
        # Row must still be PENDING
        assert len(db.get_pending_open_positions()) == 1

    def test_no_pending_rows_returns_zeros(self):
        db    = _tmp_db()
        recon = _make_reconciler(db)
        result = recon.reconcile_pending_opens()
        assert result == {'confirmed': 0, 'voided': 0, 'still_pending': 0}


# ── reconcile_pending_closes ───────────────────────────────────────────────────

class TestReconcilePendingCloses:

    def _insert_pending_close(self, db, symbol='AAPL', legs=None) -> int:
        tid = db.log_trade(symbol, FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED',
                           legs=legs or {'short_strike': 150.0, 'long_strike': 145.0})
        db.mark_pending_close(tid)
        return tid

    def _pcs_osis(self, symbol='AAPL') -> set[str]:
        return {
            _osi(symbol, FUTURE_EXPIRY, 150.0, 'PUT'),
            _osi(symbol, FUTURE_EXPIRY, 145.0, 'PUT'),
        }

    def _order(self, status, **attrs):
        o = MagicMock()
        o.status = status
        for key, value in attrs.items():
            setattr(o, key, value)
        return o

    def test_legs_gone_confirms_close(self):
        db   = _tmp_db()
        tid  = self._insert_pending_close(db)
        # Alpaca holds no positions for this symbol
        recon = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_pending_closes(alpaca_symbols=set())

        assert result['confirmed'] == 1
        assert result['reopened']  == 0
        # Row must be CLOSED now
        history = db.get_history(limit=1)
        assert history[0]['status'] == 'CLOSED'
        assert history[0]['order_id'] == 'RECONCILED'

    def test_legs_present_reopens_position(self):
        db   = _tmp_db()
        tid  = self._insert_pending_close(db)
        osis = self._pcs_osis()
        recon = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.reconcile_pending_closes(alpaca_symbols=osis)

        assert result['reopened']  == 1
        assert result['confirmed'] == 0
        # Row back to EXECUTED
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]['status'] == 'EXECUTED'

    def test_partial_legs_in_alpaca_reopens(self):
        """If only one leg is present, position is not fully closed — reopen."""
        db   = _tmp_db()
        self._insert_pending_close(db)
        # Only the short leg still in Alpaca
        partial = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT')}
        recon   = _make_reconciler(db, alpaca_symbols=partial)

        result = recon.reconcile_pending_closes(alpaca_symbols=partial)
        assert result['reopened'] == 1

    def test_legs_present_with_open_close_order_stays_pending(self):
        db   = _tmp_db()
        tid  = self._insert_pending_close(db)
        db.mark_pending_close(tid, pnl=12.34, close_order_id='CLOSE-OPEN-1')
        osis = self._pcs_osis()
        recon = _make_reconciler(db, alpaca_symbols=osis)
        recon.executor.client.get_order_by_id.return_value = self._order(
            'new',
            order_class='mleg',
            legs=[MagicMock(), MagicMock()],
        )

        result = recon.reconcile_pending_closes(alpaca_symbols=osis)

        assert result['confirmed'] == 0
        assert result['reopened'] == 0
        rows = db.get_pending_close_positions()
        assert len(rows) == 1
        assert rows[0]['order_id'] == 'CLOSE-OPEN-1'

    def test_legs_present_with_open_partial_close_order_reopens(self):
        db   = _tmp_db()
        tid  = self._insert_pending_close(db)
        db.mark_pending_close(tid, pnl=12.34, close_order_id='CLOSE-PARTIAL-1')
        osis = self._pcs_osis()
        recon = _make_reconciler(db, alpaca_symbols=osis)
        recon.executor.client.get_order_by_id.return_value = self._order(
            'new',
            symbol=_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
        )

        result = recon.reconcile_pending_closes(alpaca_symbols=osis)

        assert result['confirmed'] == 0
        assert result['reopened'] == 1
        recon.executor.client.cancel_order_by_id.assert_called_once_with('CLOSE-PARTIAL-1')
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]['status'] == 'EXECUTED'

    def test_legs_present_with_terminal_close_order_reopens(self):
        db   = _tmp_db()
        tid  = self._insert_pending_close(db)
        db.mark_pending_close(tid, pnl=12.34, close_order_id='CLOSE-CANCEL-1')
        osis = self._pcs_osis()
        recon = _make_reconciler(db, alpaca_symbols=osis)
        recon.executor.client.get_order_by_id.return_value = self._order('canceled')

        result = recon.reconcile_pending_closes(alpaca_symbols=osis)

        assert result['reopened'] == 1
        assert len(db.get_open_positions()) == 1

    def test_no_legs_data_skips_row(self):
        """Position with unrecognised strategy cannot build OSI symbols — skipped with warning."""
        db  = _tmp_db()
        # 'UNKNOWN' is not handled by _pos_to_osi_set → returns empty set → skip
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'UNKNOWN', 0.50, 0.80,
                           status='EXECUTED', legs=None)
        db.mark_pending_close(tid)
        recon  = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_pending_closes(alpaca_symbols=set())
        # Skipped — no change to DB row
        assert result['confirmed'] == 0
        assert result['reopened']  == 0
        assert len(db.get_pending_close_positions()) == 1

    def test_no_pending_close_rows_returns_zeros(self):
        db    = _tmp_db()
        recon = _make_reconciler(db)
        result = recon.reconcile_pending_closes(alpaca_symbols=set())
        assert result == {'confirmed': 0, 'reopened': 0}

    def test_stored_pnl_and_order_id_used_on_confirm(self):
        """
        When the monitor stored estimated P&L and close order ID on the
        PENDING_CLOSE row, the reconciler must use those values — not a $0
        placeholder — when it confirms the fill.
        """
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED',
                           legs={'short_strike': 150.0, 'long_strike': 145.0})
        # Phase 1: mark intent before broker call
        db.mark_pending_close(tid)
        # Phase 1b: store estimated P&L + close order ID after broker accepts
        db.mark_pending_close(tid, pnl=38.50, close_order_id='CLOSE_ORD_42')

        recon  = _make_reconciler(db, alpaca_symbols=set())
        recon.executor.client.get_order_by_id.return_value = self._order('filled')
        result = recon.reconcile_pending_closes(alpaca_symbols=set())

        assert result['confirmed'] == 1
        history = db.get_history(limit=1)
        assert history[0]['status']   == 'CLOSED'
        assert history[0]['pnl']      == pytest.approx(38.50)
        assert history[0]['order_id'] == 'CLOSE_ORD_42'

    def test_enum_style_filled_close_order_confirms_close(self):
        db = _tmp_db()
        tid = self._insert_pending_close(db)
        db.mark_pending_close(tid, pnl=38.50, close_order_id='CLOSE_ENUM_1')
        recon = _make_reconciler(db, alpaca_symbols=set())
        recon.executor.client.get_order_by_id.return_value = self._order(
            _EnumLikeStatus('OrderStatus.FILLED')
        )

        result = recon.reconcile_pending_closes(alpaca_symbols=set())

        assert result['confirmed'] == 1
        history = db.get_history(limit=1)
        assert history[0]['status'] == 'CLOSED'
        assert history[0]['order_id'] == 'CLOSE_ENUM_1'

    def test_filled_close_order_price_overrides_estimated_pnl(self):
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED',
                           legs={'short_strike': 150.0, 'long_strike': 145.0},
                           contracts=2)
        db.mark_pending_close(tid, pnl=38.50, close_order_id='CLOSE_FILL_1')

        recon = _make_reconciler(db, alpaca_symbols=set())
        order = self._order('filled')
        order.filled_avg_price = 0.20
        recon.executor.client.get_order_by_id.return_value = order
        result = recon.reconcile_pending_closes(alpaca_symbols=set())

        assert result['confirmed'] == 1
        row = db.get_history(limit=1)[0]
        assert row['pnl'] == pytest.approx(60.0)  # (0.50 - 0.20) * 100 * 2
        assert row['pnl_source'] == 'ALPACA_FILLS'
        assert row['pnl_verified'] == 1
        orders = db.get_trade_orders(tid)
        assert orders[0]['order_id'] == 'CLOSE_FILL_1'
        assert orders[0]['filled_avg_price'] == pytest.approx(0.20)

    def test_crash_recovery_uses_reconciled_fallback(self):
        """
        When the process crashed before storing close order info, the reconciler
        falls back to order_id='RECONCILED' and pnl=0 (safe defaults).
        """
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'CSP', 1.00, 0.80,
                           status='EXECUTED',
                           legs={'short_strike': 150.0})
        # Only Phase 1 completed before crash — no pnl/order stored
        db.mark_pending_close(tid)

        recon  = _make_reconciler(db, alpaca_symbols=set())
        result = recon.reconcile_pending_closes(alpaca_symbols=set())

        assert result['confirmed'] == 1
        history = db.get_history(limit=1)
        assert history[0]['status']   == 'CLOSED'
        assert history[0]['order_id'] == 'RECONCILED'
        assert history[0]['pnl_source'] == 'RECONCILED_PLACEHOLDER'
        assert history[0]['pnl_verified'] == 0


# ── reconcile_executed ────────────────────────────────────────────────────────

class TestReconcileExecuted:

    def test_expired_ghost_closed_at_full_premium(self):
        """EXECUTED position with past expiry and no Alpaca legs → CLOSED, pnl = premium×100."""
        db  = _tmp_db()
        db.log_trade('AAPL', PAST_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        recon = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed'] == 1

        history = db.get_history(limit=1)
        assert history[0]['status']   == 'CLOSED'
        assert history[0]['pnl']      == pytest.approx(50.0)   # 0.50 × 100
        assert history[0]['order_id'] == 'EXPIRED_RECONCILED'
        assert history[0]['pnl_source'] == 'EXPIRED'
        assert history[0]['pnl_verified'] == 1

    def test_expired_ghost_pnl_respects_contract_quantity(self):
        db = _tmp_db()
        db.log_trade('AAPL', PAST_EXPIRY, 150, 'CSP', 0.50, 0.80,
                     status='EXECUTED', legs={'short_strike': 150.0},
                     contracts=3)
        recon = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())

        assert result['ghost_closed'] == 1
        assert db.get_history(limit=1)[0]['pnl'] == pytest.approx(150.0)

    def test_pre_expiry_ghost_closed_at_zero_pnl(self):
        """EXECUTED with future expiry but legs missing → externally closed."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        recon = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed'] == 1

        history = db.get_history(limit=1)
        assert history[0]['status']   == 'CLOSED'
        assert history[0]['pnl']      == 0.0
        assert history[0]['order_id'] == 'EXTERNALLY_CLOSED'
        assert history[0]['pnl_source'] == 'EXTERNAL_PLACEHOLDER'
        assert history[0]['pnl_verified'] == 0

    def test_live_position_not_touched(self):
        """Position whose legs are in Alpaca must not be closed."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        live_osis = {
            _osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
            _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT'),
        }
        recon  = _make_reconciler(db, alpaca_symbols=live_osis)

        result = recon.reconcile_executed(alpaca_symbols=live_osis)
        assert result['ghost_closed'] == 0
        assert len(db.get_open_positions()) == 1

    def test_dry_run_positions_ignored(self):
        """DRY_RUN positions never had real Alpaca legs — skip ghost detection."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='DRY_RUN',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        recon  = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed'] == 0

    def test_multiple_ghosts_all_closed(self):
        db = _tmp_db()
        for _ in range(3):
            db.log_trade('AAPL', PAST_EXPIRY, 150, 'PCS', 0.50, 0.80,
                         status='EXECUTED',
                         legs={'short_strike': 150.0, 'long_strike': 145.0})
        recon  = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed'] == 3
        assert db.get_open_positions() == []

    def test_position_with_no_osi_symbols_skipped(self):
        """Position with unknown strategy and no leg data cannot be verified — skip."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'UNKNOWN', 0.50, 0.80,
                     status='EXECUTED', legs={})
        recon  = _make_reconciler(db, alpaca_symbols=set())

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed'] == 0


# ── find_orphans ──────────────────────────────────────────────────────────────

class TestFindOrphans:

    def test_no_orphans_when_all_matched(self):
        db   = _tmp_db()
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        recon   = _make_reconciler(db, alpaca_symbols=osis)
        orphans = recon.find_orphans(osis)
        assert orphans == []

    def test_orphan_detected(self):
        db  = _tmp_db()
        # Alpaca has a position we don't know about
        mystery_osi = _osi('MSFT', FUTURE_EXPIRY, 300.0, 'CALL')
        recon   = _make_reconciler(db, alpaca_symbols={mystery_osi})
        orphans = recon.find_orphans({mystery_osi})
        assert mystery_osi in orphans

    def test_pending_close_position_legs_are_not_orphans(self):
        db = _tmp_db()
        trade_id = db.log_trade(
            'MRVL', FUTURE_EXPIRY, 125, 'IC', 1.20, 0.80,
            status='EXECUTED',
            legs={
                'short_put': 125.0,
                'long_put': 105.0,
                'short_call': 175.0,
                'long_call': 195.0,
            },
        )
        db.mark_pending_close(trade_id, close_order_id='CLOSE-MRVL')
        osis = {
            _osi('MRVL', FUTURE_EXPIRY, 125.0, 'PUT'),
            _osi('MRVL', FUTURE_EXPIRY, 105.0, 'PUT'),
            _osi('MRVL', FUTURE_EXPIRY, 175.0, 'CALL'),
            _osi('MRVL', FUTURE_EXPIRY, 195.0, 'CALL'),
        }
        recon = _make_reconciler(db, alpaca_symbols=osis)

        assert recon.find_orphans(osis) == []

    def test_empty_alpaca_no_orphans(self):
        db      = _tmp_db()
        recon   = _make_reconciler(db, alpaca_symbols=set())
        orphans = recon.find_orphans(set())
        assert orphans == []

    def test_known_and_unknown_positions(self):
        db  = _tmp_db()
        known_osi   = _osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT')
        unknown_osi = _osi('TSLA', FUTURE_EXPIRY, 200.0, 'CALL')
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'CSP', 0.50, 0.80,
                     status='EXECUTED', legs={'short_strike': 150.0})
        recon   = _make_reconciler(db, alpaca_symbols={known_osi, unknown_osi})
        orphans = recon.find_orphans({known_osi, unknown_osi})
        assert orphans == [unknown_osi]
        assert known_osi not in orphans


# ── run() integration ─────────────────────────────────────────────────────────

class TestReconcilerRun:

    def test_run_returns_summary_dict(self):
        db    = _tmp_db()
        recon = _make_reconciler(db, alpaca_symbols=set())
        summary = recon.run()
        assert 'alpaca_positions' in summary
        assert 'recovered'        in summary
        assert 'pending_opens'    in summary
        assert 'pending_closes'   in summary
        assert 'executed'         in summary
        assert 'orphans'          in summary

    def test_run_alpaca_failure_returns_error(self):
        db       = _tmp_db()
        executor = _mock_executor()
        executor.is_logged_in = False
        executor.login.return_value = False
        recon    = PositionReconciler(db, executor)

        summary = recon.run()
        assert 'error' in summary

    def test_run_processes_all_checks(self):
        """End-to-end: one pending open (filled), one expired ghost, one orphan."""
        db   = _tmp_db()

        # Pending open — will be confirmed; its leg must be in Alpaca to avoid
        # being treated as a ghost in the reconcile_executed pass.
        aapl_osi = _osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT')
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'CSP', 0.50, 0.80,
                     status='PENDING', order_id='ORD-123',
                     legs={'short_strike': 150.0})

        # Expired ghost — legs absent from Alpaca, expiry in the past
        db.log_trade('MSFT', PAST_EXPIRY, 300, 'CSP', 1.00, 0.85,
                     status='EXECUTED', legs={'short_strike': 300.0})

        # Orphan in Alpaca (no DB record)
        orphan_osi = _osi('TSLA', FUTURE_EXPIRY, 200.0, 'CALL')

        order = MagicMock()
        order.status           = 'filled'
        order.filled_avg_price = 0.48

        # Alpaca holds the AAPL leg (just opened) + the TSLA orphan
        executor = _mock_executor(alpaca_symbols={aapl_osi, orphan_osi})
        executor.client.get_order_by_id.return_value = order

        recon   = PositionReconciler(db, executor, ghost_grace_minutes=0)
        summary = recon.run()

        assert summary['pending_opens']['confirmed']  == 1
        assert summary['executed']['ghost_closed']    == 1   # MSFT expired ghost
        assert orphan_osi in summary['orphans']
        # AAPL position should now be EXECUTED (confirmed) and still open
        open_pos = db.get_open_positions()
        assert any(p['symbol'] == 'AAPL' for p in open_pos)


# ── Ghost detection grace period ──────────────────────────────────────────────

class TestGhostGracePeriod:
    """reconcile_executed skips positions created within ghost_grace_minutes."""

    _LEGS = {'short_strike': 150.0, 'long_strike': 145.0}

    def test_fresh_position_skipped_with_grace(self):
        """Position inserted just now is not ghost-closed when grace_minutes > 0."""
        db = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=self._LEGS)
        recon  = _make_reconciler(db, alpaca_symbols=set(), ghost_grace_minutes=10)

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed']   == 0
        assert result['grace_skipped']  == 1
        assert len(db.get_open_positions()) == 1   # still open

    def test_fresh_position_closed_with_zero_grace(self):
        """grace_minutes=0 disables the grace window — fresh position is still ghost-checked."""
        db = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=self._LEGS)
        recon  = _make_reconciler(db, alpaca_symbols=set(), ghost_grace_minutes=0)

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed']  == 1
        assert result['grace_skipped'] == 0

    def test_old_position_not_skipped(self):
        """Position older than grace window must still be ghost-checked."""
        db = _tmp_db()
        db.log_trade('AAPL', PAST_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=self._LEGS)
        # Backdate the timestamp beyond the grace window
        import sqlite3
        old_ts = (datetime.datetime.now()
                  - datetime.timedelta(minutes=20)).isoformat()
        with sqlite3.connect(db.db_path) as conn:
            conn.execute("UPDATE trades SET timestamp = ?", (old_ts,))
        recon  = _make_reconciler(db, alpaca_symbols=set(), ghost_grace_minutes=10)

        result = recon.reconcile_executed(alpaca_symbols=set())
        assert result['ghost_closed']  == 1
        assert result['grace_skipped'] == 0

    def test_result_has_grace_skipped_key(self):
        """reconcile_executed result always contains grace_skipped key."""
        db     = _tmp_db()
        recon  = _make_reconciler(db, alpaca_symbols=set())
        result = recon.reconcile_executed(alpaca_symbols=set())
        assert 'grace_skipped' in result

    def test_live_legs_not_ghosted(self):
        """A fresh position whose legs ARE in Alpaca is never ghost-closed.
        The grace check fires first, so it counts as grace_skipped — that is
        correct and harmless (it was never going to be ghosted either way)."""
        db   = _tmp_db()
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=self._LEGS)
        recon  = _make_reconciler(db, alpaca_symbols=osis, ghost_grace_minutes=10)

        result = recon.reconcile_executed(alpaca_symbols=osis)
        assert result['ghost_closed'] == 0
        assert len(db.get_open_positions()) == 1   # position untouched


# ── recover_false_ghosts ──────────────────────────────────────────────────────

class TestRecoverFalseGhosts:
    """recover_false_ghosts reopens EXTERNALLY_CLOSED rows whose legs are in Alpaca."""

    _LEGS = {'short_strike': 150.0, 'long_strike': 145.0}

    def _insert_false_ghost(self, db, symbol='AAPL', expiry=FUTURE_EXPIRY):
        """Insert an EXECUTED row then close it as EXTERNALLY_CLOSED (simulating
        what the ghost-detection race condition would have done)."""
        tid = db.log_trade(symbol, expiry, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')
        return tid

    def test_recovers_when_legs_in_alpaca(self):
        """CLOSED/EXTERNALLY_CLOSED + legs present in Alpaca → restored to EXECUTED."""
        db   = _tmp_db()
        tid  = self._insert_false_ghost(db)
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}
        recon  = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.recover_false_ghosts(alpaca_symbols=osis)
        assert result['recovered'] == 1
        assert len(db.get_open_positions()) == 1
        pos = db.get_open_positions()[0]
        assert pos['status']   == 'EXECUTED'
        assert pos['pnl']      is None
        assert pos['order_id'] is None

    def test_no_recovery_when_legs_absent(self):
        """CLOSED/EXTERNALLY_CLOSED but legs still absent from Alpaca → stays CLOSED."""
        db  = _tmp_db()
        self._insert_false_ghost(db)
        recon  = _make_reconciler(db, alpaca_symbols=set())

        result = recon.recover_false_ghosts(alpaca_symbols=set())
        assert result['recovered'] == 0
        assert db.get_open_positions() == []

    def test_does_not_reopen_legitimate_external_close(self):
        """CLOSED with a real close order_id (not EXTERNALLY_CLOSED) is never reopened."""
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 42.0, 'REAL_CLOSE_ORDER_123')
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}
        recon  = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.recover_false_ghosts(alpaca_symbols=osis)
        assert result['recovered'] == 0
        assert db.get_open_positions() == []

    def test_does_not_reopen_past_expiry(self):
        """EXTERNALLY_CLOSED position with past expiry is excluded — already expired."""
        db  = _tmp_db()
        self._insert_false_ghost(db, expiry=PAST_EXPIRY)
        osis = {_osi('AAPL', PAST_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', PAST_EXPIRY, 145.0, 'PUT')}
        recon  = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.recover_false_ghosts(alpaca_symbols=osis)
        assert result['recovered'] == 0

    def test_recovers_multiple_false_ghosts(self):
        db   = _tmp_db()
        osis = set()
        for sym, sp, lp in [('AAPL', 150.0, 145.0), ('TSLA', 200.0, 195.0)]:
            tid = db.log_trade(sym, FUTURE_EXPIRY, sp, 'PCS', 0.50, 0.80,
                               status='EXECUTED',
                               legs={'short_strike': sp, 'long_strike': lp})
            db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')
            osis.add(_osi(sym, FUTURE_EXPIRY, sp, 'PUT'))
            osis.add(_osi(sym, FUTURE_EXPIRY, lp, 'PUT'))
        recon  = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.recover_false_ghosts(alpaca_symbols=osis)
        assert result['recovered'] == 2
        assert len(db.get_open_positions()) == 2

    def test_partial_recovery_when_only_some_in_alpaca(self):
        """Only the position whose legs are in Alpaca is recovered."""
        db   = _tmp_db()
        tid1 = self._insert_false_ghost(db, symbol='AAPL')
        tid2 = self._insert_false_ghost(db, symbol='TSLA')
        # Only AAPL legs are in Alpaca
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}
        recon  = _make_reconciler(db, alpaca_symbols=osis)

        result = recon.recover_false_ghosts(alpaca_symbols=osis)
        assert result['recovered'] == 1
        open_syms = {p['symbol'] for p in db.get_open_positions()}
        assert open_syms == {'AAPL'}

    def test_run_recovery_fires_before_ghost_detection(self):
        """In run(), recovered positions are not immediately re-ghosted."""
        db  = _tmp_db()
        tid = self._insert_false_ghost(db)
        osis = {_osi('AAPL', FUTURE_EXPIRY, 150.0, 'PUT'),
                _osi('AAPL', FUTURE_EXPIRY, 145.0, 'PUT')}

        executor = _mock_executor(alpaca_symbols=osis)
        recon    = PositionReconciler(db, executor, ghost_grace_minutes=0)
        summary  = recon.run()

        assert summary['recovered']['recovered']     == 1
        assert summary['executed']['ghost_closed']   == 0   # not re-ghosted
        assert len(db.get_open_positions())          == 1


# ── database: get_false_ghost_positions / reopen_false_ghost ─────────────────

class TestFalseGhostDbHelpers:

    _LEGS = {'short_strike': 150.0, 'long_strike': 145.0}

    def test_get_false_ghost_positions_returns_candidates(self):
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')

        candidates = db.get_false_ghost_positions()
        assert len(candidates) == 1
        assert candidates[0]['symbol'] == 'AAPL'

    def test_get_false_ghost_excludes_past_expiry(self):
        db  = _tmp_db()
        tid = db.log_trade('AAPL', PAST_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')

        assert db.get_false_ghost_positions() == []

    def test_get_false_ghost_excludes_other_close_reasons(self):
        """Rows closed by stop-loss, profit-take, or expiry are not candidates."""
        db = _tmp_db()
        for reason in ('STOP_LOSS', 'PROFIT_TAKE', 'EXPIRED_RECONCILED', 'REAL_ORDER_ID'):
            tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                               status='EXECUTED', legs=self._LEGS)
            db.close_position(tid, 0.0, reason)

        assert db.get_false_ghost_positions() == []

    def test_get_false_ghost_excludes_non_closed_statuses(self):
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=self._LEGS)
        assert db.get_false_ghost_positions() == []

    def test_reopen_false_ghost_restores_executed(self):
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')

        db.reopen_false_ghost(tid)

        pos = db.get_open_positions()
        assert len(pos) == 1
        assert pos[0]['status']   == 'EXECUTED'
        assert pos[0]['pnl']      is None
        assert pos[0]['order_id'] is None

    def test_reopen_false_ghost_only_acts_on_externally_closed(self):
        """reopen_false_ghost must not touch rows with a different close reason."""
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 100.0, 'STOP_LOSS')

        db.reopen_false_ghost(tid)   # should be a no-op

        assert db.get_open_positions() == []

    def test_reopen_false_ghost_preserves_premium(self):
        """Entry premium must survive the reopen so P&L at expiry is correct."""
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 1.25, 0.80,
                           status='EXECUTED', legs=self._LEGS)
        db.close_position(tid, 0.0, 'EXTERNALLY_CLOSED')
        db.reopen_false_ghost(tid)

        pos = db.get_open_positions()
        assert pos[0]['premium'] == pytest.approx(1.25)
