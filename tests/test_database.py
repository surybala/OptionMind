import gc
import os
import sqlite3
import sys
import tempfile
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database import TradeDatabase


def _make_db():
    """Return (TradeDatabase, db_path) using a temp file so tests don't collide."""
    fd, path = tempfile.mkstemp(suffix='.db', prefix='test_trades_', dir=tempfile.gettempdir())
    os.close(fd)
    os.remove(path)  # SQLite will recreate it; dirname is already the tmp dir
    return TradeDatabase(db_path=path), path


def _cleanup(db, db_path):
    """Release the DB object, force CPython GC to close sqlite handles, then delete."""
    del db
    gc.collect()
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
    except PermissionError:
        pass  # Windows: WAL lock still held; OS will clean up on next run


class TestTradeDatabase(unittest.TestCase):

    def setUp(self):
        self.db, self.db_path = _make_db()

    def tearDown(self):
        _cleanup(self.db, self.db_path)

    def test_log_trade_and_fetch(self):
        # Log a trade
        trade_id = self.db.log_trade(
            symbol="AAPL", 
            expiry="2024-01-01", 
            strike=150.0, 
            type="PUT", 
            premium=2.5, 
            prob_expiry=0.85
        )
        self.assertIsNotNone(trade_id)

        # Fetch history
        history = self.db.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['symbol'], "AAPL")
        self.assertEqual(history[0]['premium'], 2.5)

    def test_update_status(self):
        trade_id = self.db.log_trade(
            symbol="MSFT", 
            expiry="2024-01-01", 
            strike=300.0, 
            type="PUT", 
            premium=3.0, 
            prob_expiry=0.9
        )
        
        self.db.update_status(trade_id, "CLOSED", pnl=100.0)
        
        history = self.db.get_history()
        trade = history[0]
        self.assertEqual(trade['status'], "CLOSED")
        self.assertEqual(trade['pnl'], 100.0)

class TestTradeDatabaseEdgeCases(unittest.TestCase):

    def setUp(self):
        self.db, self.db_path = _make_db()

    def tearDown(self):
        _cleanup(self.db, self.db_path)

    # ── log_trade defaults ────────────────────────────────────────────────────

    def test_default_status_is_pending(self):
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        history = self.db.get_history()
        self.assertEqual(history[0]['status'], 'PENDING')

    def test_default_order_id_is_null(self):
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        history = self.db.get_history()
        self.assertIsNone(history[0]['order_id'])

    def test_log_trade_returns_incrementing_ids(self):
        id1 = self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.0, 0.85)
        id2 = self.db.log_trade('MSFT', '2026-04-30', 300.0, 'PUT', 2.0, 0.90)
        self.assertGreater(id2, id1)

    def test_log_trade_stores_order_id(self):
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85, order_id='ORD123')
        history = self.db.get_history()
        self.assertEqual(history[0]['order_id'], 'ORD123')

    def test_log_spread_trade_type(self):
        self.db.log_trade('TSLA', '2026-04-30', 200.0, 'PCS', 0.50, 0.80)
        history = self.db.get_history()
        self.assertEqual(history[0]['type'], 'PCS')

    # ── get_history ───────────────────────────────────────────────────────────

    def test_get_history_empty_database_returns_empty_list(self):
        self.assertEqual(self.db.get_history(), [])

    def test_get_history_respects_limit(self):
        for i in range(5):
            self.db.log_trade(f'SYM{i}', '2026-04-30', 100.0, 'PUT', 1.0, 0.85)
        history = self.db.get_history(limit=3)
        self.assertEqual(len(history), 3)

    def test_get_history_ordered_most_recent_first(self):
        self.db.log_trade('FIRST', '2026-04-30', 100.0, 'PUT', 1.0, 0.85)
        self.db.log_trade('SECOND', '2026-04-30', 100.0, 'PUT', 1.0, 0.85)
        history = self.db.get_history()
        self.assertEqual(history[0]['symbol'], 'SECOND')
        self.assertEqual(history[1]['symbol'], 'FIRST')

    # ── update_status ─────────────────────────────────────────────────────────

    def test_update_status_to_cancelled(self):
        trade_id = self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        self.db.update_status(trade_id, 'CANCELLED')
        history = self.db.get_history()
        self.assertEqual(history[0]['status'], 'CANCELLED')

    def test_update_status_with_negative_pnl(self):
        trade_id = self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        self.db.update_status(trade_id, 'CLOSED', pnl=-250.0)
        history = self.db.get_history()
        self.assertEqual(history[0]['pnl'], -250.0)

    def test_update_status_nonexistent_id_does_not_raise(self):
        # Should silently succeed (0 rows updated)
        try:
            self.db.update_status(99999, 'CLOSED', pnl=0)
        except Exception as e:
            self.fail(f"update_status raised unexpectedly: {e}")

    def test_update_status_multiple_times(self):
        trade_id = self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        self.db.update_status(trade_id, 'EXECUTED')
        self.db.update_status(trade_id, 'CLOSED', pnl=150.0)
        history = self.db.get_history()
        self.assertEqual(history[0]['status'], 'CLOSED')
        self.assertEqual(history[0]['pnl'], 150.0)

    # ── persistence ───────────────────────────────────────────────────────────

    def test_data_persists_across_connections(self):
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        # Open a fresh connection to the same file
        db2 = TradeDatabase(db_path=self.db_path)
        history = db2.get_history()
        # Explicitly release db2 before tearDown tries to delete the file
        del db2
        gc.collect()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['symbol'], 'AAPL')

    def test_init_db_is_idempotent(self):
        # Calling _init_db twice should not raise or corrupt data
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PUT', 1.5, 0.85)
        self.db._init_db()
        history = self.db.get_history()
        self.assertEqual(len(history), 1)


class TestContractsColumn(unittest.TestCase):

    def setUp(self):
        self.db, self.db_path = _make_db()

    def tearDown(self):
        _cleanup(self.db, self.db_path)

    def test_contracts_defaults_to_one(self):
        self.db.log_trade('AAPL', '2026-04-30', 150.0, 'PCS', 0.50, 0.80)
        row = self.db.get_history()[0]
        self.assertEqual(row['contracts'], 1)

    def test_contracts_stored_and_retrieved(self):
        self.db.log_trade('SPY', '2026-04-30', 450.0, 'IC', 0.80, 0.82, contracts=15)
        row = self.db.get_history()[0]
        self.assertEqual(row['contracts'], 15)

    def test_contracts_integer_coercion(self):
        self.db.log_trade('QQQ', '2026-04-30', 400.0, 'PCS', 0.60, 0.78, contracts=7)
        row = self.db.get_history()[0]
        self.assertIsInstance(row['contracts'], int)

    def test_contracts_in_open_positions(self):
        trade_id = self.db.log_trade('SPY', '2099-12-31', 500.0, 'PCS', 0.50, 0.80,
                                     status='EXECUTED', contracts=20)
        open_pos = self.db.get_open_positions()
        self.assertEqual(len(open_pos), 1)
        self.assertEqual(open_pos[0]['contracts'], 20)

    def test_migration_adds_contracts_to_existing_db(self):
        """Simulate a DB created before the contracts column existed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, expiry TEXT, strike REAL,
                    type TEXT, premium REAL, prob_expiry REAL,
                    status TEXT, order_id TEXT, pnl REAL DEFAULT 0, legs TEXT
                )
            """)
            conn.execute(
                "INSERT INTO trades (timestamp,symbol,expiry,strike,type,premium,"
                "prob_expiry,status,order_id,pnl,legs) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ('2026-01-01', 'AAPL', '2026-04-30', 150.0, 'PCS', 0.50, 0.80,
                 'EXECUTED', None, 0, None)
            )
            conn.commit()
        # Re-open — migration should add contracts column without error
        db2 = TradeDatabase(db_path=self.db_path)
        row = db2.get_history()[0]
        self.assertIn('contracts', row)
        self.assertEqual(row['contracts'], 1)


class TestOrderLedgerAndPnlSource(unittest.TestCase):

    def setUp(self):
        self.db, self.db_path = _make_db()

    def tearDown(self):
        _cleanup(self.db, self.db_path)

    def test_migration_adds_order_ledger_and_pnl_columns(self):
        with sqlite3.connect(self.db_path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(trades)")}
            order_cols = {row[1] for row in conn.execute("PRAGMA table_info(trade_orders)")}
            pred_cols = {row[1] for row in conn.execute("PRAGMA table_info(model_predictions)")}
            decision_cols = {row[1] for row in conn.execute("PRAGMA table_info(model_decisions)")}

        self.assertIn('pnl_source', cols)
        self.assertIn('pnl_verified', cols)
        self.assertIn('model_prediction_id', cols)
        self.assertIn('model_decision_id', cols)
        self.assertIn('order_id', order_cols)
        self.assertIn('filled_avg_price', order_cols)
        self.assertIn('features_hash', pred_cols)
        self.assertIn('final_outcome', decision_cols)

    def test_record_open_and_close_orders_in_ledger(self):
        tid = self.db.log_trade(
            'AAPL', '2026-04-30', 150.0, 'PCS', 0.50, 0.80,
            status='PENDING',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )

        self.db.record_open_order(tid, 'OPEN-1')
        self.db.confirm_open(tid, 'OPEN-1')
        self.db.mark_pending_close(tid, pnl=20.0, close_order_id='CLOSE-1')

        orders = self.db.get_trade_orders(tid)

        self.assertEqual([o['order_id'] for o in orders], ['OPEN-1', 'CLOSE-1'])
        self.assertEqual([o['role'] for o in orders], ['OPEN', 'CLOSE'])

    def test_get_open_positions_includes_open_order_filled_at(self):
        tid = self.db.log_trade(
            'AAPL', '2099-12-31', 150.0, 'PCS', 0.50, 0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        self.db.upsert_trade_order(
            tid,
            'OPEN-1',
            role='OPEN',
            status='filled',
            filled_at='2099-01-01T09:35:00',
        )

        positions = self.db.get_open_positions()

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['filled_at'], '2099-01-01T09:35:00')

    def test_confirm_close_stores_verified_pnl_source(self):
        tid = self.db.log_trade(
            'AAPL', '2026-04-30', 150.0, 'PCS', 0.50, 0.80,
            status='EXECUTED',
            legs={'short_strike': 150.0, 'long_strike': 145.0},
        )
        self.db.mark_pending_close(tid, pnl=20.0, close_order_id='CLOSE-1')

        self.db.confirm_close(
            tid,
            pnl=30.0,
            close_order_id='CLOSE-1',
            pnl_source='ALPACA_FILLS',
            pnl_verified=True,
        )

        row = self.db.get_trade(tid)
        self.assertEqual(row['status'], 'CLOSED')
        self.assertEqual(row['pnl_source'], 'ALPACA_FILLS')
        self.assertEqual(row['pnl_verified'], 1)

    def test_model_prediction_decision_links_to_trade_and_outcome(self):
        pick = {
            'symbol': 'SPY',
            'strategy': 'PCS',
            'expiry': '2026-06-19',
            'short_strike': 500.0,
            'long_strike': 495.0,
            'premium': 0.80,
            'prob_win': 0.72,
            'model_id': 'champion_v1',
            'model_type': 'linear_least_squares_v001',
            'model_score': 42.5,
            'features': {'dte': 26, 'option_entry_price': 1.25},
            'score_components': {'expected_pnl': 42.5, 'prob_win': 0.72},
        }

        prediction_id = self.db.record_model_prediction(pick)
        decision_id = self.db.record_model_decision(
            prediction_id=prediction_id,
            decision='SELECTED',
            selected_rank=1,
            quantity=2,
            raw_decision=pick,
        )
        tid = self.db.log_trade(
            'SPY', '2026-06-19', 500.0, 'PCS', 0.80, 0.72,
            status='EXECUTED',
            legs={'short_strike': 500.0, 'long_strike': 495.0},
            contracts=2,
            model_prediction_id=prediction_id,
            model_decision_id=decision_id,
        )
        self.db.confirm_close(tid, pnl=120.0, close_order_id='CLOSE-1')

        trade = self.db.get_trade(tid)
        prediction = self.db.get_model_predictions()[0]
        decision = self.db.get_model_decisions()[0]

        self.assertEqual(trade['model_prediction_id'], prediction_id)
        self.assertEqual(trade['model_decision_id'], decision_id)
        self.assertEqual(prediction['model_version'], 'champion_v1')
        self.assertEqual(len(prediction['features_hash']), 64)
        self.assertEqual(decision['trade_id'], tid)
        self.assertEqual(decision['final_outcome'], 'CLOSED')
        self.assertEqual(decision['realized_pnl'], 120.0)

    def test_model_rejection_decision_stores_risk_gate(self):
        prediction_id = self.db.record_model_prediction({
            'symbol': 'QQQ',
            'strategy': 'CCS',
            'expiry': '2026-06-19',
            'model_version': 'candidate_v2',
            'score': -5.0,
            'features': {'dte': 26},
        })

        decision_id = self.db.record_model_decision(
            prediction_id=prediction_id,
            decision='REJECTED',
            risk_gate='Portfolio gamma risk',
            reject_reason='stress cap exceeded',
        )

        decision = self.db.get_model_decisions()[0]
        self.assertEqual(decision['id'], decision_id)
        self.assertEqual(decision['decision'], 'REJECTED')
        self.assertEqual(decision['risk_gate'], 'Portfolio gamma risk')
        self.assertEqual(decision['reject_reason'], 'stress cap exceeded')


if __name__ == '__main__':
    unittest.main()
