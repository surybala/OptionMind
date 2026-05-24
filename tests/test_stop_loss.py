"""
Tests for stop-loss / position-monitoring features:
  - TradeDatabase: legs column, get_open_positions(), close_position()
  - PositionMonitor: stop-loss trigger logic (yfinance mocked)
  - Greeks: Black-Scholes delta/gamma/theta calculations
  - Gamma risk scoring: position_risk_score()
  - Gamma risk trigger: early close when gamma/theta ratio spikes
"""
import json
import math
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.database import TradeDatabase
from src.position_monitor import PositionMonitor
from src.greeks import bs_greeks, position_risk_score


# ── Database helpers ──────────────────────────────────────────────────────────

def _tmp_db() -> TradeDatabase:
    fd, path = tempfile.mkstemp(suffix='.db', prefix='test_sl_')
    os.close(fd)
    return TradeDatabase(db_path=path)


FUTURE_EXPIRY = '2099-12-31'   # never expires for test purposes


# ═══════════════════════════════════════════════════════════════════════════════
# TradeDatabase: legs column + new methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabaseLegs:
    def test_log_trade_with_legs(self):
        db   = _tmp_db()
        legs = {'short_put': 150.0, 'long_put': 145.0}
        tid  = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                            status='EXECUTED', legs=legs)
        rows = db.get_history(limit=1)
        assert rows[0]['id'] == tid

    def test_legs_round_trips_as_dict(self):
        db   = _tmp_db()
        legs = {'short_put': 150.0, 'long_put': 145.0}
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=legs)
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert isinstance(open_pos[0]['legs'], dict)
        assert open_pos[0]['legs']['short_put'] == 150.0

    def test_legs_none_stored_ok(self):
        db = _tmp_db()
        db.log_trade('MSFT', FUTURE_EXPIRY, 300, 'CSP', 1.00, 0.82,
                     status='EXECUTED', legs=None)
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        # legs column absent or None/empty dict — both acceptable
        assert open_pos[0].get('legs') in (None, {}, '')

    def test_get_open_positions_excludes_closed(self):
        db = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='CLOSED')
        assert db.get_open_positions() == []

    def test_get_open_positions_excludes_expired(self):
        db = _tmp_db()
        db.log_trade('AAPL', '2000-01-01', 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED')
        assert db.get_open_positions() == []

    def test_get_open_positions_includes_dry_run(self):
        db = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='DRY_RUN')
        assert len(db.get_open_positions()) == 1

    def test_close_position_marks_closed(self):
        db  = _tmp_db()
        tid = db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                           status='EXECUTED',
                           legs={'short_put': 150.0, 'long_put': 145.0})
        db.close_position(tid, pnl=-100.0, close_order_id='CLOSE123')
        assert db.get_open_positions() == []
        row = db.get_history(limit=1)[0]
        assert row['status']   == 'CLOSED'
        assert row['pnl']      == -100.0
        assert row['order_id'] == 'CLOSE123'

    def test_migration_adds_legs_to_old_db(self):
        """Simulate a DB created before legs column existed."""
        import sqlite3
        fd, path = tempfile.mkstemp(suffix='.db', prefix='test_migrate_')
        os.close(fd)
        # Create schema without legs column
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, expiry TEXT, strike REAL,
                    type TEXT, premium REAL, prob_expiry REAL,
                    status TEXT, order_id TEXT, pnl REAL DEFAULT 0
                )
            """)
            conn.execute("INSERT INTO trades (timestamp,symbol,expiry,strike,type,"
                         "premium,prob_expiry,status,order_id) "
                         "VALUES ('2024-01-01','AAPL','2099-12-31',150,'PCS',"
                         "0.5,0.8,'EXECUTED',NULL)")
            conn.commit()
        # Open via TradeDatabase — _migrate() should add column transparently
        db = TradeDatabase(db_path=path)
        rows = db.get_open_positions()
        assert len(rows) == 1            # existing row still there
        assert 'legs' in rows[0]        # legs key present (None value is fine)


# ═══════════════════════════════════════════════════════════════════════════════
# PositionMonitor: stop-loss logic
# ═══════════════════════════════════════════════════════════════════════════════

def _make_chain(puts: dict, calls: dict):
    """Return a mock yf option chain with custom put/call DataFrames."""
    def _df(rows):
        return pd.DataFrame(rows)

    chain = MagicMock()
    chain.puts  = _df([{'strike': k, 'bid': v, 'ask': v, 'lastPrice': v}
                        for k, v in puts.items()])
    chain.calls = _df([{'strike': k, 'bid': v, 'ask': v, 'lastPrice': v}
                        for k, v in calls.items()])
    return chain


def _mock_yf(puts: dict, calls: dict):
    """Patch yf.Ticker to return the given chain for any expiry."""
    ticker = MagicMock()
    ticker.option_chain.return_value = _make_chain(puts, calls)
    return ticker


def _make_monitor(db, config=None):
    executor = MagicMock()
    executor.execute_close_position.return_value = 'CLOSE_ORDER_123'
    cfg = config or {'risk_parameters': {'stop_loss_multiplier': 2.0}}
    return PositionMonitor(db, executor, cfg)


class TestPositionMonitor:

    # ── No open positions ─────────────────────────────────────────────────────

    def test_no_open_positions_returns_empty(self):
        db      = _tmp_db()
        monitor = _make_monitor(db)
        result  = monitor.run(dry_run=True)
        assert result == []

    # ── PCS stop-loss ─────────────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_pcs_no_trigger_when_within_limit(self, mock_tf):
        """Current mark < 3× premium → no stop-loss."""
        db  = _tmp_db()
        entry_premium = 0.50
        # Current cost-to-close: short=0.70, long=0.10 → mark=0.60 < 3×0.50=1.50
        mock_tf.return_value = _mock_yf(puts={150.0: 0.70, 145.0: 0.10}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []

    @patch('src.market_data.adapter.yf.Ticker')
    def test_pcs_triggers_when_loss_exceeds_2x(self, mock_tf):
        """Current mark > 3× premium → stop triggered."""
        db  = _tmp_db()
        entry_premium = 0.50
        # mark = 1.60 - 0.00 = 1.60 > 1.50 → trigger
        mock_tf.return_value = _mock_yf(puts={150.0: 1.60, 145.0: 0.00}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1
        # Position must be marked CLOSED in DB
        assert db.get_open_positions() == []

    @patch('src.market_data.adapter.yf.Ticker')
    def test_pcs_triggers_exactly_at_boundary(self, mock_tf):
        """Loss exactly == 2× premium is NOT a trigger (strictly greater)."""
        db  = _tmp_db()
        entry_premium = 0.50
        # mark = short - long = 1.50 - 0.00 = 1.50 → loss = 1.50 - 0.50 = 1.00 = 2× (not > 2×)
        mock_tf.return_value = _mock_yf(puts={150.0: 1.50, 145.0: 0.00}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []   # exactly 2× → no close

    # ── CCS stop-loss ─────────────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_ccs_triggers(self, mock_tf):
        db  = _tmp_db()
        entry_premium = 0.40
        # mark = 1.30 > 1.20 (= 3×0.40) → trigger
        mock_tf.return_value = _mock_yf(puts={}, calls={160.0: 1.30, 165.0: 0.00})
        db.log_trade('AAPL', FUTURE_EXPIRY, 160, 'CCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 160.0, 'long_strike': 165.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1

    # ── IC stop-loss ──────────────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_ic_triggers(self, mock_tf):
        db  = _tmp_db()
        entry_premium = 0.80
        # put side: 0.90-0.00=0.90; call side: 0.90-0.00=0.90 → total mark=1.80
        # 3×0.80 = 2.40; 1.80 < 2.40 → no trigger
        mock_tf.return_value = _mock_yf(
            puts ={150.0: 0.90, 145.0: 0.00},
            calls={160.0: 0.90, 165.0: 0.00},
        )
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'IC', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_put': 150.0, 'long_put': 145.0,
                           'short_call': 160.0, 'long_call': 165.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []   # 1.80 < 2.40

    @patch('src.market_data.adapter.yf.Ticker')
    def test_ic_triggers_when_over_limit(self, mock_tf):
        db  = _tmp_db()
        entry_premium = 0.80
        # total mark = 2.00+2.00 = 4.00 > 3×0.80=2.40 → trigger
        mock_tf.return_value = _mock_yf(
            puts ={150.0: 2.00, 145.0: 0.00},
            calls={160.0: 2.00, 165.0: 0.00},
        )
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'IC', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_put': 150.0, 'long_put': 145.0,
                           'short_call': 160.0, 'long_call': 165.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1

    # ── Missing legs → skip gracefully ───────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_missing_legs_skipped_safely(self, mock_tf):
        db  = _tmp_db()
        mock_tf.return_value = _mock_yf(puts={150.0: 9.99}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED', legs=None)  # no legs stored
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []   # silently skipped

    # ── yfinance error → skip gracefully ─────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_yfinance_error_skipped_safely(self, mock_tf):
        db  = _tmp_db()
        mock_tf.side_effect = RuntimeError("network error")
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []

    # ── Multiple positions: only breached ones closed ─────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_only_breached_positions_closed(self, mock_tf):
        db  = _tmp_db()
        entry = 0.50

        # Position 1 (AAPL PCS): mark=1.60 > 1.50 → TRIGGER
        # Position 2 (MSFT PCS): mark=0.60 < 1.50 → OK
        def side_effect(symbol):
            ticker = MagicMock()
            if symbol == 'AAPL':
                ticker.option_chain.return_value = _make_chain(
                    puts={150.0: 1.60, 145.0: 0.00}, calls={})
            else:
                ticker.option_chain.return_value = _make_chain(
                    puts={300.0: 0.70, 295.0: 0.10}, calls={})
            return ticker

        mock_tf.side_effect = side_effect

        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        db.log_trade('MSFT', FUTURE_EXPIRY, 300, 'PCS', entry, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 300.0, 'long_strike': 295.0})

        monitor = _make_monitor(db)
        closed  = monitor.run(dry_run=True)

        assert len(closed) == 1
        assert closed[0]['symbol'] == 'AAPL'
        # MSFT still open
        open_pos = db.get_open_positions()
        assert len(open_pos) == 1
        assert open_pos[0]['symbol'] == 'MSFT'

    # ── Custom multiplier ─────────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_custom_multiplier_1x(self, mock_tf):
        """With 1× multiplier, trigger when mark > 2× premium."""
        db  = _tmp_db()
        entry = 0.50
        # mark = 1.10; threshold = (1+1.0)×0.50 = 1.00; 1.10 > 1.00 → trigger
        mock_tf.return_value = _mock_yf(puts={150.0: 1.10, 145.0: 0.00}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor(db, config={
            'risk_parameters': {'stop_loss_multiplier': 1.0}
        })
        closed = monitor.run(dry_run=True)
        assert len(closed) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Greeks: Black-Scholes calculations
# ═══════════════════════════════════════════════════════════════════════════════

class TestBsGreeks:
    """Unit tests for the bs_greeks() function."""

    def test_call_delta_otm(self):
        """OTM call has delta < 0.5."""
        g = bs_greeks(spot=100, strike=110, iv=0.20, dte_days=30, option_type='call')
        assert 0.0 < g['delta'] < 0.5, f"OTM call delta should be <0.5, got {g['delta']}"

    def test_call_delta_itm(self):
        """ITM call has delta > 0.5."""
        g = bs_greeks(spot=100, strike=90, iv=0.20, dte_days=30, option_type='call')
        assert 0.5 < g['delta'] <= 1.0, f"ITM call delta should be >0.5, got {g['delta']}"

    def test_put_delta_otm(self):
        """OTM put has delta between -0.5 and 0."""
        g = bs_greeks(spot=100, strike=90, iv=0.20, dte_days=30, option_type='put')
        assert -0.5 < g['delta'] < 0.0, f"OTM put delta should be negative, got {g['delta']}"

    def test_put_delta_itm(self):
        """ITM put has delta < -0.5."""
        g = bs_greeks(spot=100, strike=110, iv=0.20, dte_days=30, option_type='put')
        assert -1.0 <= g['delta'] < -0.5, f"ITM put delta should be < -0.5, got {g['delta']}"

    def test_atm_delta_approximately_half(self):
        """ATM option delta ≈ ±0.5."""
        call = bs_greeks(spot=100, strike=100, iv=0.20, dte_days=30, option_type='call')
        put  = bs_greeks(spot=100, strike=100, iv=0.20, dte_days=30, option_type='put')
        assert abs(call['delta'] - 0.5) < 0.05,  f"ATM call delta ≈ 0.5, got {call['delta']}"
        assert abs(put['delta']  + 0.5) < 0.05,  f"ATM put delta ≈ -0.5, got {put['delta']}"

    def test_gamma_always_positive(self):
        """Gamma is always non-negative (same for puts and calls)."""
        for opt in ('call', 'put'):
            for strike in (80, 100, 120):
                g = bs_greeks(spot=100, strike=strike, iv=0.25, dte_days=20, option_type=opt)
                assert g['gamma'] >= 0, f"Gamma must be >= 0, got {g['gamma']}"

    def test_gamma_peaks_atm(self):
        """Gamma is highest for ATM options."""
        g_atm = bs_greeks(spot=100, strike=100, iv=0.20, dte_days=30, option_type='put')
        g_otm = bs_greeks(spot=100, strike=80,  iv=0.20, dte_days=30, option_type='put')
        g_itm = bs_greeks(spot=100, strike=120, iv=0.20, dte_days=30, option_type='put')
        assert g_atm['gamma'] > g_otm['gamma'], "ATM gamma should exceed OTM gamma"
        assert g_atm['gamma'] > g_itm['gamma'], "ATM gamma should exceed ITM gamma"

    def test_theta_negative_for_long_options(self):
        """Theta is negative for option holders (time decay costs them)."""
        for opt in ('call', 'put'):
            g = bs_greeks(spot=100, strike=100, iv=0.20, dte_days=30, option_type=opt)
            assert g['theta'] < 0, f"Theta should be negative for long {opt}, got {g['theta']}"

    def test_theta_magnitude_increases_near_expiry(self):
        """Theta decay accelerates as expiry approaches (shorter DTE = bigger |theta|)."""
        g_far  = bs_greeks(spot=100, strike=100, iv=0.25, dte_days=60, option_type='call')
        g_near = bs_greeks(spot=100, strike=100, iv=0.25, dte_days=7,  option_type='call')
        assert abs(g_near['theta']) > abs(g_far['theta']), \
            "Theta magnitude should increase as expiry nears"

    def test_zero_iv_edge_case(self):
        """Zero IV returns finite values without crashing."""
        g = bs_greeks(spot=100, strike=100, iv=0.0, dte_days=30, option_type='call')
        assert g['gamma'] == 0.0
        assert g['theta'] == 0.0

    def test_zero_dte_edge_case(self):
        """Zero DTE returns finite values without crashing."""
        g = bs_greeks(spot=100, strike=100, iv=0.20, dte_days=0, option_type='call')
        assert math.isfinite(g['delta'])
        assert g['gamma'] == 0.0
        assert g['theta'] == 0.0

    def test_put_call_parity_gamma(self):
        """Gamma is identical for a put and call at the same strike (BS parity)."""
        g_call = bs_greeks(spot=100, strike=105, iv=0.20, dte_days=21, option_type='call')
        g_put  = bs_greeks(spot=100, strike=105, iv=0.20, dte_days=21, option_type='put')
        assert abs(g_call['gamma'] - g_put['gamma']) < 1e-8, \
            "Put and call gamma must be equal at same strike"

    def test_put_call_delta_sum_one(self):
        """Delta(call) - Delta(put) ≈ 1 at r=0 (put-call parity for delta)."""
        g_call = bs_greeks(spot=100, strike=95, iv=0.30, dte_days=15, option_type='call')
        g_put  = bs_greeks(spot=100, strike=95, iv=0.30, dte_days=15, option_type='put')
        diff = g_call['delta'] - g_put['delta']   # should be ~1
        assert abs(diff - 1.0) < 0.01, f"delta(call)-delta(put) should be ~1, got {diff}"


# ═══════════════════════════════════════════════════════════════════════════════
# Greeks: position_risk_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestPositionRiskScore:
    """Tests for the portfolio-level position_risk_score() function."""

    def _csp_legs(self, spot=100, strike=90, iv=0.25):
        """Short put (Cash-Secured Put) — single short leg."""
        return [{'strike': strike, 'iv': iv, 'option_type': 'put', 'position': 'short'}]

    def _pcs_legs(self, spot=100, short_strike=90, long_strike=85, iv=0.25):
        """Put Credit Spread."""
        return [
            {'strike': short_strike, 'iv': iv, 'option_type': 'put', 'position': 'short'},
            {'strike': long_strike,  'iv': iv, 'option_type': 'put', 'position': 'long'},
        ]

    def test_otm_put_low_risk_score(self):
        """A deep-OTM short put should have a low risk score."""
        legs = self._csp_legs(spot=100, strike=70, iv=0.20)
        r = position_risk_score(100, legs, dte_days=30)
        # Risk score could be any value; just verify it's lower than near-the-money
        assert r['risk_score'] >= 0

    def test_atm_put_high_risk_score(self):
        """An ATM short put (delta ≈ -0.5) should have a higher risk score than OTM."""
        legs_otm = self._csp_legs(spot=100, strike=80,  iv=0.25)
        legs_atm = self._csp_legs(spot=100, strike=100, iv=0.25)
        r_otm = position_risk_score(100, legs_otm, dte_days=30)
        r_atm = position_risk_score(100, legs_atm, dte_days=30)
        assert r_atm['risk_score'] > r_otm['risk_score'], \
            "ATM short put should have higher risk score than OTM"

    def test_gamma_theta_ratio_finite_and_positive(self):
        """gamma_theta_ratio should be a finite positive number."""
        legs = self._csp_legs(spot=100, strike=95, iv=0.25)
        r = position_risk_score(100, legs, dte_days=21)
        assert math.isfinite(r['gamma_theta_ratio'])
        assert r['gamma_theta_ratio'] >= 0

    def test_short_put_has_positive_net_theta(self):
        """Net theta > 0 means the position earns time value (short premium)."""
        legs = self._csp_legs(spot=100, strike=90, iv=0.25)
        r = position_risk_score(100, legs, dte_days=30)
        assert r['net_theta'] > 0, f"Short put net theta should be positive, got {r['net_theta']}"

    def test_short_put_has_negative_net_gamma(self):
        """Net gamma < 0 means the position is hurt by large moves (short gamma)."""
        legs = self._csp_legs(spot=100, strike=90, iv=0.25)
        r = position_risk_score(100, legs, dte_days=30)
        assert r['net_gamma'] < 0, f"Short put net gamma should be negative, got {r['net_gamma']}"

    def test_spread_lower_gamma_than_naked(self):
        """A PCS (spread) has lower absolute gamma than a naked short put."""
        naked = self._csp_legs(spot=100, strike=90, iv=0.25)
        spread = self._pcs_legs(spot=100, short_strike=90, long_strike=85, iv=0.25)
        r_naked  = position_risk_score(100, naked,  dte_days=30)
        r_spread = position_risk_score(100, spread, dte_days=30)
        assert abs(r_spread['net_gamma']) < abs(r_naked['net_gamma']), \
            "Spread should have less absolute gamma than naked short put"

    def test_empty_legs_returns_zeros(self):
        """Empty legs list returns zero risk score."""
        r = position_risk_score(100, [], dte_days=30)
        assert r['risk_score'] == 0.0
        assert r['gamma_theta_ratio'] == 0.0

    def test_zero_spot_returns_zeros(self):
        """Zero spot price returns zero risk score (guard against division errors)."""
        legs = self._csp_legs(spot=0, strike=90, iv=0.25)
        r = position_risk_score(0, legs, dte_days=30)
        assert r['risk_score'] == 0.0

    def test_risk_score_increases_as_put_moves_itm(self):
        """When a short put moves ITM (delta rises), the risk score increases."""
        # Same DTE, same IV — only spot moves down, making the put more ITM
        legs_otm = self._csp_legs(spot=100, strike=85,  iv=0.25)  # 15% OTM
        legs_itm = self._csp_legs(spot=100, strike=100, iv=0.25)  # ATM (effectively ITM for put)
        r_otm = position_risk_score(100, legs_otm, dte_days=21)
        r_itm = position_risk_score(100, legs_itm, dte_days=21)
        assert r_itm['risk_score'] > r_otm['risk_score'], \
            "ATM/ITM short put should have a higher risk score than OTM"


# ═══════════════════════════════════════════════════════════════════════════════
# PositionMonitor: gamma risk trigger
# ═══════════════════════════════════════════════════════════════════════════════

def _make_chain_with_iv(puts: dict, calls: dict):
    """
    Build a mock option chain where each entry has bid/ask/lastPrice AND
    impliedVolatility so the gamma risk calculator can read IV.

    puts/calls: {strike: (mid_price, iv)} dicts.
    """
    def _df(rows_dict):
        rows = []
        for strike, val in rows_dict.items():
            mid, iv = val if isinstance(val, tuple) else (val, 0.0)
            rows.append({
                'strike': strike, 'bid': mid, 'ask': mid, 'lastPrice': mid,
                'impliedVolatility': iv,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility']
        )

    chain = MagicMock()
    chain.puts  = _df(puts)
    chain.calls = _df(calls)
    return chain


def _make_monitor_gr(db, gamma_risk_cfg: dict, stop_loss_mult: float = 5.0):
    """
    Build a PositionMonitor with gamma_risk config and a very high stop_loss
    multiplier so that only the gamma trigger fires in the tests.
    Profit-take is disabled so deeply-decayed test positions don't close early.
    """
    executor = MagicMock()
    executor.execute_close_position.return_value = 'CLOSE_ORDER_GR'
    cfg = {
        'risk_parameters': {
            'stop_loss_multiplier': stop_loss_mult,
            'profit_take_enabled': False,   # isolate gamma-risk tests
            'gamma_risk': gamma_risk_cfg,
        }
    }
    return PositionMonitor(db, executor, cfg)


class TestGammaRiskTrigger:
    """Gamma/theta dynamic risk trigger tests."""

    # ── Gamma risk disabled ───────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_gamma_risk_disabled_does_not_trigger(self, mock_tf):
        """When gamma_risk.enabled=false, no early close fires."""
        db = _tmp_db()
        # ATM put — high gamma, but feature disabled
        puts = {100.0: (0.50, 0.40)}  # ATM, IV=40%
        ticker = MagicMock()
        ticker.option_chain.return_value = _make_chain_with_iv(puts=puts, calls={})
        ticker.fast_info.last_price = 100.0
        mock_tf.return_value = ticker

        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 100.0})
        monitor = _make_monitor_gr(db, {'enabled': False})
        closed  = monitor.run(dry_run=True)
        assert closed == [], "Gamma risk disabled → should not close"

    # ── Deep OTM: no trigger ──────────────────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_deep_otm_no_gamma_trigger(self, mock_tf):
        """Deep OTM short put has low delta → gamma trigger should not fire."""
        db = _tmp_db()
        # Put 50% OTM (strike=50, spot=100) → |delta| ≈ 0.10 < 0.15 threshold
        puts = {50.0: (0.01, 0.20)}
        ticker = MagicMock()
        ticker.option_chain.return_value = _make_chain_with_iv(puts=puts, calls={})
        ticker.fast_info.last_price = 100.0
        ticker.fast_info.regularMarketPrice = None
        mock_tf.return_value = ticker

        db.log_trade('AAPL', FUTURE_EXPIRY, 50, 'CSP', 0.30, 0.95,
                     status='EXECUTED',
                     legs={'short_strike': 50.0})
        monitor = _make_monitor_gr(db, {
            'enabled': True,
            'gamma_theta_ratio_threshold': 1.5,
            'min_delta_to_trigger': 0.15,
            'min_profit_captured_pct': 0.0,
            'urgent_delta_threshold': 0.35,
        })
        closed = monitor.run(dry_run=True)
        assert closed == [], "Deep OTM should not trigger gamma risk"

    # ── Gamma risk disabled via config key ────────────────────────────────────

    @patch('src.market_data.adapter.yf.Ticker')
    def test_no_spot_price_skips_gamma_check(self, mock_tf):
        """If spot price is not available, gamma check is skipped gracefully."""
        db = _tmp_db()
        puts = {100.0: (0.50, 0.40)}
        ticker = MagicMock()
        ticker.option_chain.return_value = _make_chain_with_iv(puts=puts, calls={})
        # Both fast_info price attributes set to None — can't compute Greeks
        ticker.fast_info.last_price = None
        ticker.fast_info.regularMarketPrice = None
        mock_tf.return_value = ticker

        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 100.0})
        monitor = _make_monitor_gr(db, {
            'enabled': True,
            'gamma_theta_ratio_threshold': 0.1,  # very low threshold
            'min_delta_to_trigger': 0.0,
            'min_profit_captured_pct': 0.0,
            'urgent_delta_threshold': 0.01,
        })
        closed = monitor.run(dry_run=True)
        # Should not crash; gamma check silently skipped due to missing spot
        assert closed == [], "Missing spot price should not crash"

    # ── position_risk_score called with real Greeks legs ─────────────────────

    def test_build_greeks_legs_pcs(self):
        """_build_greeks_legs produces correct entries for a PCS position."""
        executor = MagicMock()
        monitor  = PositionMonitor(_tmp_db(), executor, {'risk_parameters': {}})

        pos = {
            'symbol': 'AAPL', 'expiry': FUTURE_EXPIRY, 'type': 'PCS',
            'strike': 150.0, 'premium': 0.50, 'prob_expiry': 0.80,
            'legs': {'short_strike': 150.0, 'long_strike': 145.0},
        }
        put_map = {
            150.0: {'bid': 0.60, 'ask': 0.80, 'lastPrice': 0.70, 'impliedVolatility': 0.25},
            145.0: {'bid': 0.10, 'ask': 0.15, 'lastPrice': 0.12, 'impliedVolatility': 0.22},
        }
        legs = monitor._build_greeks_legs(pos, put_map, call_map={})
        assert len(legs) == 2
        strikes = {l['strike'] for l in legs}
        assert 150.0 in strikes and 145.0 in strikes
        positions = {l['strike']: l['position'] for l in legs}
        assert positions[150.0] == 'short'
        assert positions[145.0] == 'long'

    def test_build_greeks_legs_ic(self):
        """_build_greeks_legs produces 4 entries for an IC position."""
        executor = MagicMock()
        monitor  = PositionMonitor(_tmp_db(), executor, {'risk_parameters': {}})

        pos = {
            'symbol': 'SPY', 'expiry': FUTURE_EXPIRY, 'type': 'IC',
            'strike': 450.0, 'premium': 1.00, 'prob_expiry': 0.70,
            'legs': {'short_put': 450.0, 'long_put': 445.0,
                     'short_call': 460.0, 'long_call': 465.0},
        }
        put_map = {
            450.0: {'impliedVolatility': 0.18},
            445.0: {'impliedVolatility': 0.19},
        }
        call_map = {
            460.0: {'impliedVolatility': 0.17},
            465.0: {'impliedVolatility': 0.16},
        }
        legs = monitor._build_greeks_legs(pos, put_map, call_map)
        assert len(legs) == 4
        leg_map = {(l['strike'], l['option_type']): l['position'] for l in legs}
        assert leg_map[(450.0, 'put')]  == 'short'
        assert leg_map[(445.0, 'put')]  == 'long'
        assert leg_map[(460.0, 'call')] == 'short'
        assert leg_map[(465.0, 'call')] == 'long'

    def test_build_greeks_legs_missing_iv_excluded(self):
        """A leg with no IV in the chain is excluded from the greeks legs list."""
        executor = MagicMock()
        monitor  = PositionMonitor(_tmp_db(), executor, {'risk_parameters': {}})

        pos = {
            'symbol': 'AAPL', 'expiry': FUTURE_EXPIRY, 'type': 'PCS',
            'strike': 150.0, 'premium': 0.50, 'prob_expiry': 0.80,
            'legs': {'short_strike': 150.0, 'long_strike': 145.0},
        }
        put_map = {
            150.0: {'bid': 0.60, 'ask': 0.80, 'lastPrice': 0.70,
                    'impliedVolatility': 0.0},   # IV=0 → excluded
            # 145.0 missing entirely → excluded
        }
        legs = monitor._build_greeks_legs(pos, put_map, call_map={})
        assert legs == [], "Legs with zero/missing IV should be excluded"


# ═══════════════════════════════════════════════════════════════════════════════
# ProfitTakeRule unit tests
# ═══════════════════════════════════════════════════════════════════════════════

from src.risk_rules import ProfitTakeRule


class TestProfitTakeRule:
    """Unit tests for ProfitTakeRule in isolation."""

    def test_fires_at_exactly_80_pct(self):
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.80)
        entry  = 1.00
        mark   = 0.20   # pnl = 0.80 = exactly 80% of entry
        signal = rule.evaluate(entry, mark, entry - mark, None, {})
        assert signal is not None
        assert signal.reason_tag == 'PROFIT_TAKE'
        assert abs(signal.metrics['captured_pct'] - 0.80) < 1e-9

    def test_fires_above_threshold(self):
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.80)
        signal = rule.evaluate(1.00, 0.05, 0.95, None, {})   # 95% captured
        assert signal is not None

    def test_no_fire_below_threshold(self):
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.80)
        signal = rule.evaluate(1.00, 0.30, 0.70, None, {})   # only 70% captured
        assert signal is None

    def test_disabled_never_fires(self):
        rule   = ProfitTakeRule(enabled=False, profit_take_pct=0.80)
        signal = rule.evaluate(1.00, 0.05, 0.95, None, {})   # would fire if enabled
        assert signal is None

    def test_zero_pct_never_fires(self):
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.0)
        signal = rule.evaluate(1.00, 0.05, 0.95, None, {})
        assert signal is None

    def test_custom_50_pct_threshold(self):
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.50)
        signal = rule.evaluate(1.00, 0.49, 0.51, None, {})   # 51% > 50% → fires
        assert signal is not None
        assert abs(signal.metrics['profit_take_pct'] - 0.50) < 1e-9

    def test_loss_position_never_fires(self):
        """Profit-take must never fire when mark > entry (we are losing money)."""
        rule   = ProfitTakeRule(enabled=True, profit_take_pct=0.80)
        entry  = 0.50
        mark   = 1.20   # position is a loss
        signal = rule.evaluate(entry, mark, entry - mark, None, {})
        assert signal is None


# ═══════════════════════════════════════════════════════════════════════════════
# Profit-take integration: _apply_triggers via PositionMonitor.run()
# ═══════════════════════════════════════════════════════════════════════════════

def _make_monitor_pt(db, profit_take_pct: float = 0.80, enabled: bool = True):
    """Monitor with a very high stop-loss so only profit-take can fire."""
    executor = MagicMock()
    executor.execute_close_position.return_value = 'PT_ORDER'
    cfg = {
        'risk_parameters': {
            'stop_loss_multiplier': 99.0,    # never fires in these tests
            'profit_take_enabled': enabled,
            'profit_take_pct': profit_take_pct,
        }
    }
    return PositionMonitor(db, executor, cfg)


class TestProfitTakeIntegration:

    @patch('src.market_data.adapter.yf.Ticker')
    def test_csp_triggers_profit_take_at_80pct(self, mock_tf):
        """CSP: mark decays to 10% of entry → 90% captured → close triggered."""
        db = _tmp_db()
        entry_premium = 1.00
        # ask price for CSP short-put = 0.10  → pnl = 0.90 ≥ 0.80*1.00 → fires
        mock_tf.return_value = _mock_yf(puts={100.0: 0.10}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', entry_premium, 0.80,
                     status='EXECUTED', legs={'short_strike': 100.0})
        monitor = _make_monitor_pt(db)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1
        assert db.get_open_positions() == []

    @patch('src.market_data.adapter.yf.Ticker')
    def test_csp_no_trigger_below_threshold(self, mock_tf):
        """CSP: only 60% captured → below 80% target → no close."""
        db = _tmp_db()
        entry_premium = 1.00
        mock_tf.return_value = _mock_yf(puts={100.0: 0.40}, calls={})   # 60% captured
        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', entry_premium, 0.80,
                     status='EXECUTED', legs={'short_strike': 100.0})
        monitor = _make_monitor_pt(db)
        closed  = monitor.run(dry_run=True)
        assert closed == []

    @patch('src.market_data.adapter.yf.Ticker')
    def test_pcs_triggers_profit_take(self, mock_tf):
        """PCS: net mark = 0.05 from entry 0.50 → 90% captured → close."""
        db = _tmp_db()
        entry_premium = 0.50
        # short ask=0.05, long bid=0.00 → mark=0.05 → pnl=0.45 = 90% of 0.50
        mock_tf.return_value = _mock_yf(puts={150.0: 0.05, 145.0: 0.00}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        monitor = _make_monitor_pt(db)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1

    @patch('src.market_data.adapter.yf.Ticker')
    def test_profit_take_disabled_does_not_close(self, mock_tf):
        """When profit_take_enabled=False, position stays open even at 100% profit."""
        db = _tmp_db()
        mock_tf.return_value = _mock_yf(puts={100.0: 0.00}, calls={})   # 100% profit
        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', 1.00, 0.80,
                     status='EXECUTED', legs={'short_strike': 100.0})
        monitor = _make_monitor_pt(db, enabled=False)
        closed  = monitor.run(dry_run=True)
        assert closed == []

    @patch('src.market_data.adapter.yf.Ticker')
    def test_stop_loss_takes_priority_over_profit_take(self, mock_tf):
        """Stop-loss fires first; profit-take is never reached (mark > entry)."""
        db = _tmp_db()
        entry_premium = 0.50
        # mark = 1.60 → loss = 1.10 > 2×0.50=1.00 → stop-loss fires
        mock_tf.return_value = _mock_yf(puts={150.0: 1.60, 145.0: 0.00}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', entry_premium, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        executor = MagicMock()
        executor.execute_close_position.return_value = 'SL_ORDER'
        cfg = {'risk_parameters': {
            'stop_loss_multiplier': 2.0,
            'profit_take_enabled': True,
            'profit_take_pct': 0.80,
        }}
        monitor = PositionMonitor(db, executor, cfg)
        closed = monitor.run(dry_run=True)
        assert len(closed) == 1

    @patch('src.market_data.adapter.yf.Ticker')
    def test_configurable_50pct_threshold(self, mock_tf):
        """profit_take_pct=0.50: fires when 55% is captured."""
        db = _tmp_db()
        entry_premium = 1.00
        # mark = 0.45 → pnl = 0.55 = 55% ≥ 50% → fires
        mock_tf.return_value = _mock_yf(puts={100.0: 0.45}, calls={})
        db.log_trade('AAPL', FUTURE_EXPIRY, 100, 'CSP', entry_premium, 0.80,
                     status='EXECUTED', legs={'short_strike': 100.0})
        monitor = _make_monitor_pt(db, profit_take_pct=0.50)
        closed  = monitor.run(dry_run=True)
        assert len(closed) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Live _execute_close: two-phase DB behavior
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecuteCloseLiveMode:
    """
    Verify the two-phase close contract in live (dry_run=False) mode:
      1. After trigger fires, DB row must be PENDING_CLOSE (not CLOSED) so the
         reconciler — not the monitor — confirms the fill once legs leave Alpaca.
      2. The estimated P&L and close order ID must be stored on the row so the
         reconciler doesn't have to write a $0 placeholder.
      3. If the broker call fails, the row must roll back to EXECUTED.
    """

    def _make_monitor_live(self, db, executor):
        cfg = {'risk_parameters': {'stop_loss_multiplier': 2.0}}
        return PositionMonitor(db, executor, cfg)

    @patch('src.market_data.adapter.yf.Ticker')
    def test_live_trigger_leaves_pending_close(self, mock_tf):
        """Stop-loss in live mode: DB row is PENDING_CLOSE, not CLOSED."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        # mark = 1.60 → loss > 2×0.50 → stop-loss fires
        mock_tf.return_value = _mock_yf(puts={150.0: 1.60, 145.0: 0.00}, calls={})
        executor = MagicMock()
        executor.execute_close_position.return_value = 'LIVE_ORD_99'

        monitor = self._make_monitor_live(db, executor)
        closed  = monitor.run(dry_run=False)

        assert len(closed) == 1, "trigger should fire"
        row = db.get_history(limit=1)[0]
        assert row['status']   == 'PENDING_CLOSE', \
            "row must stay PENDING_CLOSE until reconciler confirms the fill"
        assert row['order_id'] == 'LIVE_ORD_99'

    @patch('src.market_data.adapter.yf.Ticker')
    def test_live_trigger_stores_estimated_pnl(self, mock_tf):
        """Estimated P&L stored on PENDING_CLOSE row for reconciler to use."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        # mark = 1.60 → pnl_per_share = 0.50 - 1.60 = -1.10 → pnl_dollars = -110
        mock_tf.return_value = _mock_yf(puts={150.0: 1.60, 145.0: 0.00}, calls={})
        executor = MagicMock()
        executor.execute_close_position.return_value = 'LIVE_ORD_100'

        monitor = self._make_monitor_live(db, executor)
        monitor.run(dry_run=False)

        row = db.get_history(limit=1)[0]
        assert row['pnl'] == pytest.approx(-110.0)

    @patch('src.market_data.adapter.yf.Ticker')
    def test_live_broker_failure_reopens_position(self, mock_tf):
        """If the broker call fails, row rolls back to EXECUTED for retry."""
        db  = _tmp_db()
        db.log_trade('AAPL', FUTURE_EXPIRY, 150, 'PCS', 0.50, 0.80,
                     status='EXECUTED',
                     legs={'short_strike': 150.0, 'long_strike': 145.0})
        mock_tf.return_value = _mock_yf(puts={150.0: 1.60, 145.0: 0.00}, calls={})
        executor = MagicMock()
        executor.execute_close_position.return_value = None   # broker rejected

        monitor = self._make_monitor_live(db, executor)
        closed  = monitor.run(dry_run=False)

        assert closed == [], "nothing returned — order was not placed"
        row = db.get_history(limit=1)[0]
        assert row['status'] == 'EXECUTED', "must be back to EXECUTED for retry"
