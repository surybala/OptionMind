"""
test_position_monitor.py
========================
Unit tests for HFT-specific code paths in src/position_monitor.py.

All external dependencies (yfinance, Alpaca SDK, database, executor) are
mocked so no network calls or real credentials are needed.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Stub alpaca-py sub-modules before importing position_monitor
import sys as _sys
_alpaca_stub = MagicMock()
for _mod in [
    'alpaca', 'alpaca.data', 'alpaca.data.historical',
    'alpaca.data.requests', 'alpaca.data.timeframe',
]:
    _sys.modules.setdefault(_mod, _alpaca_stub)


from src.position_monitor import PositionMonitor


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_monitor(config=None):
    """Return a PositionMonitor with mocked db, executor and minimal config."""
    db       = MagicMock()
    executor = MagicMock()
    cfg = {
        'risk_parameters': {
            'stop_loss_multiplier': 2.0,
            'gamma_risk': {'enabled': False},
        },
        'hft': {
            'poll_interval_seconds': 15,
            'max_retries': 1,
            'retry_base_delay_seconds': 0.0,
        },
    }
    if config:
        cfg.update(config)
    return PositionMonitor(db, executor, cfg)


def _make_csp_pos(symbol='AAPL', expiry='2025-06-20', strike=180.0,
                  premium=1.50, spread_width=5.0):
    """Return a minimal CSP position dict."""
    return {
        'id':          1,
        'symbol':      symbol,
        'type':        'CSP',
        'expiry':      expiry,
        'strike':      strike,
        'spread_width': spread_width,
        'premium':     premium,
        'quantity':    1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PositionMonitor._build_osi_symbol — OSI string construction
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildOsiSymbol(unittest.TestCase):

    def test_put_symbol_format(self):
        result = PositionMonitor._build_osi_symbol('AAPL', '2024-01-19', 200.0, 'put')
        self.assertEqual(result, 'AAPL240119P00200000')

    def test_call_symbol_format(self):
        result = PositionMonitor._build_osi_symbol('SPY', '2025-03-21', 450.0, 'call')
        self.assertEqual(result, 'SPY250321C00450000')

    def test_fractional_strike(self):
        result = PositionMonitor._build_osi_symbol('AAPL', '2024-01-19', 185.5, 'put')
        self.assertEqual(result, 'AAPL240119P00185500')

    def test_multi_char_root(self):
        result = PositionMonitor._build_osi_symbol('TSLA', '2025-01-17', 500.0, 'call')
        self.assertEqual(result, 'TSLA250117C00500000')

    def test_round_trip_parse(self):
        """OSI we build should parse back to original values via parse_osi."""
        from src.osi import parse_osi
        osi = PositionMonitor._build_osi_symbol('MSFT', '2025-06-20', 350.0, 'put')
        parsed = parse_osi(osi)
        self.assertEqual(parsed.option_type, 'put')
        self.assertAlmostEqual(parsed.strike, 350.0)
        self.assertEqual(parsed.expiration.isoformat(), '2025-06-20')


# ══════════════════════════════════════════════════════════════════════════════
# PositionMonitor.run_hft — HFT monitoring loop
# ══════════════════════════════════════════════════════════════════════════════

class TestRunHft(unittest.TestCase):

    def test_returns_empty_list_when_no_open_positions(self):
        monitor = _make_monitor()
        monitor.db.get_open_positions.return_value = []
        result = monitor.run_hft(dry_run=True)
        self.assertEqual(result, [])

    def test_returns_closed_positions_list(self):
        monitor = _make_monitor()
        pos = _make_csp_pos()
        monitor.db.get_open_positions.return_value = [pos]
        closed_pos = {'id': 1, 'symbol': 'AAPL', 'reason_tag': 'STOP_LOSS'}
        monitor._check_position = MagicMock(return_value=closed_pos)
        result = monitor.run_hft(dry_run=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')

    def test_runtime_error_logged_but_does_not_crash(self):
        """A RuntimeError from _check_position should be caught and logged,
        not propagate to the caller."""
        monitor = _make_monitor()
        monitor.db.get_open_positions.return_value = [_make_csp_pos()]
        monitor._check_position = MagicMock(
            side_effect=RuntimeError('Alpaca retries exhausted')
        )
        # Should not raise
        result = monitor.run_hft(dry_run=True)
        self.assertEqual(result, [])

    def test_multiple_positions_all_checked(self):
        monitor = _make_monitor()
        pos1 = _make_csp_pos(symbol='AAPL')
        pos2 = _make_csp_pos(symbol='MSFT')
        monitor.db.get_open_positions.return_value = [pos1, pos2]
        monitor._check_position = MagicMock(return_value=None)
        monitor.run_hft(dry_run=True)
        self.assertEqual(monitor._check_position.call_count, 2)

    def test_failed_position_does_not_block_subsequent_positions(self):
        """If position #1 raises RuntimeError, position #2 is still checked."""
        monitor = _make_monitor()
        pos1 = _make_csp_pos(symbol='AAPL')
        pos2 = _make_csp_pos(symbol='MSFT')
        monitor.db.get_open_positions.return_value = [pos1, pos2]
        closed_pos2 = {'id': 2, 'symbol': 'MSFT', 'reason_tag': 'STOP_LOSS'}
        monitor._check_position = MagicMock(side_effect=[
            RuntimeError('AAPL Alpaca failure'),
            closed_pos2,
        ])
        result = monitor.run_hft(dry_run=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'MSFT')

    def test_none_return_from_check_not_included_in_closed(self):
        """If _check_position returns None (no trigger), position is not added."""
        monitor = _make_monitor()
        monitor.db.get_open_positions.return_value = [_make_csp_pos()]
        monitor._check_position = MagicMock(return_value=None)
        result = monitor.run_hft(dry_run=True)
        self.assertEqual(result, [])


# ══════════════════════════════════════════════════════════════════════════════
# PositionMonitor._fetch_chain_data_hft — Alpaca-only data fetch
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchChainDataHft(unittest.TestCase):

    def _make_monitor_with_alpaca(self, alpaca_client=None):
        monitor = _make_monitor()
        if alpaca_client is None:
            alpaca_client = MagicMock()
        monitor._alpaca_hft_client = MagicMock(return_value=alpaca_client)
        return monitor, alpaca_client

    @patch('src.alpaca_data.make_alpaca_data_client')
    def test_raises_when_no_alpaca_credentials(self, mock_make_client):
        """_fetch_chain_data_hft must raise RuntimeError if Alpaca client is None."""
        mock_make_client.return_value = None
        monitor = _make_monitor()
        pos = _make_csp_pos()
        with self.assertRaises(RuntimeError):
            monitor._fetch_chain_data_hft(pos)

    @patch('src.alpaca_data.make_alpaca_data_client')
    def test_raises_when_snapshot_fetch_fails_after_retries(self, mock_make_client):
        """Exhausted Alpaca retries should propagate as RuntimeError."""
        alpaca = MagicMock()
        alpaca.get_option_snapshots.side_effect = RuntimeError('retries exhausted')
        alpaca.get_spot_price_strict.return_value = 185.0
        mock_make_client.return_value = alpaca
        monitor = _make_monitor()
        pos = _make_csp_pos()
        with self.assertRaises(RuntimeError):
            monitor._fetch_chain_data_hft(pos)

    def test_no_yfinance_import_in_hft_path(self):
        """_fetch_chain_data_hft must never import or touch yfinance."""
        osi = 'AAPL250620P00180000'
        row = {
            'strike': 180.0, 'bid': 0.50, 'ask': 0.80, 'lastPrice': 0.65,
            'impliedVolatility': 0.25, 'openInterest': -1, 'volume': 0,
            'delta': -0.08, 'gamma': 0.02, 'theta': -0.04, 'vega': 0.10, 'rho': -0.01,
        }
        alpaca = MagicMock()
        alpaca.get_option_snapshots.return_value = {osi: row}
        alpaca.get_spot_price_strict.return_value = 190.0

        monitor = _make_monitor()
        pos = _make_csp_pos(expiry='2025-06-20', strike=180.0)

        # Track if yfinance is ever accessed during the call
        import builtins
        original_import = builtins.__import__

        yf_imported = []

        def patched_import(name, *args, **kwargs):
            if name == 'yfinance':
                yf_imported.append(name)
            return original_import(name, *args, **kwargs)

        with patch('src.alpaca_data.make_alpaca_data_client', return_value=alpaca):
            with patch.object(builtins, '__import__', patched_import):
                try:
                    monitor._fetch_chain_data_hft(pos)
                except Exception:
                    pass  # we only care about yfinance import, not full success

        self.assertEqual(yf_imported, [],
                         "yfinance must NOT be imported in HFT _fetch_chain_data_hft path")


class TestEnrichPnlContractScaling(unittest.TestCase):
    """_enrich_pnl must scale pnl_dollars by the contracts field."""

    def _enrich(self, entry, mark, contracts=1):
        pos = {'contracts': contracts}
        PositionMonitor._enrich_pnl(pos, entry, mark, sl_multiplier=2.0)
        return pos

    def test_single_contract_pnl(self):
        pos = self._enrich(entry=1.00, mark=0.50, contracts=1)
        self.assertAlmostEqual(pos['pnl_dollars'], 50.0)

    def test_multi_contract_pnl_scales(self):
        pos = self._enrich(entry=1.00, mark=0.50, contracts=10)
        self.assertAlmostEqual(pos['pnl_dollars'], 500.0)

    def test_twenty_contracts_loss(self):
        pos = self._enrich(entry=0.50, mark=1.20, contracts=20)
        # pnl_per_share = 0.50 - 1.20 = -0.70; total = -0.70 × 100 × 20 = -1400
        self.assertAlmostEqual(pos['pnl_dollars'], -1400.0)

    def test_missing_contracts_defaults_to_one(self):
        pos = {}
        PositionMonitor._enrich_pnl(pos, 1.00, 0.60, sl_multiplier=2.0)
        self.assertAlmostEqual(pos['pnl_dollars'], 40.0)

    def test_contracts_none_treated_as_one(self):
        pos = {'contracts': None}
        PositionMonitor._enrich_pnl(pos, 1.00, 0.60, sl_multiplier=2.0)
        self.assertAlmostEqual(pos['pnl_dollars'], 40.0)

    def test_pnl_per_share_unaffected_by_contracts(self):
        pos = self._enrich(entry=1.00, mark=0.40, contracts=5)
        self.assertAlmostEqual(pos['pnl_per_share'], 0.60, places=4)

    def test_stop_proximity_unaffected_by_contracts(self):
        pos10 = self._enrich(entry=1.00, mark=1.50, contracts=10)
        pos1  = self._enrich(entry=1.00, mark=1.50, contracts=1)
        self.assertEqual(pos10['stop_proximity_pct'], pos1['stop_proximity_pct'])


if __name__ == '__main__':
    unittest.main()
