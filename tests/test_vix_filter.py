"""
Tests for the VIX filter feature in OptionScanner.

Covers:
  - Attribute initialisation (current_vix, vix_filter_enabled, vix_ic_pause_threshold)
  - VIX filter blocking IC scans in scan_ticker() when VIX >= threshold
  - VIX filter NOT blocking IC when VIX < threshold or VIX is None
  - Exact threshold boundary: VIX == threshold → blocked (≥)
  - Disabled flag short-circuits the filter regardless of current_vix value
  - agent.py VIX injection: scanner.current_vix is set from the yfinance history
"""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner


# ── Config helpers ────────────────────────────────────────────────────────────

def _cfg(vix_enabled=True, threshold=20.0, ic_enabled=True):
    return {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {
            'min_probability_of_expiry': 0.8,
            'vix_filter': {
                'enabled':            vix_enabled,
                'ic_pause_threshold': threshold,
            },
        },
        'strategies': {
            'covered_put':       {'enabled': False},
            'put_credit_spread': {'enabled': False},
            'call_credit_spread': {'enabled': False},
            'iron_condor': {
                'enabled':          ic_enabled,
                'min_net_credit':   0.10,
                'max_delta_short_leg': 0.20,
                'put_strike_width': 5,
                'call_strike_width': 5,
                'min_prob_profit':  0.70,
            },
        },
    }


def _make_yf_ticker_mock(market_cap=5e9, price=150.0, expiry_in_days=7):
    """Return a mock yf.Ticker that passes market-cap / price / expiry checks."""
    expiry_str = (datetime.now() + timedelta(days=expiry_in_days)).strftime('%Y-%m-%d')
    mock_ticker = MagicMock()
    mock_ticker.info = {'marketCap': market_cap, 'currentPrice': price}
    mock_ticker.options = [expiry_str]
    mock_ticker.option_chain.return_value = MagicMock(
        calls=MagicMock(), puts=MagicMock()
    )
    # history() used for HV30 calculation — return empty so no override fires
    mock_ticker.history.return_value = MagicMock(empty=True)
    return mock_ticker


# ═══════════════════════════════════════════════════════════════════════════════
# Attribute initialisation
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIXFilterAttributes:

    def test_current_vix_initialises_to_none(self):
        sc = OptionScanner(_cfg())
        assert sc.current_vix is None

    def test_vix_filter_enabled_reads_from_config(self):
        sc = OptionScanner(_cfg(vix_enabled=True))
        assert sc.vix_filter_enabled is True

    def test_vix_filter_disabled_reads_from_config(self):
        sc = OptionScanner(_cfg(vix_enabled=False))
        assert sc.vix_filter_enabled is False

    def test_threshold_reads_from_config(self):
        sc = OptionScanner(_cfg(threshold=25.0))
        assert sc.vix_ic_pause_threshold == 25.0

    def test_default_threshold_is_20_when_filter_absent(self):
        cfg = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {},   # no vix_filter key
            'strategies': {},
        }
        sc = OptionScanner(cfg)
        assert sc.vix_ic_pause_threshold == 20.0

    def test_vix_filter_disabled_by_default_when_key_absent(self):
        cfg = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {},
            'strategies': {},
        }
        sc = OptionScanner(cfg)
        assert sc.vix_filter_enabled is False

    def test_current_vix_can_be_set(self):
        sc = OptionScanner(_cfg())
        sc.current_vix = 18.5
        assert sc.current_vix == 18.5


# ═══════════════════════════════════════════════════════════════════════════════
# VIX filter in scan_ticker()
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIXFilterInScanTicker:
    """
    We patch 'src.scanner.yf' (the name bound in scanner.py at import time) so
    that scan_ticker() sees our mock ticker, and mock _scan_iron_condor so we
    can assert whether it was called.
    """

    @patch('src.scanner.yf')
    def test_ic_blocked_when_vix_above_threshold(self, mock_yf):
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=True, threshold=20.0, ic_enabled=True))
        sc.current_vix = 22.0   # above threshold → IC must be skipped

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_not_called()

    @patch('src.scanner.yf')
    def test_ic_not_blocked_when_vix_below_threshold(self, mock_yf):
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=True, threshold=20.0, ic_enabled=True))
        sc.current_vix = 17.0   # below threshold → IC must run

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_called_once()

    @patch('src.scanner.yf')
    def test_ic_blocked_exactly_at_threshold(self, mock_yf):
        """VIX == threshold is treated as blocked (>= is inclusive)."""
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=True, threshold=20.0, ic_enabled=True))
        sc.current_vix = 20.0   # exactly at threshold → blocked

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_not_called()

    @patch('src.scanner.yf')
    def test_ic_not_blocked_when_current_vix_is_none(self, mock_yf):
        """If no VIX has been injected (None), the filter must not block IC."""
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=True, threshold=20.0, ic_enabled=True))
        sc.current_vix = None   # not yet fetched

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_called_once()

    @patch('src.scanner.yf')
    def test_ic_not_blocked_when_filter_disabled(self, mock_yf):
        """When vix_filter.enabled=False, IC must run even if current_vix is high."""
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=False, threshold=20.0, ic_enabled=True))
        sc.current_vix = 50.0   # extreme value — filter is off, should be ignored

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_called_once()

    @patch('src.scanner.yf')
    def test_ic_not_called_when_ic_strategy_disabled(self, mock_yf):
        """If IC strategy itself is disabled, _scan_iron_condor is never called (VIX aside)."""
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        sc = OptionScanner(_cfg(vix_enabled=True, threshold=20.0, ic_enabled=False))
        sc.current_vix = 10.0   # low VIX — filter wouldn't block, but IC is disabled

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_not_called()

    @patch('src.scanner.yf')
    def test_other_strategies_unaffected_by_vix_filter(self, mock_yf):
        """PCS and CCS must still be called even when IC is VIX-blocked."""
        mock_yf.Ticker.return_value = _make_yf_ticker_mock()
        cfg = _cfg(vix_enabled=True, threshold=20.0, ic_enabled=True)
        cfg['strategies']['put_credit_spread'] = {
            'enabled': True,
            'min_net_credit': 0.10,
            'strike_width': 5,
            'min_prob_profit': 0.60,
            'max_delta_short_leg': 0.5,
        }
        sc = OptionScanner(cfg)
        sc.current_vix = 25.0   # VIX above threshold → IC blocked

        sc._scan_iron_condor = MagicMock(return_value=[])
        sc._scan_spreads      = MagicMock(return_value=[])
        sc.scan_ticker('AAPL')

        sc._scan_iron_condor.assert_not_called()
        sc._scan_spreads.assert_called()   # PCS should still run


# ═══════════════════════════════════════════════════════════════════════════════
# agent.py VIX injection logic (unit-tested in isolation)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentVIXInjection:
    """
    Simulate the VIX injection block in agent.py without running the full agent.
    We extract the relevant logic and verify the expected scanner state changes.
    """

    def _run_vix_injection(self, scanner, vix_history_df, vix_enabled=True, threshold=20.0):
        """
        Mimics the VIX injection block from agent.py.
        Returns the log messages emitted (as a list of (level, msg) tuples).
        """
        vix_cfg = {'enabled': vix_enabled, 'ic_pause_threshold': threshold}
        log_msgs = []

        class FakeLog:
            def info(self, msg):    log_msgs.append(('info', msg))
            def warning(self, msg): log_msgs.append(('warning', msg))

        log = FakeLog()

        if vix_cfg.get('enabled', False):
            try:
                if not vix_history_df.empty:
                    scanner.current_vix = float(vix_history_df['Close'].iloc[-1])
                    ic_thresh = float(vix_cfg.get('ic_pause_threshold', 20.0))
                    if scanner.current_vix >= ic_thresh:
                        log.warning(f"VIX filter active: VIX={scanner.current_vix:.1f} >= "
                                    f"{ic_thresh:.0f} — Iron Condor scans PAUSED this session.")
                    else:
                        log.info(f"VIX={scanner.current_vix:.1f} (below {ic_thresh:.0f} threshold) — "
                                 f"Iron Condor scans enabled.")
            except Exception as exc:
                log.warning(f"Could not fetch VIX level: {exc} — VIX filter will not apply.")

        return log_msgs

    def _mock_vix_df(self, close_value):
        """Build a minimal pandas-like mock DataFrame with a Close column."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(
            iloc=MagicMock(__getitem__=MagicMock(return_value=close_value))
        ))
        return mock_df

    def test_scanner_current_vix_set_from_history(self):
        sc = OptionScanner(_cfg())
        df = self._mock_vix_df(15.5)
        self._run_vix_injection(sc, df, vix_enabled=True, threshold=20.0)
        assert sc.current_vix == 15.5

    def test_current_vix_set_above_threshold_logs_warning(self):
        sc = OptionScanner(_cfg())
        df = self._mock_vix_df(23.0)
        msgs = self._run_vix_injection(sc, df, vix_enabled=True, threshold=20.0)
        levels = [m[0] for m in msgs]
        assert 'warning' in levels

    def test_current_vix_below_threshold_logs_info(self):
        sc = OptionScanner(_cfg())
        df = self._mock_vix_df(17.0)
        msgs = self._run_vix_injection(sc, df, vix_enabled=True, threshold=20.0)
        levels = [m[0] for m in msgs]
        assert 'info' in levels

    def test_empty_history_leaves_current_vix_none(self):
        sc = OptionScanner(_cfg())
        mock_df = MagicMock()
        mock_df.empty = True
        self._run_vix_injection(sc, mock_df, vix_enabled=True, threshold=20.0)
        assert sc.current_vix is None

    def test_injection_skipped_when_filter_disabled(self):
        sc = OptionScanner(_cfg(vix_enabled=False))
        df = self._mock_vix_df(30.0)
        self._run_vix_injection(sc, df, vix_enabled=False, threshold=20.0)
        # VIX injection block guarded by enabled check → current_vix stays None
        assert sc.current_vix is None

    def test_exception_in_fetch_leaves_current_vix_none_and_logs_warning(self):
        sc = OptionScanner(_cfg())
        mock_df = MagicMock()
        mock_df.empty = False
        # Make iloc[-1] raise
        mock_df.__getitem__ = MagicMock(side_effect=Exception("network error"))
        msgs = self._run_vix_injection(sc, mock_df, vix_enabled=True, threshold=20.0)
        assert sc.current_vix is None
        levels = [m[0] for m in msgs]
        assert 'warning' in levels

    def test_vix_at_threshold_triggers_warning(self):
        sc = OptionScanner(_cfg())
        df = self._mock_vix_df(20.0)
        msgs = self._run_vix_injection(sc, df, vix_enabled=True, threshold=20.0)
        # VIX == threshold → should warn (paused)
        text = ' '.join(m[1] for m in msgs)
        assert 'PAUSED' in text or 'warning' in [m[0] for m in msgs]
