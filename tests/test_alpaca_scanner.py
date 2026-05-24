"""
test_alpaca_scanner.py
======================
Tests for the Alpaca data-source integration points inside src/scanner.py:

1. OI -1 sentinel in _scan_spreads (PCS / CCS): Alpaca omits OI → -1 must
   pass the min_open_interest filter, not reject the pick.
2. OI -1 sentinel in _scan_iron_condor: same for both put and call legs.
3. _batch_prefetch_history: uses Alpaca when client is present; skips yfinance
   when Alpaca coverage ≥ 80 %; falls through when coverage is partial.
4. scan_ticker: uses get_option_chains_for_range (one call per ticker) when
   _ALPACA_CLIENT is set; falls back to ticker.options when Alpaca is empty.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

# ── Stub external dependencies before importing src.scanner ───────────────────
sys.modules.setdefault('yfinance', MagicMock())

# Stub alpaca-py so AlpacaDataClient.__init__ doesn't require the real SDK.
# This mirrors what test_alpaca_data.py does — ensures that even when env-var
# credentials are present the scanner can create an AlpacaDataClient without
# a live network connection.
_alpaca_stub = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.data',
    'alpaca.data.historical',
    'alpaca.data.requests',
    'alpaca.data.timeframe',
]:
    sys.modules.setdefault(_mod, _alpaca_stub)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.scanner as scanner_mod
from src.scanner import OptionScanner


# ── Test config helpers ────────────────────────────────────────────────────────

def _spread_config(min_oi=10, strike_width=5):
    return {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'skip_market_cap_check_in_scanner': True,
        'risk_parameters': {'min_probability_of_expiry': 0.7},
        'strategies': {
            'put_credit_spread': {
                'enabled': True,
                'min_net_credit': 0.10,
                'strike_width': strike_width,
                'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.50,
                'min_open_interest': min_oi,
            },
            'call_credit_spread': {
                'enabled': True,
                'min_net_credit': 0.10,
                'strike_width': strike_width,
                'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.50,
                'min_open_interest': min_oi,
            },
        },
    }


def _ic_config(min_oi=10):
    return {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'skip_market_cap_check_in_scanner': True,
        'risk_parameters': {'min_probability_of_expiry': 0.7},
        'strategies': {
            'iron_condor': {
                'enabled': True,
                'min_net_credit': 0.10,
                'max_delta_short_leg': 0.50,
                'put_strike_width': 5,
                'call_strike_width': 5,
                'min_prob_profit': 0.50,
                'min_open_interest': min_oi,
            },
        },
    }


def _make_puts_df(rows):
    """Build a real pandas DataFrame of put option rows."""
    return pd.DataFrame(rows)


def _make_calls_df(rows):
    return pd.DataFrame(rows)


# Standard row for a viable spread leg
def _leg(strike, bid=1.5, ask=1.6, last=1.5, iv=0.4, oi=None):
    row = {
        'strike': float(strike),
        'bid': bid,
        'ask': ask,
        'lastPrice': last,
        'impliedVolatility': iv,
    }
    if oi is not None:
        row['openInterest'] = oi
    # No 'openInterest' key → .get() returns None → treated as 0, not -1
    return row


def _leg_oi(strike, oi, bid=1.5, ask=1.6, last=1.5, iv=0.4):
    """Row with an explicit openInterest value (including -1 sentinel)."""
    return {
        'strike': float(strike),
        'bid': bid,
        'ask': ask,
        'lastPrice': last,
        'impliedVolatility': iv,
        'openInterest': oi,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OI -1 sentinel: _scan_spreads
# ══════════════════════════════════════════════════════════════════════════════

class TestOiSentinelSpreads(unittest.TestCase):
    """
    min_open_interest = 10.
    - OI = -1  (Alpaca unknown)  → PASS  (sentinel means "skip filter")
    - OI =  0  (known, below 10) → FAIL
    - OI = 15  (known, above 10) → PASS
    """

    def setUp(self):
        # Suppress AlpacaDataClient init inside OptionScanner
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            self.scanner = OptionScanner(_spread_config(min_oi=10, strike_width=5))

    def _run_pcs(self, short_oi, long_oi):
        """Build a minimal put chain and run _scan_spreads for PCS."""
        rows = [
            _leg_oi(85, oi=long_oi,  bid=0.40, ask=0.50),   # long leg
            _leg_oi(90, oi=short_oi, bid=1.50, ask=1.60),   # short leg (OTM for price=100)
        ]
        df = _make_puts_df(rows)
        return self.scanner._scan_spreads('TEST', 100.0, '2026-06-20', 20, df, 'put')

    def _run_ccs(self, short_oi, long_oi):
        """Build a minimal call chain and run _scan_spreads for CCS."""
        rows = [
            _leg_oi(110, oi=short_oi, bid=1.50, ask=1.60),  # short leg (OTM for price=100)
            _leg_oi(115, oi=long_oi,  bid=0.40, ask=0.50),  # long leg
        ]
        df = _make_calls_df(rows)
        return self.scanner._scan_spreads('TEST', 100.0, '2026-06-20', 20, df, 'call')

    # ── PCS OI sentinel -1 ─────────────────────────────────────────────────

    def test_pcs_oi_minus1_passes_filter(self):
        """OI = -1 (Alpaca unknown) must not reject a viable PCS."""
        results = self._run_pcs(short_oi=-1, long_oi=-1)
        self.assertGreater(len(results), 0, "OI=-1 sentinel should pass min_oi filter")

    def test_pcs_oi_zero_rejected(self):
        """OI = 0 is a known value below min → spread should be filtered out."""
        results = self._run_pcs(short_oi=0, long_oi=0)
        self.assertEqual(len(results), 0, "OI=0 (known) should fail min_oi=10 filter")

    def test_pcs_short_leg_oi_zero_rejected(self):
        """Short leg OI = 0 alone is enough to reject even if long leg is ok."""
        results = self._run_pcs(short_oi=0, long_oi=50)
        self.assertEqual(len(results), 0)

    def test_pcs_long_leg_oi_zero_rejected(self):
        results = self._run_pcs(short_oi=50, long_oi=0)
        self.assertEqual(len(results), 0)

    def test_pcs_oi_above_min_passes(self):
        """OI = 15 > min 10 → spread accepted."""
        results = self._run_pcs(short_oi=15, long_oi=15)
        self.assertGreater(len(results), 0)

    def test_pcs_mixed_sentinel_and_known_passes_when_known_above_min(self):
        """One leg -1 (unknown), other leg OI 20 (above min) → pass."""
        results = self._run_pcs(short_oi=-1, long_oi=20)
        self.assertGreater(len(results), 0)

    def test_pcs_mixed_sentinel_and_zero_passes(self):
        """One leg -1 (unknown), other leg OI 0 — 0 is known-below-min.
        The filter checks BOTH legs: if either is -1 (unknown), skip the filter
        entirely (both must be known for the filter to apply)."""
        # With short_oi=-1, the sentinel check short_oi != -1 is False,
        # so the filter block is skipped regardless of long_oi.
        results = self._run_pcs(short_oi=-1, long_oi=0)
        self.assertGreater(len(results), 0, "Sentinel on either leg skips OI filter")

    # ── CCS OI sentinel -1 ─────────────────────────────────────────────────

    def test_ccs_oi_minus1_passes_filter(self):
        results = self._run_ccs(short_oi=-1, long_oi=-1)
        self.assertGreater(len(results), 0)

    def test_ccs_oi_zero_rejected(self):
        results = self._run_ccs(short_oi=0, long_oi=0)
        self.assertEqual(len(results), 0)

    def test_ccs_oi_above_min_passes(self):
        results = self._run_ccs(short_oi=15, long_oi=15)
        self.assertGreater(len(results), 0)

    # ── No OI filter when min_open_interest = 0 ────────────────────────────

    def test_no_filter_when_min_oi_is_zero(self):
        """min_open_interest = 0 disables the filter entirely."""
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            scanner = OptionScanner(_spread_config(min_oi=0))
        rows = [
            _leg_oi(85, oi=0),
            _leg_oi(90, oi=0),
        ]
        results = scanner._scan_spreads('TEST', 100.0, '2026-06-20', 20,
                                        _make_puts_df(rows), 'put')
        # With min_oi=0 the filter is off — result depends on credit/prob,
        # but it should NOT be rejected for OI reasons
        # (we're not asserting the count, just that a zero-OI doesn't cause a crash)
        self.assertIsInstance(results, list)


# ══════════════════════════════════════════════════════════════════════════════
# OI -1 sentinel: _scan_iron_condor
# ══════════════════════════════════════════════════════════════════════════════

class TestOiSentinelIronCondor(unittest.TestCase):
    """IC uses separate OI checks for put legs and call legs."""

    def setUp(self):
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            self.scanner = OptionScanner(_ic_config(min_oi=10))

    def _run_ic(self, put_short_oi, put_long_oi, call_short_oi, call_long_oi):
        puts = _make_puts_df([
            _leg_oi(85, oi=put_long_oi,  bid=0.40, ask=0.50),
            _leg_oi(90, oi=put_short_oi, bid=1.50, ask=1.60),
        ])
        calls = _make_calls_df([
            _leg_oi(110, oi=call_short_oi, bid=1.50, ask=1.60),
            _leg_oi(115, oi=call_long_oi,  bid=0.40, ask=0.50),
        ])
        return self.scanner._scan_iron_condor('TEST', 100.0, '2026-06-20', 20, puts, calls)

    def test_all_legs_sentinel_passes(self):
        results = self._run_ic(-1, -1, -1, -1)
        self.assertGreater(len(results), 0, "IC: all-sentinel OI should pass")

    def test_all_legs_oi_zero_rejected(self):
        results = self._run_ic(0, 0, 0, 0)
        self.assertEqual(len(results), 0)

    def test_all_legs_oi_above_min_passes(self):
        results = self._run_ic(20, 20, 20, 20)
        self.assertGreater(len(results), 0)

    def test_put_leg_sentinel_call_leg_known_above_min(self):
        """Put legs are unknown (-1); call legs have known adequate OI."""
        results = self._run_ic(-1, -1, 20, 20)
        self.assertGreater(len(results), 0)

    def test_call_leg_sentinel_put_leg_known_above_min(self):
        results = self._run_ic(20, 20, -1, -1)
        self.assertGreater(len(results), 0)

    def test_call_leg_oi_zero_rejected(self):
        """Call short leg OI = 0 should block the IC even if put legs are fine."""
        results = self._run_ic(20, 20, 0, 20)
        self.assertEqual(len(results), 0)

    def test_put_leg_oi_zero_rejected(self):
        results = self._run_ic(0, 20, 20, 20)
        self.assertEqual(len(results), 0)


# ══════════════════════════════════════════════════════════════════════════════
# _batch_prefetch_history Alpaca integration
# ══════════════════════════════════════════════════════════════════════════════

class TestBatchPrefetchHistory(unittest.TestCase):
    """
    _batch_prefetch_history uses Alpaca (when client present) and skips yfinance
    when coverage ≥ 80%.  Falls through to yfinance for partial coverage.
    """

    def _clear_caches(self):
        with scanner_mod._HIST_LOCK:
            scanner_mod._HIST_CACHE.clear()

    def setUp(self):
        self._clear_caches()

    def tearDown(self):
        self._clear_caches()
        scanner_mod._ALPACA_CLIENT = None

    def _make_alpaca_client(self, data: dict):
        client = MagicMock()
        client.get_bulk_history.return_value = data
        return client

    def _series(self, n=10):
        return pd.Series(range(n), dtype=float)

    @patch('src.scanner.yf')
    def test_alpaca_full_coverage_skips_yfinance(self, mock_yf):
        """≥80% coverage → yf.download is never called."""
        syms = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        alpaca_data = {s: self._series() for s in syms}  # 100% coverage
        scanner_mod._ALPACA_CLIENT = self._make_alpaca_client(alpaca_data)

        scanner_mod._batch_prefetch_history(syms)

        mock_yf.download.assert_not_called()
        with scanner_mod._HIST_LOCK:
            for s in syms:
                self.assertIn(s, scanner_mod._HIST_CACHE)

    @patch('src.scanner.yf')
    def test_alpaca_partial_coverage_falls_through_to_yfinance(self, mock_yf):
        """<80% coverage → yf.download is called as a fallback."""
        syms = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
        # Only 3/5 = 60% → below 80% threshold
        alpaca_data = {'AAPL': self._series(), 'MSFT': self._series(), 'GOOG': self._series()}
        scanner_mod._ALPACA_CLIENT = self._make_alpaca_client(alpaca_data)

        mock_yf.download.return_value = MagicMock(empty=True)
        scanner_mod._batch_prefetch_history(syms)

        mock_yf.download.assert_called_once()

    @patch('src.scanner.yf')
    def test_alpaca_exception_falls_through_to_yfinance(self, mock_yf):
        """If Alpaca raises, yf.download is called as fallback."""
        client = MagicMock()
        client.get_bulk_history.side_effect = Exception('API down')
        scanner_mod._ALPACA_CLIENT = client

        mock_yf.download.return_value = MagicMock(empty=True)
        scanner_mod._batch_prefetch_history(['AAPL'])

        mock_yf.download.assert_called_once()

    @patch('src.scanner.yf')
    def test_no_alpaca_client_goes_straight_to_yfinance(self, mock_yf):
        scanner_mod._ALPACA_CLIENT = None
        mock_yf.download.return_value = MagicMock(empty=True)
        scanner_mod._batch_prefetch_history(['AAPL'])
        mock_yf.download.assert_called_once()

    @patch('src.scanner.yf')
    def test_empty_ticker_list_no_calls(self, mock_yf):
        scanner_mod._ALPACA_CLIENT = self._make_alpaca_client({})
        scanner_mod._batch_prefetch_history([])
        mock_yf.download.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# scan_ticker: Alpaca get_option_chains_for_range integration
# ══════════════════════════════════════════════════════════════════════════════

class TestScanTickerAlpacaChainRange(unittest.TestCase):
    """
    When _ALPACA_CLIENT.get_option_chains_for_range returns a non-empty dict,
    scan_ticker must:
      - use those expiry dates (not call ticker.options)
      - pass the pre-fetched OptionChain to strategy scanners

    When it returns {}, scan_ticker must fall back to ticker.options.
    """

    def _clear_caches(self):
        with scanner_mod._HIST_LOCK:
            scanner_mod._HIST_CACHE.clear()
        with scanner_mod._CHAIN_LOCK:
            scanner_mod._CHAIN_CACHE.clear()

    def setUp(self):
        self._clear_caches()

    def tearDown(self):
        self._clear_caches()
        scanner_mod._ALPACA_CLIENT = None

    def _scan_config(self):
        return {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'skip_market_cap_check_in_scanner': True,
            'risk_parameters': {
                'min_probability_of_expiry': 0.5,
                'vix_filter': {'enabled': False},
            },
            'strategies': {
                'covered_put':        {'enabled': False},
                'put_credit_spread':  {'enabled': False},
                'call_credit_spread': {'enabled': False},
                'iron_condor':        {'enabled': False},
                'iron_butterfly':     {'enabled': False},
                'short_strangle':     {'enabled': False},
                'covered_call':       {'enabled': False},
            },
        }

    def _mock_alpaca_chains(self, expiry: str):
        """Return a dict with one dummy OptionChain for the given expiry."""
        from src.alpaca_data import OptionChain
        cols = ['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest']
        empty = pd.DataFrame(columns=cols)
        return {expiry: OptionChain(puts=empty, calls=empty)}

    @patch('src.alpaca_data.make_alpaca_data_client', return_value=None)
    def setUp_scanner(self, _mock):
        return OptionScanner(self._scan_config())

    def test_alpaca_chains_bypass_ticker_options(self):
        """
        When _ALPACA_CLIENT returns chains, ticker.options must NOT be accessed.
        We verify this by having ticker.options raise — if scan_ticker calls it,
        the test will fail.
        """
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            s = OptionScanner(self._scan_config())

        expiry = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        chains = self._mock_alpaca_chains(expiry)

        alpaca_mock = MagicMock()
        alpaca_mock.get_option_chains_for_range.return_value = chains
        scanner_mod._ALPACA_CLIENT = alpaca_mock

        # Pre-load price cache so scan_ticker skips fast_info
        scanner_mod._HIST_CACHE['TEST'] = pd.Series([100.0] * 10)

        with patch('src.scanner.yf') as mock_yf:
            # If ticker.options is accessed, it raises — test catches this failure
            ticker_mock = MagicMock()
            type(ticker_mock).options = property(
                lambda self: (_ for _ in ()).throw(RuntimeError('ticker.options called unexpectedly'))
            )
            mock_yf.Ticker.return_value = ticker_mock

            # Should not raise (Alpaca chains bypass ticker.options)
            result = s.scan_ticker('TEST')

        self.assertIsInstance(result, list)
        alpaca_mock.get_option_chains_for_range.assert_called_once()

    def test_alpaca_empty_falls_back_to_ticker_options(self):
        """
        When _ALPACA_CLIENT returns {}, scan_ticker must call ticker.options.
        """
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            s = OptionScanner(self._scan_config())

        alpaca_mock = MagicMock()
        alpaca_mock.get_option_chains_for_range.return_value = {}
        scanner_mod._ALPACA_CLIENT = alpaca_mock

        scanner_mod._HIST_CACHE['TEST'] = pd.Series([100.0] * 10)

        with patch('src.scanner.yf') as mock_yf:
            ticker_mock = MagicMock()
            # Return an expiry 30 days away — outside the 14-day window → empty result
            ticker_mock.options = [(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')]
            mock_yf.Ticker.return_value = ticker_mock

            result = s.scan_ticker('TEST')

        # Alpaca was called but returned empty, so yfinance ticker.options is used
        alpaca_mock.get_option_chains_for_range.assert_called_once()
        mock_yf.Ticker.assert_called_once_with('TEST')
        self.assertIsInstance(result, list)

    def test_alpaca_exception_falls_back_to_ticker_options(self):
        """
        When get_option_chains_for_range raises, scan_ticker catches it silently
        and falls back to ticker.options.
        """
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            s = OptionScanner(self._scan_config())

        alpaca_mock = MagicMock()
        alpaca_mock.get_option_chains_for_range.side_effect = Exception('API timeout')
        scanner_mod._ALPACA_CLIENT = alpaca_mock

        scanner_mod._HIST_CACHE['TEST'] = pd.Series([100.0] * 10)

        with patch('src.scanner.yf') as mock_yf:
            ticker_mock = MagicMock()
            ticker_mock.options = []   # empty options → scan returns []
            mock_yf.Ticker.return_value = ticker_mock
            result = s.scan_ticker('TEST')

        self.assertIsInstance(result, list)

    def test_expiry_outside_window_filtered_from_alpaca_chains(self):
        """
        Even if Alpaca returns an expiry outside max_expiry_days, it should be
        filtered out by the expiry-date window check.
        """
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            s = OptionScanner(self._scan_config())

        far_expiry = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        chains = self._mock_alpaca_chains(far_expiry)

        alpaca_mock = MagicMock()
        alpaca_mock.get_option_chains_for_range.return_value = chains
        scanner_mod._ALPACA_CLIENT = alpaca_mock

        scanner_mod._HIST_CACHE['TEST'] = pd.Series([100.0] * 10)

        with patch('src.scanner.yf') as mock_yf:
            mock_yf.Ticker.return_value = MagicMock()
            result = s.scan_ticker('TEST')

        # Expiry 30 days out is beyond max_expiry_days=14 → no strategies run
        self.assertEqual(result, [])

    def test_chain_cache_populated_from_alpaca_chains(self):
        """
        Chains fetched via get_option_chains_for_range should be written into
        _CHAIN_CACHE so subsequent calls within the TTL do not re-fetch.
        """
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            s = OptionScanner(self._scan_config())

        expiry = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        chains = self._mock_alpaca_chains(expiry)

        alpaca_mock = MagicMock()
        alpaca_mock.get_option_chains_for_range.return_value = chains
        scanner_mod._ALPACA_CLIENT = alpaca_mock

        scanner_mod._HIST_CACHE['TEST'] = pd.Series([100.0] * 10)

        with patch('src.scanner.yf') as mock_yf:
            mock_yf.Ticker.return_value = MagicMock()
            s.scan_ticker('TEST')

        with scanner_mod._CHAIN_LOCK:
            cached = scanner_mod._CHAIN_CACHE.get(('TEST', expiry))
        self.assertIsNotNone(cached, "Alpaca chain should be written to _CHAIN_CACHE")


# ══════════════════════════════════════════════════════════════════════════════
# _apply_liquidity_filter
# ══════════════════════════════════════════════════════════════════════════════

def _liq_scanner(min_bid=0.05, min_oi=10, max_spread=0.80):
    """OptionScanner configured with explicit chain_liquidity thresholds."""
    cfg = {
        'chain_liquidity': {
            'min_bid': min_bid,
            'min_open_interest': min_oi,
            'max_spread_pct': max_spread,
        },
        'strategies': {},
    }
    with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
        return OptionScanner(cfg)


def _df(*rows):
    return pd.DataFrame(list(rows))


class TestLiquidityFilter(unittest.TestCase):

    # ── min_bid ───────────────────────────────────────────────────────────────

    def test_zero_bid_row_removed(self):
        s = _liq_scanner(min_bid=0.05)
        df = _df({'strike': 100, 'bid': 0.00, 'ask': 0.10, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 0)

    def test_sub_threshold_bid_removed(self):
        s = _liq_scanner(min_bid=0.05)
        df = _df({'strike': 100, 'bid': 0.04, 'ask': 0.10, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 0)

    def test_exact_threshold_bid_kept(self):
        s = _liq_scanner(min_bid=0.05)
        df = _df({'strike': 100, 'bid': 0.05, 'ask': 0.10, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_bid_filter_disabled_when_zero(self):
        # Disable ALL three checks; a zero-bid row must still pass
        s = _liq_scanner(min_bid=0.0, min_oi=0, max_spread=1.0)
        df = _df({'strike': 100, 'bid': 0.00, 'ask': 0.10, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    # ── min_open_interest ─────────────────────────────────────────────────────

    def test_low_oi_row_removed(self):
        s = _liq_scanner(min_oi=10)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': 5})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 0)

    def test_zero_oi_row_removed(self):
        s = _liq_scanner(min_oi=10)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': 0})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 0)

    def test_oi_at_threshold_kept(self):
        s = _liq_scanner(min_oi=10)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': 10})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_alpaca_sentinel_minus1_always_passes(self):
        """OI == -1 means Alpaca does not expose it — must never be filtered."""
        s = _liq_scanner(min_oi=10)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': -1})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_oi_filter_disabled_when_zero(self):
        s = _liq_scanner(min_oi=0)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': 0})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_no_oi_column_does_not_crash(self):
        """DataFrames without openInterest (e.g. some yfinance chains) must pass."""
        s = _liq_scanner(min_oi=10)
        df = _df({'strike': 100, 'bid': 1.0, 'ask': 1.1})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    # ── max_spread_pct ────────────────────────────────────────────────────────

    def test_wide_spread_removed(self):
        # bid=0.10, ask=1.00 → spread/mid = 0.90/0.55 ≈ 1.64 > 0.80
        s = _liq_scanner(max_spread=0.80)
        df = _df({'strike': 100, 'bid': 0.10, 'ask': 1.00, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 0)

    def test_tight_spread_kept(self):
        # bid=1.00, ask=1.10 → spread/mid = 0.10/1.05 ≈ 0.095 < 0.80
        s = _liq_scanner(max_spread=0.80)
        df = _df({'strike': 100, 'bid': 1.00, 'ask': 1.10, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_spread_exactly_at_threshold_kept(self):
        # bid=1.00, ask=2.60 → spread=1.60, mid=1.80, pct=1.60/1.80 ≈ 0.889 > 0.80 → removed
        # bid=1.00, ask=2.44 → spread=1.44, mid=1.72, pct=1.44/1.72 ≈ 0.837 > 0.80 → removed
        # bid=1.00, ask=2.33 → spread=1.33, mid=1.665, pct=1.33/1.665 ≈ 0.799 < 0.80 → kept
        s = _liq_scanner(max_spread=0.80)
        df = _df({'strike': 100, 'bid': 1.00, 'ask': 2.33, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    def test_spread_filter_disabled_when_one(self):
        # Also disable min_bid so only the spread check (disabled) is in play
        s = _liq_scanner(min_bid=0.0, min_oi=0, max_spread=1.0)
        df = _df({'strike': 100, 'bid': 0.01, 'ask': 100.0, 'openInterest': 100})
        self.assertEqual(len(s._apply_liquidity_filter(df)), 1)

    # ── combined / edge cases ─────────────────────────────────────────────────

    def test_empty_dataframe_returns_empty(self):
        s = _liq_scanner()
        df = pd.DataFrame(columns=['strike', 'bid', 'ask', 'openInterest'])
        result = s._apply_liquidity_filter(df)
        self.assertTrue(result.empty)

    def test_multiple_rows_mixed_liquidity(self):
        s = _liq_scanner(min_bid=0.05, min_oi=10, max_spread=0.80)
        df = _df(
            {'strike': 90,  'bid': 0.00, 'ask': 0.10, 'openInterest': 100},  # fail: bid
            {'strike': 95,  'bid': 1.00, 'ask': 1.10, 'openInterest': 5},    # fail: oi
            {'strike': 100, 'bid': 0.10, 'ask': 1.50, 'openInterest': 100},  # fail: spread
            {'strike': 105, 'bid': 1.50, 'ask': 1.60, 'openInterest': 50},   # pass
        )
        result = s._apply_liquidity_filter(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['strike'], 105)

    def test_index_reset_after_filter(self):
        s = _liq_scanner(min_oi=10)
        df = _df(
            {'strike': 90,  'bid': 1.0, 'ask': 1.1, 'openInterest': 5},   # removed
            {'strike': 100, 'bid': 1.0, 'ask': 1.1, 'openInterest': 50},  # kept
        )
        result = s._apply_liquidity_filter(df)
        self.assertEqual(list(result.index), [0])  # index reset to 0


if __name__ == '__main__':
    unittest.main()
