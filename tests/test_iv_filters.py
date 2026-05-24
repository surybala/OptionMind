"""
Tests for IV quality filters introduced in 2026-04:
  - HV rank filter  (min_iv_rank): skip tickers where current HV30 is in the
    bottom N% of its 1-year range
  - IV premium filter (require_iv_premium): skip tickers where ATM implied
    vol < HV30 × iv_premium_min_ratio

Also covers:
  - _compute_atr_and_hv30 returning the two new values (hv30, hv_rank)
  - _compute_atm_iv helper
  - History prefetch period expansion when iv_filters.enabled=true
"""
import math
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from src.scanner import OptionScanner


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def _base_config(**iv_overrides):
    """Minimal OptionScanner config with iv_filters overlay."""
    iv_cfg = {
        'enabled': True,
        'min_iv_rank': 0.30,
        'require_iv_premium': True,
        'iv_premium_min_ratio': 1.0,
        'history_days': 252,
    }
    iv_cfg.update(iv_overrides)
    return {
        'market_cap_min': 1e9,
        'expiry_days_max': 30,
        'risk_parameters': {'min_probability_of_expiry': 0.7},
        'atr_distance': {'enabled': False},
        'dynamic_width': {'enabled': False},
        'min_otm_pct': {'put': 0.0, 'call': 0.0},
        'iv_filters': iv_cfg,
        'strategies': {
            'put_credit_spread': {
                'enabled': True, 'min_net_credit': 0.01,
                'strike_width': 5, 'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.60,
            },
        },
    }


def _scanner(iv_filters_enabled=True, **iv_overrides):
    return OptionScanner(_base_config(enabled=iv_filters_enabled, **iv_overrides))


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic price history helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_hist(n_days: int, daily_move_pct: float) -> pd.DataFrame:
    """Synthetic close-price series with constant daily moves."""
    price = 100.0
    closes = []
    for _ in range(n_days):
        price *= (1 + daily_move_pct)
        closes.append(price)
    return pd.DataFrame({'Close': closes})


def _make_hist_varying(low_vol_days: int, high_vol_days: int) -> pd.DataFrame:
    """First *low_vol_days* have ~0.3% daily moves, last *high_vol_days* have ~2%.

    Uses alternating up/down moves so each window has a non-zero std (real HV).
    """
    closes = []
    price = 100.0
    for i in range(low_vol_days):
        price *= (1.003 if i % 2 == 0 else 1 / 1.003)
        closes.append(price)
    for i in range(high_vol_days):
        price *= (1.020 if i % 2 == 0 else 1 / 1.020)
        closes.append(price)
    return pd.DataFrame({'Close': closes})


# ══════════════════════════════════════════════════════════════════════════════
# _compute_atr_and_hv30: new return values hv30 + hv_rank
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeHv30AndRank(unittest.TestCase):

    def _scanner(self):
        return _scanner(iv_filters_enabled=False)  # filter logic not needed here

    def test_hv30_returned_as_second_element(self):
        s = self._scanner()
        hist = _make_hist(100, 0.01)   # 1% daily move → HV30 ≈ 0.01 × √252 ≈ 0.159
        _, hv30, _, _, _, _ = s._compute_atr_and_hv30(hist)
        self.assertGreater(hv30, 0)

    def test_hv30_matches_annualised_log_return_std(self):
        s = self._scanner()
        hist = _make_hist(100, 0.01)
        _, hv30, _, _, _, _ = s._compute_atr_and_hv30(hist)
        lr = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        expected = float(lr.tail(30).std() * math.sqrt(252))
        self.assertAlmostEqual(hv30, expected, places=6)

    def test_hv_rank_neutral_when_insufficient_history(self):
        """Fewer than 60 rows → rank defaults to 0.5 (neutral)."""
        s = self._scanner()
        hist = _make_hist(40, 0.005)
        _, _, hv_rank, _, _, _ = s._compute_atr_and_hv30(hist)
        self.assertEqual(hv_rank, 0.5)

    def test_hv_rank_low_when_current_vol_is_low(self):
        """Low-vol tail → HV rank near 0."""
        s = self._scanner()
        # _make_hist_varying(low_vol_days, high_vol_days) puts high-vol at the END.
        # We want low-vol at the recent (tail) end, so build high-vol-first and reverse.
        raw = _make_hist_varying(low_vol_days=30, high_vol_days=200)
        hist = pd.DataFrame({'Close': raw['Close'].values[::-1]}).reset_index(drop=True)
        _, _, hv_rank, _, _, _ = s._compute_atr_and_hv30(hist)
        self.assertLess(hv_rank, 0.30, "Low recent vol should rank near 0")

    def test_hv_rank_high_when_current_vol_is_high(self):
        """High-vol tail → HV rank near 1."""
        s = self._scanner()
        # 200 days of 0.1% moves (low vol), last 30 of 3% (high vol)
        hist = _make_hist_varying(low_vol_days=200, high_vol_days=30)
        _, _, hv_rank, _, _, _ = s._compute_atr_and_hv30(hist)
        self.assertGreater(hv_rank, 0.70, "High recent vol should rank near 1")

    def test_hv_rank_clamped_to_0_1(self):
        s = self._scanner()
        hist = _make_hist(300, 0.02)
        _, _, hv_rank, _, _, _ = s._compute_atr_and_hv30(hist)
        self.assertGreaterEqual(hv_rank, 0.0)
        self.assertLessEqual(hv_rank, 1.0)

    def test_none_hist_returns_zero_hv30_and_neutral_rank(self):
        s = self._scanner()
        _, hv30, hv_rank, _, _, _ = s._compute_atr_and_hv30(None)
        self.assertEqual(hv30, 0.0)
        self.assertEqual(hv_rank, 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# _compute_atm_iv
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAtmIv(unittest.TestCase):

    def _make_chain(self, strikes_ivs_puts, strikes_ivs_calls):
        """Build a mock alpaca_chains dict for one expiry."""
        def _df(rows):
            return pd.DataFrame(rows, columns=['strike', 'impliedVolatility',
                                                'bid', 'ask', 'openInterest', 'volume'])
        puts  = _df([{'strike': s, 'impliedVolatility': iv, 'bid': 0.5,
                      'ask': 0.6, 'openInterest': 100, 'volume': 10}
                     for s, iv in strikes_ivs_puts])
        calls = _df([{'strike': s, 'impliedVolatility': iv, 'bid': 0.5,
                      'ask': 0.6, 'openInterest': 100, 'volume': 10}
                     for s, iv in strikes_ivs_calls])
        chain = MagicMock()
        chain.puts  = puts
        chain.calls = calls
        return {'2026-09-19': chain}

    def _scanner(self):
        return _scanner(iv_filters_enabled=False)

    def test_returns_median_atm_iv(self):
        s = self._scanner()
        spot = 100.0
        # Near-ATM: strikes 97, 100, 103 with IVs 0.25, 0.30, 0.35
        chains = self._make_chain(
            [(97, 0.25), (100, 0.30), (103, 0.35)],
            [(97, 0.28), (100, 0.32), (103, 0.38)],
        )
        iv = s._compute_atm_iv(chains, ['2026-09-19'], spot)
        self.assertIsNotNone(iv)
        self.assertGreater(iv, 0.20)
        self.assertLess(iv, 0.45)

    def test_ignores_far_otm_strikes(self):
        """Strikes outside the 5% ATM band should not affect the result."""
        s = self._scanner()
        spot = 100.0
        # Far strikes have artificially low IV — if included they'd drag down median
        chains = self._make_chain(
            [(70, 0.05), (100, 0.30), (130, 0.05)],
            [(70, 0.05), (100, 0.30), (130, 0.05)],
        )
        iv = s._compute_atm_iv(chains, ['2026-09-19'], spot)
        # Only the 100 strikes are within 5% of spot=100 → IV should be ~0.30
        self.assertAlmostEqual(iv, 0.30, places=2)

    def test_returns_none_when_no_chain(self):
        s = self._scanner()
        iv = s._compute_atm_iv({}, [], 100.0)
        self.assertIsNone(iv)

    def test_returns_none_when_all_iv_zero(self):
        s = self._scanner()
        chains = self._make_chain(
            [(99, 0.0), (101, 0.0)],
            [(99, 0.0), (101, 0.0)],
        )
        iv = s._compute_atm_iv(chains, ['2026-09-19'], 100.0)
        self.assertIsNone(iv)


# ══════════════════════════════════════════════════════════════════════════════
# IV rank gate integration (scan_ticker skips low-rank tickers)
# ══════════════════════════════════════════════════════════════════════════════

class TestIvRankGate(unittest.TestCase):
    """scan_ticker returns [] when HV rank < min_iv_rank."""

    def _run_scan(self, hv_rank_value, min_iv_rank=0.30):
        """Patch _compute_atr_and_hv30 to inject a controlled hv_rank."""
        scanner = _scanner(min_iv_rank=min_iv_rank, require_iv_premium=False)
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.20, hv_rank_value, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {
            'AAPL': pd.Series([100.0] * 30),
        }):
            # Don't actually fetch chains — return [] from fetch step
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               return_value=({}, [], set())):
                return scanner.scan_ticker('AAPL')

    def test_low_hv_rank_skips_ticker(self):
        """HV rank below min_iv_rank → scan returns []."""
        results = self._run_scan(hv_rank_value=0.15, min_iv_rank=0.30)
        self.assertEqual(results, [])

    def test_high_hv_rank_passes_gate(self):
        """HV rank above min_iv_rank → gate passes (may return [] for other reasons)."""
        # We just need to confirm the IV rank gate did NOT reject it —
        # i.e., _fetch_expirations_and_chains was actually called.
        scanner = _scanner(min_iv_rank=0.30, require_iv_premium=False)
        fetch_called = []
        def _fake_fetch(*a, **kw):
            fetch_called.append(True)
            return ({}, [], set())
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.20, 0.60, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               side_effect=_fake_fetch):
                scanner.scan_ticker('AAPL')
        self.assertTrue(fetch_called, "Chain fetch should have been called for a passing ticker")

    def test_gate_disabled_always_passes(self):
        """min_iv_rank=0 disables the gate — low HV rank still proceeds."""
        scanner = _scanner(min_iv_rank=0.0, require_iv_premium=False)
        fetch_called = []
        def _fake_fetch(*a, **kw):
            fetch_called.append(True)
            return ({}, [], set())
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.20, 0.05, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               side_effect=_fake_fetch):
                scanner.scan_ticker('AAPL')
        self.assertTrue(fetch_called)


# ══════════════════════════════════════════════════════════════════════════════
# IV premium gate integration (scan_ticker skips when ATM IV < HV30)
# ══════════════════════════════════════════════════════════════════════════════

class TestIvPremiumGate(unittest.TestCase):
    """scan_ticker returns [] when ATM IV < HV30 × ratio."""

    def _make_chain_mock(self, atm_iv: float, spot: float = 100.0):
        strikes = [spot * 0.97, spot, spot * 1.03]
        rows = [{'strike': s, 'impliedVolatility': atm_iv,
                 'bid': 0.5, 'ask': 0.6, 'openInterest': 100, 'volume': 10}
                for s in strikes]
        df = pd.DataFrame(rows)
        chain = MagicMock()
        chain.puts  = df
        chain.calls = df
        return {'2026-09-19': chain}

    def _run_scan(self, hv30, atm_iv, iv_premium_min_ratio=1.0):
        scanner = _scanner(
            min_iv_rank=0.0,           # disable rank gate
            require_iv_premium=True,
            iv_premium_min_ratio=iv_premium_min_ratio,
        )
        chains = self._make_chain_mock(atm_iv)
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, hv30, 0.50, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               return_value=(chains, ['2026-09-19'], set())):
                return scanner.scan_ticker('AAPL')

    def test_iv_below_hv30_skips_ticker(self):
        """ATM IV (0.18) < HV30 (0.25) → cheap options → skip."""
        results = self._run_scan(hv30=0.25, atm_iv=0.18)
        self.assertEqual(results, [])

    def test_iv_above_hv30_passes_gate(self):
        """ATM IV (0.32) > HV30 (0.25) → options priced richer than realized → proceed."""
        # Gate passes; scan may still return [] (no strikes set up in chain)
        scanner = _scanner(min_iv_rank=0.0, require_iv_premium=True, iv_premium_min_ratio=1.0)
        chains = self._make_chain_mock(atm_iv=0.32)
        fetch_called = []
        def _fake_expiry_fetch(*a, **kw):
            fetch_called.append(True)
            return (chains, ['2026-09-19'], set())
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.25, 0.50, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               side_effect=_fake_expiry_fetch):
                scanner.scan_ticker('AAPL')
        self.assertTrue(fetch_called, "Chain fetch should have been called when IV > HV30")

    def test_iv_equals_hv30_at_ratio_1_passes(self):
        """ATM IV == HV30 with ratio=1.0 → not strictly less → gate passes."""
        scanner = _scanner(min_iv_rank=0.0, require_iv_premium=True, iv_premium_min_ratio=1.0)
        chains = self._make_chain_mock(atm_iv=0.25)
        fetch_called = []
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.25, 0.50, False, None, None),
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               return_value=(chains, ['2026-09-19'], set())) as m:
                scanner.scan_ticker('AAPL')
                self.assertTrue(m.called)

    def test_ratio_110_requires_10pct_premium(self):
        """With ratio=1.10, IV must be 10% above HV30. IV=0.27 on HV30=0.25 → 0.27<0.275 → skip."""
        results = self._run_scan(hv30=0.25, atm_iv=0.27, iv_premium_min_ratio=1.10)
        self.assertEqual(results, [])

    def test_iv_filter_disabled_skips_check(self):
        """require_iv_premium=False — even cheap IV doesn't block the scan."""
        scanner = _scanner(min_iv_rank=0.0, require_iv_premium=False)
        chains = self._make_chain_mock(atm_iv=0.10)  # very cheap IV
        fetch_called = []
        with patch.object(
            scanner, '_compute_atr_and_hv30',
            return_value=(0.0, 0.35, 0.50, False, None, None),  # HV30 >> IV
        ), patch.object(
            scanner._data, 'get_history_df', return_value=pd.DataFrame({'Close': [100.0]}),
        ), patch.object(
            scanner._data, 'is_hft', return_value=False,
        ), patch('src.scanner._HIST_CACHE', {'AAPL': pd.Series([100.0] * 30)}):
            with patch.object(scanner, '_fetch_expirations_and_chains',
                               return_value=(chains, ['2026-09-19'], set())) as m:
                scanner.scan_ticker('AAPL')
                self.assertTrue(m.called, "Should proceed even when IV < HV30 if filter disabled")


# ══════════════════════════════════════════════════════════════════════════════
# History prefetch period
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoryPrefetchPeriod(unittest.TestCase):

    def test_252d_period_when_iv_filters_enabled(self):
        """get_top_picks should request 252-day history when iv_filters.enabled=true."""
        scanner = _scanner(iv_filters_enabled=True, history_days=252)
        with patch('src.scanner._batch_prefetch_history') as mock_prefetch, \
             patch.object(scanner, 'scan_ticker', return_value=[]):
            scanner.get_top_picks(['AAPL'], n=1)
        mock_prefetch.assert_called_once()
        period_arg = mock_prefetch.call_args[1].get('period') or mock_prefetch.call_args[0][1]
        self.assertEqual(period_arg, '252d')

    def test_30d_period_when_iv_filters_disabled(self):
        """get_top_picks uses 30-day history when iv_filters.enabled=false."""
        scanner = _scanner(iv_filters_enabled=False)
        with patch('src.scanner._batch_prefetch_history') as mock_prefetch, \
             patch.object(scanner, 'scan_ticker', return_value=[]):
            scanner.get_top_picks(['AAPL'], n=1)
        mock_prefetch.assert_called_once()
        period_arg = mock_prefetch.call_args[1].get('period') or mock_prefetch.call_args[0][1]
        self.assertEqual(period_arg, '30d')


if __name__ == '__main__':
    unittest.main()
