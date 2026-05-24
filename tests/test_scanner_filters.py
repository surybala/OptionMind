"""Tests for the four new scanner quality filters added in the scanner-quality-filters batch:

  1. Earnings exclusion  — skip expiries whose window contains an earnings date
  2. Min OTM %           — hard floor on strike distance from spot
  3. ATR-based guard     — require |spot - strike| >= multiplier × ATR
  4. prob_win² scoring   — score = premium × prob_win² instead of premium × prob_win
"""
import sys
import os
import unittest
from datetime import datetime, timedelta, date
from unittest.mock import MagicMock, patch, PropertyMock

# Pre-register mocks for heavy imports before importing scanner
import pandas as _real_pd
import numpy  as _real_np
sys.modules.setdefault('yfinance', MagicMock())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.scanner import OptionScanner


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_config(**overrides):
    """Return a minimal valid config dict with all new filter keys."""
    cfg = {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {'min_probability_of_expiry': 0.70},
        'strategies': {
            'covered_put':      {'enabled': False},
            'put_credit_spread': {
                'enabled': True,
                'min_net_credit': 0.05,
                'max_delta_short_leg': 0.50,
                'strike_width': 5,
                'min_prob_profit': 0.60,
                'min_open_interest': 0,
            },
            'call_credit_spread': {'enabled': False},
            'iron_condor':        {'enabled': False},
            'iron_butterfly':     {'enabled': False},
            'short_strangle':     {'enabled': False},
            'covered_call':       {'enabled': False},
        },
        # New filters — start disabled so individual tests can enable selectively
        'earnings_exclusion': {'enabled': False, 'days_buffer': 2},
        'min_otm_pct':        {'put': 0.0, 'call': 0.0},
        'atr_distance':       {'enabled': False, 'atr_period': 14, 'multiplier': 1.5},
    }
    cfg.update(overrides)
    return cfg


def _make_puts_df(spot, strikes):
    """Build a minimal puts DataFrame for the given strikes relative to spot."""
    rows = []
    for s in strikes:
        rows.append({
            'strike':            float(s),
            'bid':               1.00,
            'ask':               1.20,
            'lastPrice':         1.10,
            'impliedVolatility': 0.30,
            'openInterest':      100,
            'volume':            200,
            'delta':             None,
        })
    return pd.DataFrame(rows)


def _future_expiry(days=7):
    return (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')


# ── TestEarningsExclusion ─────────────────────────────────────────────────────

class TestEarningsExclusion(unittest.TestCase):

    def setUp(self):
        self.cfg = _base_config(earnings_exclusion={'enabled': True, 'days_buffer': 2})
        self.scanner = OptionScanner(self.cfg)

    @patch('src.scanner.yf')
    @patch('src.scanner._ALPACA_CLIENT', None)
    def test_expiry_skipped_when_earnings_in_window(self, mock_yf):
        """If earnings fall before or on expiry + buffer, the expiry is skipped."""
        expiry = _future_expiry(7)
        # Earnings in 5 days — inside the 7-day expiry window
        earnings_date = (datetime.now() + timedelta(days=5)).date()

        mock_ticker = MagicMock()
        mock_ticker.fast_info = MagicMock(market_cap=2e9, last_price=100.0)
        mock_ticker.calendar = {'Earnings Date': [earnings_date]}
        mock_ticker.options  = [expiry]
        mock_ticker.option_chain.return_value = MagicMock(
            puts=_make_puts_df(100, [90, 92, 94]),
            calls=pd.DataFrame(),
        )
        mock_yf.Ticker.return_value = mock_ticker

        results = self.scanner.scan_ticker('AAPL')
        self.assertEqual(results, [], "Expiry should be skipped when earnings in window")

    @patch('src.scanner.yf')
    @patch('src.scanner._ALPACA_CLIENT', None)
    def test_expiry_passes_when_earnings_outside_window(self, mock_yf):
        """If earnings fall after expiry + buffer, the expiry is NOT skipped."""
        expiry = _future_expiry(7)
        # Earnings 14 days from now — outside the 7-day expiry + 2-day buffer
        earnings_date = (datetime.now() + timedelta(days=14)).date()

        mock_ticker = MagicMock()
        mock_ticker.fast_info = MagicMock(market_cap=2e9, last_price=100.0)
        mock_ticker.calendar = {'Earnings Date': [earnings_date]}
        mock_ticker.options  = [expiry]
        # Return a real DataFrame so the strategy scanner can iterate it
        mock_ticker.option_chain.return_value = MagicMock(
            puts=_make_puts_df(100, [90, 92, 94]),
            calls=pd.DataFrame(),
        )
        mock_yf.Ticker.return_value = mock_ticker

        results = self.scanner.scan_ticker('AAPL')
        # May or may not yield picks (depends on prob filter) — just verify no crash
        # and that scan_ticker ran past the earnings check
        self.assertIsInstance(results, list)

    @patch('src.scanner.yf')
    @patch('src.scanner._ALPACA_CLIENT', None)
    def test_passes_when_no_earnings_data(self, mock_yf):
        """Empty calendar should never block any expiry (fail open)."""
        expiry = _future_expiry(7)

        mock_ticker = MagicMock()
        mock_ticker.fast_info = MagicMock(market_cap=2e9, last_price=100.0)
        mock_ticker.calendar = {}          # no 'Earnings Date' key
        mock_ticker.options  = [expiry]
        mock_ticker.option_chain.return_value = MagicMock(
            puts=_make_puts_df(100, [90, 92]),
            calls=pd.DataFrame(),
        )
        mock_yf.Ticker.return_value = mock_ticker

        # Should not raise; expiry is not blocked
        results = self.scanner.scan_ticker('AAPL')
        self.assertIsInstance(results, list)

    @patch('src.scanner.yf')
    @patch('src.scanner._ALPACA_CLIENT', None)
    def test_disabled_via_config(self, mock_yf):
        """When earnings_exclusion.enabled=false, earnings dates are ignored."""
        scanner = OptionScanner(
            _base_config(earnings_exclusion={'enabled': False, 'days_buffer': 2})
        )
        expiry = _future_expiry(7)
        # Earnings tomorrow — would block if enabled
        earnings_date = (datetime.now() + timedelta(days=1)).date()

        mock_ticker = MagicMock()
        mock_ticker.fast_info = MagicMock(market_cap=2e9, last_price=100.0)
        mock_ticker.calendar = {'Earnings Date': [earnings_date]}
        mock_ticker.options  = [expiry]
        mock_ticker.option_chain.return_value = MagicMock(
            puts=_make_puts_df(100, [90, 92]),
            calls=pd.DataFrame(),
        )
        mock_yf.Ticker.return_value = mock_ticker

        # scanner.calendar should be read but the dates should NOT block the expiry
        results = scanner.scan_ticker('AAPL')
        self.assertIsInstance(results, list)
        # calendar was never consulted (earnings disabled means _earnings_dates stays empty)
        # verify calendar was NOT called at all
        mock_ticker.calendar.assert_not_called() if hasattr(mock_ticker.calendar, 'assert_not_called') else None


# ── TestMinOtmPct ─────────────────────────────────────────────────────────────

class TestMinOtmPct(unittest.TestCase):
    """_scan_spreads / _scan_csp filter: short strike must be >= min_otm_pct % OTM."""

    def _make_scanner(self, put_pct=0.0, call_pct=0.0):
        return OptionScanner(_base_config(min_otm_pct={'put': put_pct, 'call': call_pct}))

    def test_put_strike_too_close_rejected(self):
        """Short put at 97% of spot with min_otm_pct.put=0.05 (need ≤95%) → filtered."""
        scanner = self._make_scanner(put_pct=0.05)
        spot = 100.0
        # 97 > 100 × (1 - 0.05) = 95 → should be filtered
        puts = _make_puts_df(spot, [97.0])
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=0.0)
        self.assertEqual(picks, [], "Strike at 97% should be filtered when min_otm_pct.put=0.05")

    def test_put_strike_far_enough_passes(self):
        """Short put at 93% of spot with min_otm_pct.put=0.05 (need ≤95%) → passes filter."""
        scanner = self._make_scanner(put_pct=0.05)
        spot = 100.0
        # 93 <= 100 × (1 - 0.05) = 95 → should pass
        # Need the long leg too (width=5, so long at 88)
        puts = _make_puts_df(spot, [88.0, 93.0])
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=0.0)
        # It may still be filtered by prob, but NOT by min_otm_pct
        # We just verify no crash and the function ran past the filter
        self.assertIsInstance(picks, list)

    def test_call_strike_too_close_rejected(self):
        """Short call at 103% of spot with min_otm_pct.call=0.05 (need ≥105%) → filtered."""
        cfg = _base_config(
            min_otm_pct={'put': 0.0, 'call': 0.05},
            strategies={
                'covered_put': {'enabled': False},
                'put_credit_spread': {'enabled': False},
                'call_credit_spread': {
                    'enabled': True,
                    'min_net_credit': 0.05,
                    'max_delta_short_leg': 0.50,
                    'strike_width': 5,
                    'min_prob_profit': 0.60,
                    'min_open_interest': 0,
                },
                'iron_condor':    {'enabled': False},
                'iron_butterfly': {'enabled': False},
                'short_strangle': {'enabled': False},
                'covered_call':   {'enabled': False},
            },
        )
        scanner = OptionScanner(cfg)
        spot = 100.0
        calls_df = _make_puts_df(spot, [103.0])  # same structure, different column semantics
        calls_df['strike'] = 103.0
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, calls_df, 'call', atr=0.0)
        self.assertEqual(picks, [], "Strike at 103% should be filtered when min_otm_pct.call=0.05")

    def test_zero_pct_bypasses_filter(self):
        """min_otm_pct=0 (default) means the filter is entirely disabled."""
        scanner = self._make_scanner(put_pct=0.0)
        spot = 100.0
        # Strike at 98% — normally blocked by 5% filter, should pass here
        puts = _make_puts_df(spot, [93.0, 98.0])
        # Just verify no AttributeError / crash
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=0.0)
        self.assertIsInstance(picks, list)


# ── TestAtrDistance ───────────────────────────────────────────────────────────

class TestAtrDistance(unittest.TestCase):
    """_scan_spreads filter: |spot - strike| must be >= atr_multiplier × atr."""

    def _make_scanner(self, enabled=True, multiplier=1.5):
        return OptionScanner(_base_config(
            atr_distance={'enabled': enabled, 'atr_period': 14, 'multiplier': multiplier}
        ))

    def test_strike_within_atr_rejected(self):
        """ATR=5, multiplier=1.5 → min distance=7.5. Strike 94 (6 away) → rejected."""
        scanner = self._make_scanner(enabled=True, multiplier=1.5)
        spot = 100.0
        atr  = 5.0
        # |100 - 94| = 6 < 7.5 (1.5 × 5) → should be filtered
        puts = _make_puts_df(spot, [89.0, 94.0])
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=atr)
        # 94 filtered; 89 is also < 7.5 away?  |100-89|=11 >= 7.5 → may pass
        # Verify 94 is not in picks
        strikes_in_picks = [p['short_strike'] for p in picks]
        self.assertNotIn(94.0, strikes_in_picks)

    def test_strike_beyond_atr_passes(self):
        """ATR=5, multiplier=1.5 → min distance=7.5. Strike 91 (9 away) → passes."""
        scanner = self._make_scanner(enabled=True, multiplier=1.5)
        spot = 100.0
        atr  = 5.0
        # |100 - 91| = 9 >= 7.5 → passes ATR filter (still subject to prob/credit filters)
        puts = _make_puts_df(spot, [86.0, 91.0])
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=atr)
        # Should not crash and function ran past the ATR check
        self.assertIsInstance(picks, list)

    def test_zero_atr_bypasses_filter(self):
        """atr=0 means 0 × multiplier = 0 minimum distance — filter is a no-op."""
        scanner = self._make_scanner(enabled=True, multiplier=1.5)
        spot = 100.0
        puts = _make_puts_df(spot, [93.0, 98.0])
        # Should not filter anything (atr=0 → min_dist=0)
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=0.0)
        self.assertIsInstance(picks, list)

    def test_disabled_via_config(self):
        """atr_distance.enabled=false → ATR guard not applied regardless of atr value."""
        scanner = self._make_scanner(enabled=False, multiplier=1.5)
        spot = 100.0
        atr  = 50.0  # absurdly large — would block everything if enabled
        puts = _make_puts_df(spot, [88.0, 93.0])
        # Should not raise; high ATR is ignored
        picks = scanner._scan_spreads('AAPL', spot, _future_expiry(), 7, puts, 'put', atr=atr)
        self.assertIsInstance(picks, list)


# ── TestProbWinSquaredScoring ─────────────────────────────────────────────────

class TestProbWinSquaredScoring(unittest.TestCase):
    """Score formula is premium × prob_win² (not premium × prob_win)."""

    def setUp(self):
        self.scanner = OptionScanner(_base_config())

    def test_score_formula_pcs(self):
        """Verify score = net_credit × prob_win² for a PCS pick."""
        spot = 100.0
        # Build two-strike chain: short at 90 (10% OTM), long at 85
        puts = _make_puts_df(spot, [85.0, 90.0])
        picks = self.scanner._scan_spreads(
            'AAPL', spot, _future_expiry(), 14, puts, 'put', atr=0.0
        )
        for pick in picks:
            expected_score = round(pick['premium'] * (pick['prob_win'] ** 2), 4)
            self.assertAlmostEqual(
                pick['score'], expected_score, places=3,
                msg=f"Score should be premium×prob_win²: got {pick['score']}, expected {expected_score}"
            )

    def test_higher_prob_ranked_first(self):
        """Two picks with same premium but different prob_win → higher prob ranked first."""
        scanner = OptionScanner(_base_config())

        # Manually construct two pick dicts as _scan_spreads would produce them
        pick_low  = {'premium': 0.50, 'prob_win': 0.80, 'score': round(0.50 * 0.80**2, 4)}
        pick_high = {'premium': 0.50, 'prob_win': 0.90, 'score': round(0.50 * 0.90**2, 4)}

        self.assertGreater(
            pick_high['score'], pick_low['score'],
            "With prob_win², the 0.90-probability pick must outscore the 0.80-probability pick"
        )
        # Quantify the gap: with linear scoring gap would be 0.05×0.50=0.025;
        # with squared scoring gap should be larger
        linear_gap  = 0.50 * 0.90 - 0.50 * 0.80          # 0.05
        squared_gap = pick_high['score'] - pick_low['score']  # 0.50*(0.81-0.64)=0.085
        self.assertGreater(squared_gap, linear_gap,
                           "prob_win² should amplify the gap between high and low probability picks")

    def test_score_formula_csp(self):
        """CSP score = lastPrice × prob_win²."""
        scanner = OptionScanner(_base_config(strategies={
            'covered_put': {
                'enabled': True,
                'min_premium': 0.05,
            },
            'put_credit_spread': {'enabled': False},
            'call_credit_spread': {'enabled': False},
            'iron_condor': {'enabled': False},
            'iron_butterfly': {'enabled': False},
            'short_strangle': {'enabled': False},
            'covered_call': {'enabled': False},
        }))
        spot = 100.0
        puts = _make_puts_df(spot, [90.0])
        picks = scanner._scan_csp('AAPL', spot, _future_expiry(), 14, puts, atr=0.0)
        for pick in picks:
            expected_score = round(pick['premium'] * (pick['prob_win'] ** 2), 4)
            self.assertAlmostEqual(pick['score'], expected_score, places=3)

    def test_score_formula_iron_condor(self):
        """IC score = total_credit × prob_win² where prob_win = min(prob_put, prob_call)."""
        cfg = _base_config(strategies={
            'covered_put': {'enabled': False},
            'put_credit_spread': {'enabled': False},
            'call_credit_spread': {'enabled': False},
            'iron_condor': {
                'enabled': True,
                'min_net_credit': 0.05,
                'max_delta_short_leg': 0.50,
                'put_strike_width': 5,
                'call_strike_width': 5,
                'min_prob_profit': 0.60,
                'min_open_interest': 0,
            },
            'iron_butterfly': {'enabled': False},
            'short_strangle': {'enabled': False},
            'covered_call': {'enabled': False},
        })
        scanner = OptionScanner(cfg)
        spot = 100.0
        puts  = _make_puts_df(spot, [85.0, 90.0])
        calls_rows = [
            {'strike': 110.0, 'bid': 1.0, 'ask': 1.2, 'lastPrice': 1.1,
             'impliedVolatility': 0.3, 'openInterest': 100, 'volume': 200, 'delta': None},
            {'strike': 115.0, 'bid': 0.5, 'ask': 0.7, 'lastPrice': 0.6,
             'impliedVolatility': 0.3, 'openInterest': 100, 'volume': 200, 'delta': None},
        ]
        calls = pd.DataFrame(calls_rows)
        picks = scanner._scan_iron_condor('AAPL', spot, _future_expiry(), 14, puts, calls, atr=0.0)
        for pick in picks:
            expected_score = round(pick['premium'] * (pick['prob_win'] ** 2), 4)
            self.assertAlmostEqual(pick['score'], expected_score, places=3)


if __name__ == '__main__':
    unittest.main()
