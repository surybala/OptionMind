"""
test_greeks.py
==============
Unit tests for src/greeks.py — Black-Scholes greeks and position risk scoring.

Covers both the B-S path (position_risk_score) and the pre-fetched-greeks
path (position_risk_score_from_greeks) introduced for HFT mode.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.greeks import bs_greeks, position_risk_score, position_risk_score_from_greeks


# ══════════════════════════════════════════════════════════════════════════════
# bs_greeks — Black-Scholes greeks for a single option
# ══════════════════════════════════════════════════════════════════════════════

class TestBsGreeks(unittest.TestCase):

    def test_call_delta_between_zero_and_one(self):
        g = bs_greeks(100, 105, 0.25, 30, 'call')
        self.assertGreater(g['delta'], 0.0)
        self.assertLess(g['delta'], 1.0)

    def test_put_delta_between_minus_one_and_zero(self):
        g = bs_greeks(100, 95, 0.25, 30, 'put')
        self.assertLess(g['delta'], 0.0)
        self.assertGreater(g['delta'], -1.0)

    def test_gamma_is_positive(self):
        g = bs_greeks(100, 100, 0.25, 30, 'call')
        self.assertGreater(g['gamma'], 0.0)

    def test_theta_is_negative_for_long(self):
        g = bs_greeks(100, 100, 0.25, 30, 'call')
        self.assertLess(g['theta'], 0.0)

    def test_put_call_parity_gamma(self):
        """Gamma is the same for a put and call at the same strike."""
        call_g = bs_greeks(100, 100, 0.30, 30, 'call')
        put_g  = bs_greeks(100, 100, 0.30, 30, 'put')
        self.assertAlmostEqual(call_g['gamma'], put_g['gamma'], places=6)

    def test_degenerate_dte_zero_returns_intrinsic_delta(self):
        g = bs_greeks(110, 100, 0.25, 0, 'call')
        self.assertEqual(g['delta'], 1.0)
        self.assertEqual(g['gamma'], 0.0)
        self.assertEqual(g['theta'], 0.0)

    def test_degenerate_iv_zero(self):
        g = bs_greeks(100, 100, 0.0, 30, 'call')
        self.assertEqual(g['gamma'], 0.0)
        self.assertEqual(g['theta'], 0.0)

    def test_case_insensitive_option_type(self):
        g_lower = bs_greeks(100, 100, 0.25, 30, 'call')
        g_upper = bs_greeks(100, 100, 0.25, 30, 'CALL')
        self.assertAlmostEqual(g_lower['delta'], g_upper['delta'], places=6)


# ══════════════════════════════════════════════════════════════════════════════
# position_risk_score — multi-leg risk using B-S internally
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionRiskScore(unittest.TestCase):

    def _csp_legs(self, strike=95.0, iv=0.25):
        return [{'strike': strike, 'iv': iv, 'option_type': 'put', 'position': 'short'}]

    def test_returns_all_required_keys(self):
        result = position_risk_score(100, self._csp_legs(), 30)
        for key in ['risk_score', 'gamma_theta_ratio', 'net_short_delta',
                    'net_gamma', 'net_theta']:
            self.assertIn(key, result)

    def test_net_gamma_is_negative_for_short(self):
        result = position_risk_score(100, self._csp_legs(), 30)
        self.assertLess(result['net_gamma'], 0.0)

    def test_net_theta_is_positive_for_short(self):
        result = position_risk_score(100, self._csp_legs(), 30)
        self.assertGreater(result['net_theta'], 0.0)

    def test_spread_reduces_gamma_vs_naked(self):
        spot = 100
        dte  = 14
        naked = position_risk_score(spot, [
            {'strike': 95, 'iv': 0.3, 'option_type': 'put', 'position': 'short'}
        ], dte)
        spread = position_risk_score(spot, [
            {'strike': 95, 'iv': 0.3, 'option_type': 'put', 'position': 'short'},
            {'strike': 90, 'iv': 0.3, 'option_type': 'put', 'position': 'long'},
        ], dte)
        self.assertLess(abs(spread['net_gamma']), abs(naked['net_gamma']))

    def test_degenerate_inputs_return_zeros(self):
        result = position_risk_score(0, [], 30)
        self.assertEqual(result['risk_score'], 0.0)
        result2 = position_risk_score(100, [], 0)
        self.assertEqual(result2['risk_score'], 0.0)

    def test_risk_score_increases_as_delta_approaches_itm(self):
        """A short put that is further OTM should have a lower risk score."""
        far_otm  = position_risk_score(100, [
            {'strike': 70, 'iv': 0.3, 'option_type': 'put', 'position': 'short'}
        ], 30)
        near_atm = position_risk_score(100, [
            {'strike': 98, 'iv': 0.3, 'option_type': 'put', 'position': 'short'}
        ], 30)
        self.assertLess(far_otm['risk_score'], near_atm['risk_score'])


# ══════════════════════════════════════════════════════════════════════════════
# position_risk_score_from_greeks — HFT path using pre-fetched broker greeks
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionRiskScoreFromGreeks(unittest.TestCase):

    def _short_put_leg(self, delta=-0.15, gamma=0.04, theta=-0.08):
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'position': 'short'}

    def _long_put_leg(self, delta=-0.05, gamma=0.02, theta=-0.03):
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'position': 'long'}

    def test_returns_all_required_keys(self):
        result = position_risk_score_from_greeks([self._short_put_leg()])
        for key in ['risk_score', 'gamma_theta_ratio', 'net_short_delta',
                    'net_gamma', 'net_theta']:
            self.assertIn(key, result)

    def test_empty_list_returns_zeros(self):
        result = position_risk_score_from_greeks([])
        self.assertEqual(result['risk_score'], 0.0)
        self.assertEqual(result['net_gamma'], 0.0)
        self.assertEqual(result['net_theta'], 0.0)

    def test_net_gamma_negative_for_single_short(self):
        """Short position inverts sign: net_gamma = -|broker_gamma|."""
        result = position_risk_score_from_greeks([self._short_put_leg(gamma=0.04)])
        self.assertAlmostEqual(result['net_gamma'], -0.04, places=6)

    def test_net_theta_positive_for_short(self):
        """Short option earns theta (positive from our perspective)."""
        result = position_risk_score_from_greeks([self._short_put_leg(theta=-0.08)])
        self.assertAlmostEqual(result['net_theta'], 0.08, places=6)

    def test_spread_reduces_net_gamma(self):
        """Long leg partially offsets the short leg's gamma."""
        naked  = position_risk_score_from_greeks([self._short_put_leg(gamma=0.04)])
        spread = position_risk_score_from_greeks([
            self._short_put_leg(gamma=0.04),
            self._long_put_leg(gamma=0.02),
        ])
        self.assertLess(abs(spread['net_gamma']), abs(naked['net_gamma']))

    def test_net_short_delta_accumulates_short_legs_only(self):
        result = position_risk_score_from_greeks([
            {'delta': -0.20, 'gamma': 0.03, 'theta': -0.05, 'position': 'short'},
            {'delta': -0.30, 'gamma': 0.04, 'theta': -0.06, 'position': 'short'},
            {'delta': -0.10, 'gamma': 0.01, 'theta': -0.02, 'position': 'long'},
        ])
        # net_short_delta = sum of delta values of short legs (raw, not sign-flipped)
        self.assertAlmostEqual(result['net_short_delta'], -0.50, places=6)

    def test_matches_bs_risk_score_structure(self):
        """
        position_risk_score_from_greeks and position_risk_score should yield
        the same dict shape (same keys).  Values differ because one uses
        broker greeks and the other uses B-S, but the structure must match.
        """
        from_greeks = position_risk_score_from_greeks([self._short_put_leg()])
        from_bs     = position_risk_score(100, [
            {'strike': 95, 'iv': 0.25, 'option_type': 'put', 'position': 'short'}
        ], 30)
        self.assertEqual(set(from_greeks.keys()), set(from_bs.keys()))

    def test_high_delta_triggers_risk_penalty(self):
        """Risk score should be higher when the short leg delta is above the neutral zone."""
        safe_otm = position_risk_score_from_greeks([
            {'delta': -0.05, 'gamma': 0.02, 'theta': -0.05, 'position': 'short'}
        ])
        deep_itm = position_risk_score_from_greeks([
            {'delta': -0.45, 'gamma': 0.02, 'theta': -0.05, 'position': 'short'}
        ])
        self.assertGreater(deep_itm['risk_score'], safe_otm['risk_score'])

    def test_leg_missing_greeks_is_skipped(self):
        """A leg without delta/gamma/theta should be silently skipped."""
        result = position_risk_score_from_greeks([
            {'delta': None, 'gamma': 0.02, 'theta': -0.05, 'position': 'short'},
            {'delta': -0.15, 'gamma': 0.03, 'theta': -0.06, 'position': 'short'},
        ])
        # Only the second leg (complete) should contribute
        self.assertAlmostEqual(result['net_gamma'], -0.03, places=6)

    def test_gamma_theta_ratio_with_near_zero_theta(self):
        """When theta is near-zero, ratio = abs_gamma * 1000 (high-risk convention)."""
        # With gamma=1.5 and near-zero theta: ratio = 1.5 * 1000 = 1500.0
        result = position_risk_score_from_greeks([
            {'delta': -0.10, 'gamma': 1.5, 'theta': -1e-10, 'position': 'short'}
        ])
        # abs_gamma (net) = 1.5, ratio = 1.5 * 1000 = 1500
        self.assertAlmostEqual(result['gamma_theta_ratio'], 1500.0, places=1)
        self.assertGreater(result['gamma_theta_ratio'], 100.0)


if __name__ == '__main__':
    unittest.main()
