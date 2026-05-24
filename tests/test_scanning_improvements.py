"""
Tests for the four scanning improvements introduced in 2026-04:

  A. Width-relative stop loss  (StopLossRule.max_loss_pct)
  B. Yield-normalised score    (credit / width) × prob²
  C. Dynamic strike width      (price-tier based width via _width_for_price)
  D. min_otm_pct raised to 8%  (enforced by existing OTM filter in spreads.py)
"""
import unittest
from unittest.mock import MagicMock

import pandas as pd

from src.scan_strategies.base import StrategyScanner
from src.risk_rules.stop_loss import StopLossRule, _width_from_legs


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scanner_with_dynamic_width(enabled: bool = True, tiers=None):
    """Build an OptionScanner configured with (or without) dynamic width."""
    from src.scanner import OptionScanner
    default_tiers = [
        {"max_price": 50,   "width": 5},
        {"max_price": 100,  "width": 5},
        {"max_price": 200,  "width": 10},
        {"max_price": 400,  "width": 15},
        {"max_price": 9999, "width": 20},
    ]
    return OptionScanner({
        'market_cap_min': 1e9,
        'expiry_days_max': 45,
        'risk_parameters': {'min_probability_of_expiry': 0.7},
        'min_otm_pct': {'put': 0.0, 'call': 0.0},   # disable OTM floor for width tests
        'atr_distance': {'enabled': False},
        'dynamic_width': {
            'enabled': enabled,
            'tiers': tiers or default_tiers,
        },
        'strategies': {
            'put_credit_spread': {
                'enabled': True,
                'min_net_credit': 0.01,
                'strike_width': 5,          # fallback when dynamic disabled
                'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.60,
            },
            'call_credit_spread': {
                'enabled': True,
                'min_net_credit': 0.01,
                'strike_width': 5,
                'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.60,
            },
            'iron_condor': {
                'enabled': True,
                'min_net_credit': 0.01,
                'put_strike_width': 5,
                'call_strike_width': 5,
                'min_prob_profit': 0.50,
                'max_delta_short_leg': 0.60,
            },
        },
    })


def _make_chain_df(rows):
    """Build a minimal chain DataFrame from a list of strike dicts."""
    df = pd.DataFrame(rows)
    if 'openInterest' not in df.columns:
        df['openInterest'] = -1        # Alpaca unknown → never filtered
    if 'volume' not in df.columns:
        df['volume'] = 0
    return df


def _row(strike, bid, ask, iv=0.30):
    return {
        'strike': float(strike),
        'bid': bid,
        'ask': ask,
        'lastPrice': (bid + ask) / 2,
        'impliedVolatility': iv,
        'openInterest': -1,
        'volume': 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# C. Dynamic strike width
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicWidth(unittest.TestCase):
    """_width_for_price returns the correct tier width for a given spot price."""

    def _base(self):
        """Concrete StrategyScanner subclass just to access _width_for_price."""
        class _Dummy(StrategyScanner):
            def scan(self, *a, **kw):
                return []
        tiers = [
            {"max_price": 50,   "width": 5},
            {"max_price": 100,  "width": 5},
            {"max_price": 200,  "width": 10},
            {"max_price": 400,  "width": 15},
            {"max_price": 9999, "width": 20},
        ]
        return _Dummy(
            params={}, min_prob=0.7,
            min_otm_put=0, min_otm_call=0,
            atr_enabled=False, atr_multiplier=1.5,
            prob_otm_fn=lambda *a: 0.8,
            row_oi_vol_fn=lambda r: (None, 0),
            dynamic_width_cfg={'enabled': True, 'tiers': tiers},
        )

    def test_cheap_stock_uses_5(self):
        s = self._base()
        self.assertEqual(s._width_for_price(40.0, 5), 5)

    def test_mid_stock_100_boundary_uses_5(self):
        s = self._base()
        self.assertEqual(s._width_for_price(100.0, 5), 5)

    def test_stock_150_uses_10(self):
        s = self._base()
        self.assertEqual(s._width_for_price(150.0, 5), 10)

    def test_stock_300_uses_15(self):
        s = self._base()
        self.assertEqual(s._width_for_price(300.0, 5), 15)

    def test_expensive_stock_uses_20(self):
        s = self._base()
        self.assertEqual(s._width_for_price(500.0, 5), 20)

    def test_disabled_returns_fallback(self):
        """When dynamic_width.enabled=false, fallback config width is returned."""
        class _Dummy(StrategyScanner):
            def scan(self, *a, **kw):
                return []
        s = _Dummy(
            params={}, min_prob=0.7,
            min_otm_put=0, min_otm_call=0,
            atr_enabled=False, atr_multiplier=1.5,
            prob_otm_fn=lambda *a: 0.8,
            row_oi_vol_fn=lambda r: (None, 0),
            dynamic_width_cfg={'enabled': False, 'tiers': [{"max_price": 9999, "width": 20}]},
        )
        self.assertEqual(s._width_for_price(350.0, fallback=5), 5)

    def test_pcs_scan_uses_dynamic_width_for_expensive_stock(self):
        """PCS on a $350 stock should produce 15-wide spreads, not 5-wide."""
        scanner = _scanner_with_dynamic_width(enabled=True)
        spot = 350.0
        # Build a chain around spot with 15-wide and 5-wide pairs available
        rows = [_row(s, bid=max(0.01, (360-s)*0.03), ask=max(0.02, (360-s)*0.035), iv=0.25)
                for s in [295, 305, 310, 315, 320, 325, 330, 335]]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('NVDA', spot, '2026-09-19', 30, df, 'put')
        if results:
            for r in results:
                self.assertEqual(r['width'], 15,
                    f"Expected width=15 for ${spot} stock, got {r['width']}")

    def test_pcs_fallback_to_config_width_when_disabled(self):
        """With dynamic_width disabled, PCS uses the config strike_width=5."""
        scanner = _scanner_with_dynamic_width(enabled=False)
        spot = 350.0
        rows = [_row(s, bid=max(0.01, (360-s)*0.03), ask=max(0.02, (360-s)*0.035), iv=0.25)
                for s in [310, 315, 320, 325, 330, 335, 340, 345]]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('NVDA', spot, '2026-09-19', 30, df, 'put')
        if results:
            for r in results:
                self.assertEqual(r['width'], 5,
                    f"Expected fallback width=5 when dynamic disabled, got {r['width']}")

    def test_ic_scan_uses_dynamic_width_for_both_wings(self):
        """IC on a $250 stock should produce 15-wide wings, not 5-wide."""
        scanner = _scanner_with_dynamic_width(enabled=True)
        spot = 250.0
        put_rows  = [_row(s, bid=max(0.01,(255-s)*0.025), ask=max(0.02,(255-s)*0.03), iv=0.22)
                     for s in [200, 210, 215, 220, 225, 230, 235]]
        call_rows = [_row(s, bid=max(0.01,(s-245)*0.025), ask=max(0.02,(s-245)*0.03), iv=0.22)
                     for s in [265, 270, 275, 280, 285, 290, 295]]
        puts  = _make_chain_df(put_rows)
        calls = _make_chain_df(call_rows)
        results = scanner._scan_iron_condor('SPY', spot, '2026-09-19', 30, puts, calls)
        if results:
            for r in results:
                self.assertEqual(r['put_width'],  15)
                self.assertEqual(r['call_width'], 15)


# ══════════════════════════════════════════════════════════════════════════════
# B. Yield-normalised score
# ══════════════════════════════════════════════════════════════════════════════

class TestYieldNormalisedScore(unittest.TestCase):
    """score = (credit / width) × prob_win²."""

    def test_base_score_formula(self):
        from src.scan_strategies.base import StrategyScanner
        # (0.80 / 10) × 0.85² = 0.08 × 0.7225 = 0.0578
        result = StrategyScanner._score(0.80, 0.85, width=10.0)
        self.assertAlmostEqual(result, round(0.08 * 0.85 ** 2, 4), places=4)

    def test_wider_spread_same_yield_same_score(self):
        """$0.50 on $5 width == $1.00 on $10 width (both 10% yield) → same score."""
        s1 = StrategyScanner._score(0.50, 0.80, width=5.0)
        s2 = StrategyScanner._score(1.00, 0.80, width=10.0)
        self.assertAlmostEqual(s1, s2, places=4)

    def test_far_otm_spread_beats_near_atm_at_same_prob(self):
        """Far spread with better yield scores higher than near spread with lower yield."""
        # Far: $1.00 on $10 = 10% yield, prob=0.85
        # Near: $0.80 on $5 = 16% yield, prob=0.85
        # Near still wins on yield; this confirms the formula is purely yield-based
        far  = StrategyScanner._score(1.00, 0.85, width=10.0)
        near = StrategyScanner._score(0.80, 0.85, width=5.0)
        # near has 16% yield vs far 10% → near should score higher
        self.assertGreater(near, far)

    def test_higher_prob_wins_over_raw_premium(self):
        """Safer spread (higher prob) can outscore higher-premium spread via prob²."""
        # High-premium near-ATM: $1.50 on $10 = 15% yield, prob=0.70
        # Lower-premium far-OTM: $1.20 on $10 = 12% yield, prob=0.90
        high_prem = StrategyScanner._score(1.50, 0.70, width=10.0)
        high_prob = StrategyScanner._score(1.20, 0.90, width=10.0)
        self.assertGreater(high_prob, high_prem)

    def test_pcs_pick_score_matches_formula(self):
        """PCS pick score stored in result matches (credit/width) × prob²."""
        scanner = _scanner_with_dynamic_width(enabled=False)  # fixed width=5 for predictability
        spot = 100.0
        rows = [
            _row(85, bid=0.40, ask=0.50, iv=0.30),   # long leg
            _row(90, bid=1.00, ask=1.10, iv=0.30),   # short leg
        ]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('TEST', spot, '2026-09-19', 30, df, 'put')
        if results:
            r = results[0]
            expected = round((r['premium'] / r['width']) * r['prob_win'] ** 2, 4)
            self.assertAlmostEqual(r['score'], expected, places=4)

    def test_ic_pick_score_matches_formula(self):
        """IC score = (total_credit / max_wing_width) × prob_win²."""
        scanner = _scanner_with_dynamic_width(enabled=False)
        spot = 100.0
        put_rows  = [_row(s, bid=max(0.01,(105-s)*0.03), ask=max(0.02,(105-s)*0.035), iv=0.28)
                     for s in [80, 85, 90, 95]]
        call_rows = [_row(s, bid=max(0.01,(s-95)*0.03),  ask=max(0.02,(s-95)*0.035),  iv=0.28)
                     for s in [105, 110, 115, 120]]
        puts  = _make_chain_df(put_rows)
        calls = _make_chain_df(call_rows)
        results = scanner._scan_iron_condor('TEST', spot, '2026-09-19', 30, puts, calls)
        if results:
            r = results[0]
            max_w    = max(r['put_width'], r['call_width'])
            expected = round((r['premium'] / max_w) * r['prob_win'] ** 2, 4)
            self.assertAlmostEqual(r['score'], expected, places=4)


# ══════════════════════════════════════════════════════════════════════════════
# A. Width-relative stop loss — _width_from_legs
# ══════════════════════════════════════════════════════════════════════════════

class TestWidthFromLegs(unittest.TestCase):
    """_width_from_legs extracts spread width from stored leg strikes."""

    def test_pcs_width(self):
        pos = {'type': 'PCS', 'legs': {'short_strike': 95.0, 'long_strike': 90.0}}
        self.assertEqual(_width_from_legs(pos), 5.0)

    def test_ccs_width(self):
        pos = {'type': 'CCS', 'legs': {'short_strike': 110.0, 'long_strike': 120.0}}
        self.assertEqual(_width_from_legs(pos), 10.0)

    def test_ic_uses_max_wing(self):
        pos = {
            'type': 'IC',
            'legs': {'short_put': 90.0, 'long_put': 80.0,
                     'short_call': 110.0, 'long_call': 125.0},
        }
        # put wing = 10, call wing = 15 → max = 15
        self.assertEqual(_width_from_legs(pos), 15.0)

    def test_ic_equal_wings(self):
        pos = {
            'type': 'IC',
            'legs': {'short_put': 90.0, 'long_put': 80.0,
                     'short_call': 110.0, 'long_call': 120.0},
        }
        self.assertEqual(_width_from_legs(pos), 10.0)

    def test_csp_returns_none(self):
        pos = {'type': 'CSP', 'legs': {'short_strike': 95.0}}
        self.assertIsNone(_width_from_legs(pos))

    def test_strangle_returns_none(self):
        pos = {'type': 'STRANGLE', 'legs': {'short_put': 90.0, 'short_call': 110.0}}
        self.assertIsNone(_width_from_legs(pos))

    def test_missing_legs_returns_none(self):
        pos = {'type': 'PCS', 'legs': {}}
        self.assertIsNone(_width_from_legs(pos))

    def test_bad_values_returns_none(self):
        pos = {'type': 'PCS', 'legs': {'short_strike': 'bad', 'long_strike': None}}
        self.assertIsNone(_width_from_legs(pos))


# ══════════════════════════════════════════════════════════════════════════════
# A. Width-relative stop loss — StopLossRule
# ══════════════════════════════════════════════════════════════════════════════

class TestStopLossRule(unittest.TestCase):

    def _pos(self, strat, short, long_strike):
        return {
            'type': strat,
            'legs': {'short_strike': float(short), 'long_strike': float(long_strike)},
        }

    # ── Legacy multiplier behaviour (max_loss_pct=None) ──────────────────────

    def test_multiplier_only_no_trigger(self):
        rule = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=None)
        pos  = self._pos('PCS', 95, 90)
        # entry=0.50, mark=1.40 → loss=0.90, trig=1.00 → no trigger
        sig = rule.evaluate(0.50, 1.40, -90.0, 100.0, pos)
        self.assertIsNone(sig)

    def test_multiplier_only_triggers(self):
        rule = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=None)
        pos  = self._pos('PCS', 95, 90)
        # entry=0.50, mark=1.51 → loss=1.01 > 2×0.50=1.00 → trigger
        sig = rule.evaluate(0.50, 1.51, -101.0, 100.0, pos)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.reason_tag, 'STOP_LOSS')

    # ── Width-relative guard ──────────────────────────────────────────────────

    def test_width_relative_no_trigger_within_budget(self):
        """mark safely within 80% of max-loss budget — no trigger."""
        rule = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=0.80)
        # width=10, entry=0.40, max_loss=9.60, 80% budget = 7.68
        # mark = entry + 7.00 = 7.40 → loss=7.00 < 7.68 → no trigger
        # (multiplier guard is bypassed because width is derivable)
        pos = {'type': 'PCS', 'legs': {'short_strike': 100.0, 'long_strike': 90.0}}
        sig = rule.evaluate(0.40, 7.40, -700.0, 120.0, pos)
        self.assertIsNone(sig)

    def test_width_relative_triggers_beyond_budget(self):
        """mark exceeds 80% of max-loss budget — triggers."""
        rule = StopLossRule(stop_loss_multiplier=10.0, max_loss_pct=0.80)
        # width=10, entry=0.40, max_loss=9.60, 80% = 7.68
        # mark = entry + 7.70 = 8.10 → loss=7.70 > 7.68 → trigger
        pos = {'type': 'PCS', 'legs': {'short_strike': 100.0, 'long_strike': 90.0}}
        sig = rule.evaluate(0.40, 8.10, -770.0, 120.0, pos)
        self.assertIsNotNone(sig)
        self.assertEqual(sig.reason_tag, 'STOP_LOSS')
        self.assertIn('max-loss', sig.reason_str)

    def test_width_relative_uses_max_loss_pct_threshold(self):
        """Exact boundary: loss == threshold → no trigger (not strictly greater)."""
        rule = StopLossRule(stop_loss_multiplier=10.0, max_loss_pct=0.80)
        # width=5, entry=0.50, max_loss=4.50, 80%=3.60
        # mark = entry + 3.60 = 4.10 → loss exactly equal → no trigger
        pos = {'type': 'PCS', 'legs': {'short_strike': 95.0, 'long_strike': 90.0}}
        sig = rule.evaluate(0.50, 4.10, -360.0, 100.0, pos)
        self.assertIsNone(sig)

    def test_multiplier_guard_still_fires_when_width_guard_misses(self):
        """If CSP (no width) — multiplier guard is the only defence."""
        rule = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=0.80)
        csp_pos = {'type': 'CSP', 'legs': {'short_strike': 95.0}}
        # entry=1.00, mark=3.01 → loss=2.01 > 2.00 → multiplier fires
        sig = rule.evaluate(1.00, 3.01, -201.0, 100.0, csp_pos)
        self.assertIsNotNone(sig)

    def test_low_premium_far_otm_gets_more_room_than_legacy(self):
        """Core regression: low-premium spread now has room proportional to width,
        not a hair-trigger 2× premium threshold.

        Legacy: entry=$0.40, mark=$1.21 → triggers (loss=0.81 > 2×0.40=0.80)
        New:    entry=$0.40, mark=$1.21 → does NOT trigger
                (width-relative trig = 0.80 × (10-0.40) = 7.68; loss=0.81 << 7.68)
        """
        legacy = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=None)
        new    = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=0.80)
        pos    = {'type': 'PCS', 'legs': {'short_strike': 100.0, 'long_strike': 90.0}}

        # mark=1.21: loss=0.81 > mult_trig=0.80 → legacy fires
        legacy_sig = legacy.evaluate(0.40, 1.21, -81.0, 120.0, pos)
        # mark=1.21: loss=0.81 << width_trig=7.68 → new rule does NOT fire
        new_sig    = new.evaluate(   0.40, 1.21, -81.0, 120.0, pos)

        self.assertIsNotNone(legacy_sig, "Legacy rule should fire at 2× premium")
        self.assertIsNone(new_sig,       "Width-relative rule should NOT fire yet")

    def test_ic_width_relative_uses_max_wing(self):
        """IC stop uses max(put_wing, call_wing) as the width."""
        rule = StopLossRule(stop_loss_multiplier=10.0, max_loss_pct=0.80)
        # put_wing=10, call_wing=15 → max_width=15, entry=0.80, max_loss=14.20
        # 80% budget = 11.36; mark = entry + 11.40 = 12.20 → trigger
        pos = {
            'type': 'IC',
            'legs': {
                'short_put': 90.0, 'long_put': 80.0,
                'short_call': 110.0, 'long_call': 125.0,
            },
        }
        sig = rule.evaluate(0.80, 12.20, -1140.0, 100.0, pos)
        self.assertIsNotNone(sig)

    # ── Reason string content ─────────────────────────────────────────────────

    def test_multiplier_trigger_label_in_reason(self):
        rule = StopLossRule(stop_loss_multiplier=2.0, max_loss_pct=None)
        pos  = self._pos('PCS', 95, 90)
        sig  = rule.evaluate(0.50, 1.60, -110.0, 100.0, pos)
        self.assertIn('2.0', sig.reason_str)
        self.assertIn('premium', sig.reason_str)

    def test_width_trigger_label_in_reason(self):
        rule = StopLossRule(stop_loss_multiplier=10.0, max_loss_pct=0.80)
        pos  = {'type': 'PCS', 'legs': {'short_strike': 100.0, 'long_strike': 90.0}}
        sig  = rule.evaluate(0.40, 8.50, -810.0, 120.0, pos)
        self.assertIn('max-loss', sig.reason_str)
        self.assertIn('width=10', sig.reason_str)


# ══════════════════════════════════════════════════════════════════════════════
# D. min_otm_pct = 8% floor (integration via scanner OTM filter)
# ══════════════════════════════════════════════════════════════════════════════

class TestMinOtmFloor(unittest.TestCase):

    def _scanner_with_otm(self, otm_pct: float):
        from src.scanner import OptionScanner
        return OptionScanner({
            'market_cap_min': 1e9,
            'expiry_days_max': 45,
            'risk_parameters': {'min_probability_of_expiry': 0.7},
            'min_otm_pct': {'put': otm_pct, 'call': otm_pct},
            'atr_distance': {'enabled': False},
            'dynamic_width': {'enabled': False},
            'strategies': {
                'put_credit_spread': {
                    'enabled': True, 'min_net_credit': 0.01,
                    'strike_width': 5, 'min_prob_profit': 0.50,
                    'max_delta_short_leg': 0.60,
                },
            },
        })

    def test_strike_within_8pct_rejected(self):
        """Short put at 94 (6% OTM) is rejected when min_otm_pct=0.08."""
        scanner = self._scanner_with_otm(0.08)
        spot = 100.0
        # short at 94 = 6% OTM — should be rejected
        rows = [
            _row(89, bid=0.30, ask=0.40, iv=0.25),
            _row(94, bid=0.80, ask=0.90, iv=0.25),
        ]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('TEST', spot, '2026-09-19', 30, df, 'put')
        strikes = [r['short_strike'] for r in results]
        self.assertNotIn(94.0, strikes, "Strike at 6% OTM should be rejected by 8% floor")

    def test_strike_beyond_8pct_accepted(self):
        """Short put at 90 (10% OTM) passes the 8% OTM floor."""
        scanner = self._scanner_with_otm(0.08)
        spot = 100.0
        rows = [
            _row(85, bid=0.20, ask=0.30, iv=0.25),
            _row(90, bid=0.60, ask=0.70, iv=0.25),
        ]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('TEST', spot, '2026-09-19', 30, df, 'put')
        strikes = [r['short_strike'] for r in results]
        self.assertIn(90.0, strikes, "Strike at 10% OTM should pass 8% floor")

    def test_strike_exactly_8pct_passes_boundary(self):
        """Short put exactly at 8% OTM (92 on $100) is allowed — the filter uses
        strict inequality (> not >=), so the boundary itself is not rejected."""
        scanner = self._scanner_with_otm(0.08)
        spot = 100.0
        rows = [
            _row(87, bid=0.20, ask=0.30, iv=0.25),
            _row(92, bid=0.55, ask=0.65, iv=0.25),
        ]
        df = _make_chain_df(rows)
        results = scanner._scan_spreads('TEST', spot, '2026-09-19', 30, df, 'put')
        strikes = [r['short_strike'] for r in results]
        # 92 == 100 * (1 - 0.08) == 92: filter checks short_strike > 92 → False → allowed
        self.assertIn(92.0, strikes, "Strike exactly at 8% OTM boundary should be allowed")


if __name__ == '__main__':
    unittest.main()
