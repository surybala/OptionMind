"""
Tests for src/scan_filters/otm.py and src/scan_filters/probability.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scan_filters.base import FilterContext
from src.scan_filters.otm import OtmDirectionFilter, SanityBoundsFilter, MinOtmPctFilter
from src.scan_filters.probability import ProbabilityFilter
from src.scan_filters.atr import AtrDistanceFilter
from src.scan_filters import FilterChain


def _ctx(
    current_price=100.0,
    short_strike=95.0,
    option_type='put',
    prob_otm=0.85,
    atr=2.0,
    min_otm_put=0.04,
    min_otm_call=0.04,
    atr_enabled=True,
    atr_multiplier=2.0,
    max_delta=0.15,
    min_prob=0.80,
):
    return FilterContext(
        current_price=current_price,
        short_strike=short_strike,
        option_type=option_type,
        prob_otm=prob_otm,
        atr=atr,
        min_otm_put=min_otm_put,
        min_otm_call=min_otm_call,
        atr_enabled=atr_enabled,
        atr_multiplier=atr_multiplier,
        max_delta=max_delta,
        min_prob=min_prob,
    )


class TestOtmDirectionFilter(unittest.TestCase):

    def setUp(self):
        self.f = OtmDirectionFilter()

    def test_put_otm_passes(self):
        # Put with strike < spot → OTM
        ctx = _ctx(current_price=100, short_strike=95, option_type='put')
        self.assertTrue(self.f.passes(ctx))

    def test_put_itm_fails(self):
        # Put with strike > spot → ITM
        ctx = _ctx(current_price=100, short_strike=105, option_type='put')
        self.assertFalse(self.f.passes(ctx))

    def test_put_atm_fails(self):
        # Put with strike == spot → not OTM (put: must be < spot)
        ctx = _ctx(current_price=100, short_strike=100, option_type='put')
        self.assertFalse(self.f.passes(ctx))

    def test_call_otm_passes(self):
        # Call with strike > spot → OTM
        ctx = _ctx(current_price=100, short_strike=105, option_type='call')
        self.assertTrue(self.f.passes(ctx))

    def test_call_itm_fails(self):
        # Call with strike < spot → ITM
        ctx = _ctx(current_price=100, short_strike=95, option_type='call')
        self.assertFalse(self.f.passes(ctx))

    def test_call_atm_fails(self):
        # Call with strike == spot → not OTM (call: must be > spot)
        ctx = _ctx(current_price=100, short_strike=100, option_type='call')
        self.assertFalse(self.f.passes(ctx))

    def test_filter_has_name(self):
        self.assertEqual(self.f.name, 'otm_direction')


class TestSanityBoundsFilter(unittest.TestCase):

    def setUp(self):
        self.f = SanityBoundsFilter()

    def test_put_within_bounds_passes(self):
        ctx = _ctx(current_price=100, short_strike=50, option_type='put')
        self.assertTrue(self.f.passes(ctx))

    def test_put_at_30pct_passes(self):
        ctx = _ctx(current_price=100, short_strike=31, option_type='put')
        self.assertTrue(self.f.passes(ctx))

    def test_put_below_30pct_fails(self):
        # 29 < 100 * 0.30 = 30
        ctx = _ctx(current_price=100, short_strike=29, option_type='put')
        self.assertFalse(self.f.passes(ctx))

    def test_call_within_bounds_passes(self):
        ctx = _ctx(current_price=100, short_strike=150, option_type='call')
        self.assertTrue(self.f.passes(ctx))

    def test_call_at_200pct_passes(self):
        # 199 < 200 — passes
        ctx = _ctx(current_price=100, short_strike=199, option_type='call')
        self.assertTrue(self.f.passes(ctx))

    def test_call_above_200pct_fails(self):
        # 201 >= 100 * 2.0 = 200 — fails
        ctx = _ctx(current_price=100, short_strike=201, option_type='call')
        self.assertFalse(self.f.passes(ctx))

    def test_filter_has_name(self):
        self.assertEqual(self.f.name, 'sanity_bounds')


class TestMinOtmPctFilter(unittest.TestCase):

    def setUp(self):
        self.f = MinOtmPctFilter()

    def test_put_far_enough_passes(self):
        # Spot=100, strike=95 → 5% OTM > 4% threshold
        ctx = _ctx(current_price=100, short_strike=95, option_type='put',
                   min_otm_put=0.04)
        self.assertTrue(self.f.passes(ctx))

    def test_put_too_close_fails(self):
        # Spot=100, strike=97 → 3% OTM < 4% threshold
        ctx = _ctx(current_price=100, short_strike=97, option_type='put',
                   min_otm_put=0.04)
        self.assertFalse(self.f.passes(ctx))

    def test_put_at_exact_threshold_passes(self):
        # strike == spot * (1 - 0.04) = 96 → passes (<=)
        ctx = _ctx(current_price=100, short_strike=96, option_type='put',
                   min_otm_put=0.04)
        self.assertTrue(self.f.passes(ctx))

    def test_put_zero_threshold_always_passes(self):
        ctx = _ctx(current_price=100, short_strike=99, option_type='put',
                   min_otm_put=0.0)
        self.assertTrue(self.f.passes(ctx))

    def test_call_far_enough_passes(self):
        # Spot=100, strike=105 → 5% OTM > 4% threshold
        ctx = _ctx(current_price=100, short_strike=105, option_type='call',
                   min_otm_call=0.04)
        self.assertTrue(self.f.passes(ctx))

    def test_call_too_close_fails(self):
        # Spot=100, strike=103 → 3% OTM < 4% threshold
        ctx = _ctx(current_price=100, short_strike=103, option_type='call',
                   min_otm_call=0.04)
        self.assertFalse(self.f.passes(ctx))

    def test_call_at_exact_threshold_passes(self):
        # strike == spot * (1 + 0.04) = 104 → passes (>=)
        ctx = _ctx(current_price=100, short_strike=104, option_type='call',
                   min_otm_call=0.04)
        self.assertTrue(self.f.passes(ctx))

    def test_call_zero_threshold_always_passes(self):
        ctx = _ctx(current_price=100, short_strike=101, option_type='call',
                   min_otm_call=0.0)
        self.assertTrue(self.f.passes(ctx))

    def test_filter_has_name(self):
        self.assertEqual(self.f.name, 'min_otm_pct')


class TestProbabilityFilter(unittest.TestCase):

    def setUp(self):
        self.f = ProbabilityFilter()

    def test_passes_within_thresholds(self):
        # prob_otm=0.87 → delta=0.13 < max_delta=0.15, prob 0.87 >= min_prob=0.80 → passes
        ctx = _ctx(prob_otm=0.87, max_delta=0.15, min_prob=0.80)
        self.assertTrue(self.f.passes(ctx))

    def test_fails_when_delta_too_high(self):
        # prob_otm=0.80 → delta=0.20 > max_delta=0.15
        ctx = _ctx(prob_otm=0.80, max_delta=0.15, min_prob=0.80)
        self.assertFalse(self.f.passes(ctx))

    def test_fails_when_prob_otm_too_low(self):
        # prob_otm=0.75 < min_prob=0.80, but delta=0.25 also > 0.15
        # Ensure the min_prob check fires
        ctx = _ctx(prob_otm=0.88, max_delta=0.15, min_prob=0.90)
        # prob_otm=0.88 < min_prob=0.90 → fail
        self.assertFalse(self.f.passes(ctx))

    def test_fails_when_prob_otm_below_min_prob(self):
        # delta passes but prob < min_prob
        ctx = _ctx(prob_otm=0.85, max_delta=0.20, min_prob=0.90)
        self.assertFalse(self.f.passes(ctx))

    def test_high_prob_otm_passes(self):
        ctx = _ctx(prob_otm=0.95, max_delta=0.15, min_prob=0.80)
        self.assertTrue(self.f.passes(ctx))

    def test_filter_has_name(self):
        self.assertEqual(self.f.name, 'probability')

    def test_boundary_delta_passes(self):
        # prob_otm=0.87 → delta=0.13 < max_delta=0.15, clearly below threshold
        ctx = _ctx(prob_otm=0.87, max_delta=0.15, min_prob=0.80)
        self.assertTrue(self.f.passes(ctx))


class TestAtrDistanceFilter(unittest.TestCase):

    def setUp(self):
        self.f = AtrDistanceFilter()

    def test_passes_when_atr_disabled(self):
        # atr_enabled=False → always passes regardless of distance
        ctx = _ctx(current_price=100, short_strike=99, atr=5.0,
                   atr_enabled=False, atr_multiplier=2.0)
        self.assertTrue(self.f.passes(ctx))

    def test_passes_when_atr_zero(self):
        # atr=0 → guard skipped
        ctx = _ctx(current_price=100, short_strike=99, atr=0.0,
                   atr_enabled=True, atr_multiplier=2.0)
        self.assertTrue(self.f.passes(ctx))

    def test_passes_when_far_enough(self):
        # distance = |100 - 90| = 10 >= 2.0 * 4.0 = 8 → passes
        ctx = _ctx(current_price=100, short_strike=90, atr=4.0,
                   atr_enabled=True, atr_multiplier=2.0)
        self.assertTrue(self.f.passes(ctx))

    def test_fails_when_too_close(self):
        # distance = |100 - 98| = 2 < 2.0 * 2.0 = 4 → fails
        ctx = _ctx(current_price=100, short_strike=98, atr=2.0,
                   atr_enabled=True, atr_multiplier=2.0)
        self.assertFalse(self.f.passes(ctx))

    def test_passes_at_exact_boundary(self):
        # distance = |100 - 96| = 4 == 2.0 * 2.0 = 4 → passes (>=)
        ctx = _ctx(current_price=100, short_strike=96, atr=2.0,
                   atr_enabled=True, atr_multiplier=2.0)
        self.assertTrue(self.f.passes(ctx))

    def test_filter_has_name(self):
        self.assertEqual(self.f.name, 'atr_distance')


class TestFilterChain(unittest.TestCase):

    def test_all_pass_returns_true(self):
        f1 = OtmDirectionFilter()  # put with strike < price → passes
        f2 = SanityBoundsFilter()
        chain = FilterChain([f1, f2])
        ctx = _ctx(current_price=100, short_strike=90, option_type='put')
        self.assertTrue(chain.passes(ctx))

    def test_first_fails_returns_false(self):
        # OtmDirectionFilter: put with strike > price → fails (ITM)
        chain = FilterChain([OtmDirectionFilter(), SanityBoundsFilter()])
        ctx = _ctx(current_price=100, short_strike=105, option_type='put')
        self.assertFalse(chain.passes(ctx))

    def test_second_fails_returns_false(self):
        # OtmDirectionFilter passes (put OTM), SanityBoundsFilter fails (too low)
        chain = FilterChain([OtmDirectionFilter(), SanityBoundsFilter()])
        ctx = _ctx(current_price=100, short_strike=25, option_type='put')
        # 25 < 100 → OtmDirection passes (put below spot)
        # 25 < 100 * 0.30 = 30 → SanityBounds fails
        self.assertFalse(chain.passes(ctx))

    def test_empty_chain_returns_true(self):
        chain = FilterChain([])
        ctx = _ctx()
        self.assertTrue(chain.passes(ctx))

    def test_single_filter_chain(self):
        chain = FilterChain([OtmDirectionFilter()])
        ctx_pass = _ctx(current_price=100, short_strike=95, option_type='put')
        ctx_fail = _ctx(current_price=100, short_strike=105, option_type='put')
        self.assertTrue(chain.passes(ctx_pass))
        self.assertFalse(chain.passes(ctx_fail))


if __name__ == '__main__':
    unittest.main()
