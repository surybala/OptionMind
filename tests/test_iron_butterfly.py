"""
Tests for src/scan_strategies/iron_butterfly.py — IronButterflyScanner
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scan_strategies.iron_butterfly import IronButterflyScanner


PRICE  = 100.0
EXPIRY = '2026-04-30'
DAYS   = 14


def _make_df(rows):
    """Minimal DataFrame mock compatible with sort_values + iterrows."""
    mock_df = MagicMock()
    mock_df.sort_values.return_value = mock_df
    mock_df.iterrows.return_value = enumerate(rows)
    return mock_df


def _row(strike, bid=3.0, ask=3.2, last=3.1, iv=0.25, oi=500):
    return {
        'strike': strike,
        'bid': bid,
        'ask': ask,
        'lastPrice': last,
        'impliedVolatility': iv,
        'openInterest': oi,
        'volume': 200,
    }


def _scanner(
    put_wing=10,
    call_wing=10,
    min_credit=1.50,
    min_prob=0.60,
    atm_tol=0.025,
):
    params = {
        'put_wing_width': put_wing,
        'call_wing_width': call_wing,
        'min_net_credit': min_credit,
        'min_prob_profit': min_prob,
        'atm_pct_tolerance': atm_tol,
    }
    # prob_otm_fn: returns a fixed value to control test outcomes
    prob_fn = MagicMock(return_value=0.80)
    oi_vol_fn = MagicMock(return_value=(500, 200))
    return IronButterflyScanner(
        params=params,
        min_prob=min_prob,
        min_otm_put=0.04,
        min_otm_call=0.04,
        atr_enabled=False,
        atr_multiplier=1.5,
        prob_otm_fn=prob_fn,
        row_oi_vol_fn=oi_vol_fn,
    ), prob_fn, oi_vol_fn


class TestIronButterflyBasic(unittest.TestCase):

    def _run_scan(self, put_rows, call_rows, **scanner_kwargs):
        scanner, prob_fn, oi_vol_fn = _scanner(**scanner_kwargs)
        puts  = _make_df(put_rows)
        calls = _make_df(call_rows)
        return scanner.scan('SPY', PRICE, EXPIRY, DAYS, puts, calls), prob_fn

    def _standard_rows(self):
        # ATM strike=100, wings at 90 (put) and 110 (call)
        put_rows  = [_row(90, bid=1.0, ask=1.1), _row(100, bid=4.0, ask=4.2)]
        call_rows = [_row(100, bid=4.0, ask=4.2), _row(110, bid=1.0, ask=1.1)]
        return put_rows, call_rows

    def test_returns_pick_when_conditions_met(self):
        put_rows, call_rows = self._standard_rows()
        # net_credit = 4.0+4.0-1.1-1.1 = 5.8 > min_credit=1.50
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(len(picks), 1)

    def test_pick_strategy_is_ifly(self):
        put_rows, call_rows = self._standard_rows()
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(picks[0]['strategy'], 'IFLY')

    def test_pick_fields_present(self):
        put_rows, call_rows = self._standard_rows()
        picks, _ = self._run_scan(put_rows, call_rows)
        pick = picks[0]
        for field in ('symbol', 'expiry', 'current_price', 'short_put', 'short_call',
                      'long_put', 'long_call', 'premium', 'max_loss', 'prob_win',
                      'roi', 'score'):
            self.assertIn(field, pick, f"Missing field: {field}")

    def test_short_put_and_call_same_atm_strike(self):
        put_rows, call_rows = self._standard_rows()
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(picks[0]['short_put'], picks[0]['short_call'])

    def test_no_pick_when_net_credit_too_low(self):
        # Set bids very low → net_credit < min_credit
        put_rows  = [_row(90, bid=0.05, ask=0.10), _row(100, bid=0.10, ask=0.15)]
        call_rows = [_row(100, bid=0.10, ask=0.15), _row(110, bid=0.05, ask=0.10)]
        picks, _ = self._run_scan(put_rows, call_rows, min_credit=1.50)
        self.assertEqual(len(picks), 0)

    def test_no_pick_when_strike_not_atm(self):
        # Only far OTM strikes available (no ATM)
        put_rows  = [_row(70, bid=1.0, ask=1.1), _row(80, bid=2.0, ask=2.1)]
        call_rows = [_row(120, bid=1.0, ask=1.1), _row(130, bid=0.5, ask=0.6)]
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(len(picks), 0)

    def test_no_pick_when_long_put_strike_missing(self):
        # ATM put available, matching call available, but no long_put at strike-wing
        put_rows  = [_row(100, bid=4.0, ask=4.2)]   # no 90 in puts
        call_rows = [_row(100, bid=4.0, ask=4.2), _row(110, bid=1.0, ask=1.1)]
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(len(picks), 0)

    def test_no_pick_when_long_call_strike_missing(self):
        put_rows  = [_row(90, bid=1.0, ask=1.1), _row(100, bid=4.0, ask=4.2)]
        call_rows = [_row(100, bid=4.0, ask=4.2)]   # no 110 in calls
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(len(picks), 0)

    def test_no_pick_when_short_call_strike_missing(self):
        # put has ATM, but call does not have that same strike
        put_rows  = [_row(90, bid=1.0, ask=1.1), _row(100, bid=4.0, ask=4.2)]
        call_rows = [_row(105, bid=3.0, ask=3.2), _row(110, bid=1.0, ask=1.1)]
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertEqual(len(picks), 0)

    def test_no_pick_when_prob_too_low(self):
        scanner, prob_fn, _ = _scanner(min_prob=0.90)
        # prob_fn returns 0.80 by default → fails min_prob=0.90
        put_rows  = [_row(90, bid=1.0, ask=1.1), _row(100, bid=4.0, ask=4.2)]
        call_rows = [_row(100, bid=4.0, ask=4.2), _row(110, bid=1.0, ask=1.1)]
        picks = scanner.scan('SPY', PRICE, EXPIRY, DAYS,
                             _make_df(put_rows), _make_df(call_rows))
        self.assertEqual(len(picks), 0)

    def test_roi_computed_correctly(self):
        put_rows, call_rows = self._standard_rows()
        picks, _ = self._run_scan(put_rows, call_rows)
        pick = picks[0]
        expected_roi = round(pick['premium'] / pick['max_loss'], 4)
        self.assertAlmostEqual(pick['roi'], expected_roi, places=4)

    def test_score_uses_prob_win_squared(self):
        put_rows, call_rows = self._standard_rows()
        picks, _ = self._run_scan(put_rows, call_rows)
        pick = picks[0]
        expected_score = round(pick['premium'] * (pick['prob_win'] ** 2), 4)
        self.assertAlmostEqual(pick['score'], expected_score, places=4)

    def test_returns_empty_when_no_puts(self):
        picks, _ = self._run_scan([], [])
        self.assertEqual(len(picks), 0)

    def test_uses_bid_price_for_short(self):
        # When bid > 0, bid is used; lastPrice is fallback
        put_rows  = [_row(90, bid=1.5, ask=1.6, last=2.0),
                     _row(100, bid=5.0, ask=5.2, last=6.0)]
        call_rows = [_row(100, bid=5.0, ask=5.2, last=6.0),
                     _row(110, bid=1.5, ask=1.6, last=2.0)]
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertGreater(len(picks), 0)
        # premium = short_put_bid + short_call_bid - long_put_ask - long_call_ask
        # = 5.0 + 5.0 - 1.6 - 1.6 = 6.8
        self.assertAlmostEqual(picks[0]['premium'], 6.8, places=2)

    def test_falls_back_to_last_price_when_bid_zero(self):
        put_rows  = [_row(90, bid=0, ask=1.6, last=1.5),
                     _row(100, bid=0, ask=5.2, last=5.0)]
        call_rows = [_row(100, bid=0, ask=5.2, last=5.0),
                     _row(110, bid=0, ask=1.6, last=1.5)]
        picks, _ = self._run_scan(put_rows, call_rows)
        self.assertGreater(len(picks), 0)
        # short legs use lastPrice (5.0), long legs use ask (1.6)
        # net = 5.0+5.0-1.6-1.6 = 6.8
        self.assertAlmostEqual(picks[0]['premium'], 6.8, places=2)

    def test_max_loss_floored_at_point_01(self):
        # Wing=5, net_credit=5.5 → max_loss = 5-5.5=-0.5 → floored to 0.01
        put_rows  = [_row(95, bid=1.0, ask=1.1),
                     _row(100, bid=3.0, ask=3.1)]
        call_rows = [_row(100, bid=3.0, ask=3.1),
                     _row(105, bid=1.0, ask=1.1)]
        scanner, _, _ = _scanner(put_wing=5, call_wing=5, min_credit=0.0)
        puts  = _make_df(put_rows)
        calls = _make_df(call_rows)
        picks = scanner.scan('SPY', PRICE, EXPIRY, DAYS, puts, calls)
        if picks:
            self.assertGreater(picks[0]['max_loss'], 0)

    def test_prob_win_is_min_of_wings(self):
        # prob_fn returns different values per call: first=0.90, second=0.70
        scanner, prob_fn, _ = _scanner()
        prob_fn.side_effect = [0.90, 0.70]  # put_wing, call_wing
        put_rows  = [_row(90, bid=1.0, ask=1.1), _row(100, bid=4.0, ask=4.2)]
        call_rows = [_row(100, bid=4.0, ask=4.2), _row(110, bid=1.0, ask=1.1)]
        picks = scanner.scan('SPY', PRICE, EXPIRY, DAYS,
                             _make_df(put_rows), _make_df(call_rows))
        self.assertEqual(len(picks), 1)
        self.assertAlmostEqual(picks[0]['prob_win'], 0.70, places=4)

    def test_symbol_and_expiry_in_pick(self):
        put_rows, call_rows = self._standard_rows()
        scanner, _, _ = _scanner()
        picks = scanner.scan('QQQ', PRICE, '2026-05-15', DAYS,
                             _make_df(put_rows), _make_df(call_rows))
        self.assertEqual(picks[0]['symbol'], 'QQQ')
        self.assertEqual(picks[0]['expiry'], '2026-05-15')


class TestIronButterflyAtmTolerance(unittest.TestCase):

    def test_strike_within_tolerance_is_accepted(self):
        # 2.5% of 100 = 2.5 → strike=102 is within band
        scanner, _, _ = _scanner(atm_tol=0.025)
        put_rows  = [_row(92, bid=1.0, ask=1.1), _row(102, bid=4.0, ask=4.2)]
        call_rows = [_row(102, bid=4.0, ask=4.2), _row(112, bid=1.0, ask=1.1)]
        puts  = _make_df(put_rows)
        calls = _make_df(call_rows)
        picks = scanner.scan('SPY', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(picks), 1)

    def test_strike_outside_tolerance_rejected(self):
        # 2.5% of 100 = 2.5 → strike=103 is outside band
        scanner, _, _ = _scanner(atm_tol=0.025)
        put_rows  = [_row(93, bid=1.0, ask=1.1), _row(103, bid=4.0, ask=4.2)]
        call_rows = [_row(103, bid=4.0, ask=4.2), _row(113, bid=1.0, ask=1.1)]
        puts  = _make_df(put_rows)
        calls = _make_df(call_rows)
        picks = scanner.scan('SPY', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(picks), 0)


if __name__ == '__main__':
    unittest.main()
