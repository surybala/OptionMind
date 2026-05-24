"""
Tests for the three new option premium strategies:
  - Iron Condor   (_scan_iron_condor)
  - Short Strangle (_scan_strangle)
  - Covered Call   (_scan_covered_call)

Mock strategy
─────────────
Each scan method receives pandas-like DataFrames.  We replicate the same
lightweight mock pattern used throughout the rest of the test suite:
  mock_df.sort_values.return_value = mock_df
  mock_df.iterrows.return_value    = enumerate(list_of_dicts)

Row dicts carry the keys the scanner accesses: strike, lastPrice, bid, ask,
impliedVolatility.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock

sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas',   MagicMock())

# Mock alpaca-py modules so src.executor can be imported without the real SDK.
_alpaca_mock = MagicMock()
for _mod in ['alpaca', 'alpaca.trading', 'alpaca.trading.client',
             'alpaca.trading.enums', 'alpaca.trading.requests']:
    sys.modules.setdefault(_mod, _alpaca_mock)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(rows):
    """Minimal DataFrame mock that supports sort_values + iterrows."""
    mock_df = MagicMock()
    mock_df.sort_values.return_value = mock_df
    mock_df.iterrows.return_value = enumerate(rows)
    return mock_df


def _row(strike, last=1.0, bid=0.9, ask=1.1, iv=0.3):
    return {'strike': strike, 'lastPrice': last, 'bid': bid, 'ask': ask,
            'impliedVolatility': iv}


PRICE = 100.0   # underlying price used across tests
DAYS  = 30      # DTE used across tests
EXPIRY = '2026-04-30'


def _scanner(strategy_cfg):
    """Build an OptionScanner with only the requested strategy enabled."""
    config = {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {'min_probability_of_expiry': 0.8},
        'strategies': strategy_cfg,
    }
    return OptionScanner(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Iron Condor tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIronCondor(unittest.TestCase):

    IC_CFG = {
        'iron_condor': {
            'enabled': True,
            'min_net_credit': 0.40,
            'max_delta_short_leg': 0.25,
            'put_strike_width': 5,
            'call_strike_width': 5,
            'min_prob_profit': 0.70,
        }
    }

    def setUp(self):
        self.scanner = _scanner(self.IC_CFG)

    # -- helpers for put/call chain mocks ------------------------------------

    def _puts(self, rows):  return _make_df(rows)
    def _calls(self, rows): return _make_df(rows)

    def _standard_puts(self):
        # short_put=85 (OTM, deep enough for high prob), long_put=80
        return self._puts([
            _row(80,  last=0.20, bid=0.15, ask=0.25, iv=0.25),  # long leg (buy)
            _row(85,  last=0.70, bid=0.65, ask=0.75, iv=0.25),  # short leg (sell)
        ])

    def _standard_calls(self):
        # short_call=115 (OTM, deep enough for high prob), long_call=120
        return self._calls([
            _row(115, last=0.70, bid=0.65, ask=0.75, iv=0.25),  # short leg (sell)
            _row(120, last=0.20, bid=0.15, ask=0.25, iv=0.25),  # long leg (buy)
        ])

    # -- valid IC -------------------------------------------------------------

    def test_identifies_valid_iron_condor(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['strategy'], 'IC')
        self.assertEqual(r['symbol'],   'AAPL')

    def test_ic_short_put_and_call_strikes_are_correct(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            self.assertEqual(r['short_put'],  85)
            self.assertEqual(r['long_put'],   80)
            self.assertEqual(r['short_call'], 115)
            self.assertEqual(r['long_call'],  120)

    def test_ic_total_credit_is_sum_of_both_sides(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            # put_credit  = short_put_bid  - long_put_ask  = 0.65 - 0.25 = 0.40
            # call_credit = short_call_bid - long_call_ask = 0.65 - 0.25 = 0.40
            # total       = 0.80
            self.assertAlmostEqual(results[0]['premium'], 0.80, places=2)

    def test_ic_max_loss_is_max_width_minus_credit(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            expected = max(r['put_width'], r['call_width']) - r['premium']
            self.assertAlmostEqual(r['max_loss'], expected, places=2)

    def test_ic_prob_win_is_minimum_of_both_legs(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            self.assertLessEqual(r['prob_win'], r['prob_put'])
            self.assertLessEqual(r['prob_win'], r['prob_call'])

    def test_ic_score_is_yield_normalised(self):
        """Score = (credit / max_wing_width) × prob_win² — yield-normalised to remove ATM bias."""
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            max_width = max(r['put_width'], r['call_width'])
            expected  = round((r['premium'] / max_width) * r['prob_win'] ** 2, 4)
            self.assertAlmostEqual(r['score'], expected, places=4)

    def test_ic_roi_is_credit_over_max_loss(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            self.assertAlmostEqual(r['roi'], round(r['premium'] / r['max_loss'], 4), places=4)

    # -- filters --------------------------------------------------------------

    def test_rejects_when_long_put_leg_missing(self):
        # Only provide the short_put strike, no long_put at 80
        puts = self._puts([_row(85, bid=0.65, ask=0.75, iv=0.25)])
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS, puts, self._standard_calls()
        )
        self.assertEqual(len(results), 0)

    def test_rejects_when_long_call_leg_missing(self):
        # Only provide the short_call strike, no long_call at 120
        calls = self._calls([_row(115, bid=0.65, ask=0.75, iv=0.25)])
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS, self._standard_puts(), calls
        )
        self.assertEqual(len(results), 0)

    def test_rejects_when_total_credit_below_minimum(self):
        # Drive put_credit and call_credit very low so sum < 0.40
        puts  = self._puts([
            _row(80,  bid=0.14, ask=0.15, iv=0.25),   # long_put  ask = 0.15
            _row(85,  bid=0.25, ask=0.35, iv=0.25),   # short_put bid = 0.25 → credit = 0.10
        ])
        calls = self._calls([
            _row(115, bid=0.25, ask=0.35, iv=0.25),   # short_call bid = 0.25 → credit = 0.10
            _row(120, bid=0.14, ask=0.15, iv=0.25),   # long_call  ask = 0.15
        ])
        results = self.scanner._scan_iron_condor('AAPL', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(results), 0)

    def test_rejects_itm_put_short_leg(self):
        # short_put >= current_price → ITM, should be skipped
        puts = self._puts([
            _row(95,  bid=5.0, ask=5.2, iv=0.3),   # ITM short put
            _row(90,  bid=0.2, ask=0.3, iv=0.3),   # long put
        ])
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS, puts, self._standard_calls()
        )
        self.assertEqual(len(results), 0)

    def test_rejects_itm_call_short_leg(self):
        # short_call <= current_price → ITM, should be skipped
        calls = self._calls([
            _row(105, bid=5.0, ask=5.2, iv=0.3),   # ITM short call
            _row(110, bid=0.2, ask=0.3, iv=0.3),   # long call
        ])
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS, self._standard_puts(), calls
        )
        self.assertEqual(len(results), 0)

    def test_rejects_when_short_put_above_short_call(self):
        # Degenerate: wings overlap (short_put >= short_call)
        puts  = self._puts([
            _row(80,  bid=0.15, ask=0.25, iv=0.25),
            _row(115, bid=0.65, ask=0.75, iv=0.25),  # short_put = 115 == short_call
        ])
        calls = self._calls([
            _row(115, bid=0.65, ask=0.75, iv=0.25),
            _row(120, bid=0.15, ask=0.25, iv=0.25),
        ])
        results = self.scanner._scan_iron_condor('AAPL', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(results), 0)

    def test_returns_empty_when_no_valid_options(self):
        results = self.scanner._scan_iron_condor(
            'AAPL', PRICE, EXPIRY, DAYS, _make_df([]), _make_df([])
        )
        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════════
# Short Strangle tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShortStrangle(unittest.TestCase):

    STRANGLE_CFG = {
        'short_strangle': {
            'enabled': True,
            'min_net_credit': 0.50,
            'max_delta_short_leg': 0.20,
            'min_prob_profit': 0.75,
        }
    }

    def setUp(self):
        self.scanner = _scanner(self.STRANGLE_CFG)

    def _standard_puts(self):
        # Deep-OTM put: strike=80, high prob, solid bid
        return _make_df([_row(80, bid=0.60, ask=0.70, iv=0.25)])

    def _standard_calls(self):
        # Deep-OTM call: strike=120, high prob, solid bid
        return _make_df([_row(120, bid=0.60, ask=0.70, iv=0.25)])

    # -- valid strangle -------------------------------------------------------

    def test_identifies_valid_strangle(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['strategy'],   'STRANGLE')
        self.assertEqual(r['short_put'],  80)
        self.assertEqual(r['short_call'], 120)

    def test_strangle_total_credit_is_sum_of_bids(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            # put_bid = 0.60, call_bid = 0.60 → total = 1.20
            self.assertAlmostEqual(results[0]['premium'], 1.20, places=2)

    def test_strangle_max_loss_proxy_is_put_strike_times_100(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            self.assertEqual(results[0]['max_loss'], 80 * 100)

    def test_strangle_prob_win_is_min_of_both_legs(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            self.assertLessEqual(r['prob_win'], r['prob_put'])
            self.assertLessEqual(r['prob_win'], r['prob_call'])

    def test_strangle_score_is_credit_times_prob_win_squared(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS,
            self._standard_puts(), self._standard_calls()
        )
        if results:
            r = results[0]
            self.assertAlmostEqual(r['score'], round(r['premium'] * r['prob_win'] ** 2, 4), places=3)

    # -- filters --------------------------------------------------------------

    def test_rejects_when_total_credit_below_minimum(self):
        puts  = _make_df([_row(80,  bid=0.15, ask=0.25, iv=0.25)])  # 0.15 credit
        calls = _make_df([_row(120, bid=0.20, ask=0.30, iv=0.25)])  # 0.20 credit → total 0.35 < 0.50
        results = self.scanner._scan_strangle('AAPL', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(results), 0)

    def test_rejects_itm_put(self):
        puts = _make_df([_row(100, bid=5.0, ask=5.2, iv=0.3)])  # ATM/ITM
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS, puts, self._standard_calls()
        )
        self.assertEqual(len(results), 0)

    def test_rejects_itm_call(self):
        calls = _make_df([_row(100, bid=5.0, ask=5.2, iv=0.3)])  # ATM/ITM
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS, self._standard_puts(), calls
        )
        self.assertEqual(len(results), 0)

    def test_rejects_when_put_delta_too_high(self):
        # Near-the-money put with high IV → high delta → filtered
        puts = _make_df([_row(99, bid=3.0, ask=3.2, iv=2.0)])  # near ATM, high IV → low prob
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, 3, DAYS, puts, self._standard_calls()  # 3 DTE → even lower prob
        )
        self.assertEqual(len(results), 0)

    def test_rejects_when_put_above_call(self):
        puts  = _make_df([_row(120, bid=0.60, ask=0.70, iv=0.25)])
        calls = _make_df([_row(80,  bid=0.60, ask=0.70, iv=0.25)])
        results = self.scanner._scan_strangle('AAPL', PRICE, EXPIRY, DAYS, puts, calls)
        self.assertEqual(len(results), 0)

    def test_returns_empty_for_empty_chains(self):
        results = self.scanner._scan_strangle(
            'AAPL', PRICE, EXPIRY, DAYS, _make_df([]), _make_df([])
        )
        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════════
# Covered Call tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoveredCall(unittest.TestCase):

    CC_CFG = {
        'covered_call': {
            'enabled': True,
            'min_premium': 0.10,
        }
    }

    def setUp(self):
        self.scanner = _scanner(self.CC_CFG)

    def _make_calls(self, rows):
        return _make_df(rows)

    # -- valid CC -------------------------------------------------------------

    def test_identifies_valid_otm_covered_call(self):
        calls = self._make_calls([_row(110, bid=1.20, ask=1.30, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['strategy'],     'CC')
        self.assertEqual(r['symbol'],       'AAPL')
        self.assertEqual(r['short_strike'], 110)
        self.assertIsNone(r['long_strike'])

    def test_cc_max_loss_is_none(self):
        # max_loss depends on cost basis — scanner intentionally omits it
        calls = self._make_calls([_row(110, bid=1.20, ask=1.30, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        if results:
            self.assertIsNone(results[0]['max_loss'])

    def test_cc_premium_uses_bid_price(self):
        calls = self._make_calls([_row(110, bid=1.20, ask=1.40, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        if results:
            self.assertAlmostEqual(results[0]['premium'], 1.20)

    def test_cc_falls_back_to_last_price_when_bid_zero(self):
        calls = self._make_calls([_row(110, last=1.30, bid=0, ask=1.40, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        if results:
            self.assertAlmostEqual(results[0]['premium'], 1.30)

    def test_cc_roi_is_premium_over_current_price(self):
        calls = self._make_calls([_row(110, bid=1.20, ask=1.30, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        if results:
            self.assertAlmostEqual(results[0]['roi'], 1.20 / PRICE, places=6)

    def test_cc_score_is_premium_times_prob_win_squared(self):
        calls = self._make_calls([_row(110, bid=1.20, ask=1.30, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        if results:
            r = results[0]
            self.assertAlmostEqual(r['score'], round(r['premium'] * r['prob_win'] ** 2, 4), places=3)

    # -- filters --------------------------------------------------------------

    def test_rejects_atm_call(self):
        calls = self._make_calls([_row(100, bid=5.0, ask=5.2, iv=0.3)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        self.assertEqual(len(results), 0)

    def test_rejects_itm_call(self):
        calls = self._make_calls([_row(90, bid=12.0, ask=12.5, iv=0.3)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        self.assertEqual(len(results), 0)

    def test_rejects_below_min_premium(self):
        calls = self._make_calls([_row(110, bid=0.05, ask=0.10, iv=0.25)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        self.assertEqual(len(results), 0)

    def test_rejects_below_min_probability(self):
        # Near ATM with high IV and short DTE → low prob of expiring OTM (worthless)
        calls = self._make_calls([_row(101, bid=2.0, ask=2.2, iv=2.0)])
        results = self.scanner._scan_covered_call('AAPL', PRICE, 3, 3, calls)
        self.assertEqual(len(results), 0)

    def test_multiple_strikes_only_valid_pass(self):
        calls = self._make_calls([
            _row(110, bid=1.20, ask=1.30, iv=0.25),  # valid OTM
            _row(100, bid=5.00, ask=5.20, iv=0.30),  # ATM → rejected
            _row(115, bid=0.05, ask=0.10, iv=0.20),  # below min premium
        ])
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, calls)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['short_strike'], 110)

    def test_returns_empty_for_empty_chain(self):
        results = self.scanner._scan_covered_call('AAPL', PRICE, EXPIRY, DAYS, _make_df([]))
        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-strategy: scan_ticker dispatching and get_top_picks ordering
# ═══════════════════════════════════════════════════════════════════════════════

class TestAllStrategiesInTopPicks(unittest.TestCase):
    """Verify get_top_picks returns results from all 6 strategies, ranked by score."""

    def test_top_picks_ranks_all_strategies_by_score(self):
        scanner = OptionScanner({
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {'min_probability_of_expiry': 0.8},
            'strategies': {},
        })

        # Inject mock scan_ticker that returns one result per strategy
        strategies = ['CSP', 'PCS', 'CCS', 'IC', 'STRANGLE', 'CC']
        scanner.scan_ticker = MagicMock(side_effect=lambda sym: [
            {'strategy': s, 'symbol': sym, 'score': float(i)}
            for i, s in enumerate(strategies)
        ])

        picks = scanner.get_top_picks(['AAPL'], n=10)
        # Should be sorted descending by score
        scores = [p['score'] for p in picks]
        self.assertEqual(scores, sorted(scores, reverse=True))
        # All 6 strategy types present
        returned_strategies = {p['strategy'] for p in picks}
        self.assertEqual(returned_strategies, set(strategies))

    def test_top_picks_respects_n_across_all_strategy_types(self):
        scanner = OptionScanner({
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {'min_probability_of_expiry': 0.8},
            'strategies': {},
        })
        scanner.scan_ticker = MagicMock(return_value=[
            {'strategy': s, 'score': 1.0} for s in ['IC', 'STRANGLE', 'CC', 'CSP']
        ])
        picks = scanner.get_top_picks(['AAPL', 'MSFT'], n=5)
        self.assertEqual(len(picks), 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Executor: new dry-run methods
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutorNewStrategies(unittest.TestCase):
    """Dry-run and login tests for the three new executor methods."""

    def setUp(self):
        from unittest.mock import patch
        self._patch = patch('src.executor.load_config', return_value={
            'alpaca': {'api_key': 'test_key', 'api_secret': 'test_secret', 'paper': True}
        })
        self._patch.start()
        from src.executor import AlpacaExecutor
        self.executor = AlpacaExecutor()
        self._patch.stop()

    # Iron Condor
    def test_ic_dry_run_returns_dry_run_id(self):
        result = self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_ic_dry_run_does_not_login(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, dry_run=True
        )
        self.assertFalse(self.executor.is_logged_in)

    def test_ic_live_mode_fails_gracefully_without_login(self):
        from unittest.mock import patch
        with patch('src.executor.load_config', return_value={
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True}
        }):
            from src.executor import AlpacaExecutor
            ex = AlpacaExecutor()
        result = ex.execute_sell_iron_condor('AAPL', '2026-04-30', 85, 80, 115, 120, dry_run=False)
        self.assertIsNone(result)

    # Short Strangle
    def test_strangle_dry_run_returns_dry_run_id(self):
        result = self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_strangle_dry_run_does_not_login(self):
        self.executor.execute_sell_strangle('AAPL', '2026-04-30', 80, 120, dry_run=True)
        self.assertFalse(self.executor.is_logged_in)

    def test_strangle_live_mode_fails_gracefully_without_login(self):
        from unittest.mock import patch
        with patch('src.executor.load_config', return_value={
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True}
        }):
            from src.executor import AlpacaExecutor
            ex = AlpacaExecutor()
        result = ex.execute_sell_strangle('AAPL', '2026-04-30', 80, 120, dry_run=False)
        self.assertIsNone(result)

    # Covered Call
    def test_cc_dry_run_returns_dry_run_id(self):
        result = self.executor.execute_sell_covered_call(
            'AAPL', '2026-04-30', 115, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_cc_dry_run_does_not_login(self):
        self.executor.execute_sell_covered_call('AAPL', '2026-04-30', 115, dry_run=True)
        self.assertFalse(self.executor.is_logged_in)

    def test_cc_live_mode_fails_gracefully_without_login(self):
        from unittest.mock import patch
        with patch('src.executor.load_config', return_value={
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True}
        }):
            from src.executor import AlpacaExecutor
            ex = AlpacaExecutor()
        result = ex.execute_sell_covered_call('AAPL', '2026-04-30', 115, dry_run=False)
        self.assertIsNone(result)

    # Amount parameter forwarded
    def test_ic_amount_parameter_accepted(self):
        result = self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, amount=3, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_strangle_amount_parameter_accepted(self):
        result = self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, amount=2, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_cc_amount_parameter_accepted(self):
        result = self.executor.execute_sell_covered_call(
            'AAPL', '2026-04-30', 115, amount=5, dry_run=True
        )
        self.assertEqual(result, 'DRY_RUN_ID')


if __name__ == '__main__':
    unittest.main()
