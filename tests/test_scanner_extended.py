import unittest
import sys
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner


def _base_config(**overrides):
    cfg = {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {'min_probability_of_expiry': 0.8},
        'strategies': {
            'covered_put': {'enabled': True, 'min_premium': 0.10},
            'put_credit_spread': {'enabled': False},
            'call_credit_spread': {'enabled': False},
        },
    }
    cfg.update(overrides)
    return cfg


class TestCallProbability(unittest.TestCase):
    """get_probability_of_expiry for call options (complementary to test_scanner.py's put tests)."""

    def setUp(self):
        self.scanner = OptionScanner(_base_config())

    def test_deep_otm_call_has_high_prob(self):
        # Call at 200 when price is 100: very likely to expire worthless
        prob = self.scanner.get_probability_of_expiry(100, 200, 0.2, 30, 'call')
        self.assertGreater(prob, 0.95)

    def test_atm_call_has_near_half_prob(self):
        prob = self.scanner.get_probability_of_expiry(100, 100, 0.2, 30, 'call')
        self.assertAlmostEqual(prob, 0.5, delta=0.05)

    def test_deep_itm_call_has_low_prob(self):
        # Call at 50 when price is 100: almost certain to be exercised (ITM)
        prob = self.scanner.get_probability_of_expiry(100, 50, 0.2, 30, 'call')
        self.assertLess(prob, 0.05)

    def test_zero_iv_call_otm_returns_099(self):
        prob = self.scanner.get_probability_of_expiry(100, 110, 0, 10, 'call')
        self.assertEqual(prob, 0.99)

    def test_zero_iv_call_itm_returns_001(self):
        prob = self.scanner.get_probability_of_expiry(100, 90, 0, 10, 'call')
        self.assertEqual(prob, 0.01)


class TestScanTicker(unittest.TestCase):
    """scan_ticker() wires together market cap check, expiry window, and strategy dispatching.

    We patch 'src.scanner.yf' directly — that's the name bound inside scanner.py at import
    time, so patching sys.modules['yfinance'] would have no effect on it.
    """

    def setUp(self):
        self.scanner = OptionScanner(_base_config())

    @staticmethod
    def _make_ticker_mock(market_cap, price, expiries=None):
        mock_ticker = MagicMock()
        mock_ticker.info = {'marketCap': market_cap, 'currentPrice': price}
        mock_ticker.options = expiries or []
        mock_ticker.option_chain.return_value = MagicMock(
            calls=MagicMock(), puts=MagicMock()
        )
        return mock_ticker

    @patch('src.scanner.yf')
    def test_skips_ticker_below_market_cap(self, mock_yf):
        mock_yf.Ticker.return_value = self._make_ticker_mock(5e8, 50.0)  # 500M < 1B min
        self.assertEqual(self.scanner.scan_ticker('TINY'), [])

    @patch('src.scanner.yf')
    def test_returns_empty_when_no_expiries_in_window(self, mock_yf):
        # Expiry is 30 days away but the scanner's window is only 14 days
        future = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        mock_yf.Ticker.return_value = self._make_ticker_mock(2e9, 100.0, [future])
        self.assertEqual(self.scanner.scan_ticker('AAPL'), [])

    @patch('src.scanner.yf')
    def test_calls_csp_scan_for_valid_ticker(self, mock_yf):
        future = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        mock_yf.Ticker.return_value = self._make_ticker_mock(2e9, 100.0, [future])

        self.scanner._scan_csp = MagicMock(return_value=[{'strategy': 'CSP', 'score': 1.0}])
        self.scanner._scan_spreads = MagicMock(return_value=[])

        results = self.scanner.scan_ticker('AAPL')
        self.scanner._scan_csp.assert_called_once()
        self.assertEqual(len(results), 1)

    @patch('src.scanner.yf')
    def test_returns_empty_on_yfinance_exception(self, mock_yf):
        mock_yf.Ticker.side_effect = Exception("network error")
        self.assertEqual(self.scanner.scan_ticker('BOOM'), [])

    @patch('src.scanner.yf')
    def test_falls_back_to_history_when_current_price_missing(self, mock_yf):
        """If currentPrice is absent from info, scan_ticker returns [] rather than crashing."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'marketCap': 5e9, 'currentPrice': None, 'regularMarketPrice': None}
        mock_ticker.options = []
        mock_yf.Ticker.return_value = mock_ticker
        result = self.scanner.scan_ticker('NOPRICE')
        self.assertIsInstance(result, list)


class TestGetTopPicks(unittest.TestCase):

    def setUp(self):
        self.scanner = OptionScanner(_base_config())

    def test_returns_top_n_sorted_by_score(self):
        self.scanner.scan_ticker = MagicMock(side_effect=lambda sym: [
            {'strategy': 'CSP', 'symbol': sym, 'score': float(i), 'prob_win': 0.9}
            for i in range(3)
        ])
        picks = self.scanner.get_top_picks(['A', 'B'], n=3)
        self.assertEqual(len(picks), 3)
        scores = [p['score'] for p in picks]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_respects_n_limit(self):
        self.scanner.scan_ticker = MagicMock(return_value=[
            {'strategy': 'CSP', 'symbol': 'X', 'score': float(i)} for i in range(10)
        ])
        picks = self.scanner.get_top_picks(['X'], n=4)
        self.assertEqual(len(picks), 4)

    def test_returns_empty_for_empty_ticker_list(self):
        picks = self.scanner.get_top_picks([], n=10)
        self.assertEqual(picks, [])

    def test_returns_empty_when_all_scans_empty(self):
        self.scanner.scan_ticker = MagicMock(return_value=[])
        picks = self.scanner.get_top_picks(['A', 'B', 'C'], n=10)
        self.assertEqual(picks, [])

    def test_aggregates_results_across_tickers(self):
        def side_effect(sym):
            return [{'strategy': 'CSP', 'symbol': sym, 'score': 1.0}]

        self.scanner.scan_ticker = MagicMock(side_effect=side_effect)
        picks = self.scanner.get_top_picks(['AAPL', 'MSFT'], n=10)
        symbols = {p['symbol'] for p in picks}
        self.assertEqual(symbols, {'AAPL', 'MSFT'})


class TestSpreadEdgeCases(unittest.TestCase):
    """Edge cases in _scan_spreads not covered by test_credit_spreads.py."""

    def setUp(self):
        config = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {'min_probability_of_expiry': 0.7},
            'strategies': {
                'put_credit_spread': {
                    'enabled': True,
                    'min_net_credit': 0.20,
                    'strike_width': 5,
                    'min_prob_profit': 0.60,
                    'max_delta_short_leg': 0.5,
                },
                'call_credit_spread': {
                    'enabled': True,
                    'min_net_credit': 0.20,
                    'strike_width': 5,
                    'min_prob_profit': 0.60,
                    'max_delta_short_leg': 0.5,
                },
            },
        }
        self.scanner = OptionScanner(config)

    def _make_df(self, rows):
        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.iterrows.return_value = enumerate(rows)
        col = MagicMock()
        col.tolist.return_value = [r['strike'] for r in rows]
        mock_df.__getitem__.return_value = col
        return mock_df

    def test_rejects_spread_when_long_leg_missing(self):
        # Short leg at 90 but no matching long leg at 85
        rows = [
            {'strike': 90, 'lastPrice': 1.0, 'bid': 1.0, 'ask': 1.1, 'impliedVolatility': 0.3},
        ]
        results = self.scanner._scan_spreads('TEST', 100, '2026-04-30', 20, self._make_df(rows), 'put')
        self.assertEqual(len(results), 0)

    def test_rejects_put_spread_when_short_leg_is_itm(self):
        rows = [
            {'strike': 100, 'lastPrice': 5.0, 'bid': 4.9, 'ask': 5.1, 'impliedVolatility': 0.3},
            {'strike': 105, 'lastPrice': 7.0, 'bid': 6.9, 'ask': 7.1, 'impliedVolatility': 0.3},
        ]
        results = self.scanner._scan_spreads('TEST', 100, '2026-04-30', 20, self._make_df(rows), 'put')
        self.assertEqual(len(results), 0)

    def test_rejects_call_spread_when_short_leg_is_itm(self):
        rows = [
            {'strike': 90, 'lastPrice': 12.0, 'bid': 11.9, 'ask': 12.1, 'impliedVolatility': 0.3},
            {'strike': 95, 'lastPrice': 8.0, 'bid': 7.9, 'ask': 8.1, 'impliedVolatility': 0.3},
        ]
        results = self.scanner._scan_spreads('TEST', 100, '2026-04-30', 20, self._make_df(rows), 'call')
        self.assertEqual(len(results), 0)

    def test_negative_net_credit_is_rejected(self):
        # Short bid < long ask → negative credit
        rows = [
            {'strike': 85, 'lastPrice': 2.0, 'bid': 1.9, 'ask': 2.1, 'impliedVolatility': 0.3},
            {'strike': 90, 'lastPrice': 1.0, 'bid': 0.5, 'ask': 1.1, 'impliedVolatility': 0.3},
        ]
        results = self.scanner._scan_spreads('TEST', 100, '2026-04-30', 20, self._make_df(rows), 'put')
        self.assertEqual(len(results), 0)

    def test_roi_and_delta_are_present_in_spread_result(self):
        rows = [
            {'strike': 85, 'lastPrice': 0.5, 'bid': 0.4, 'ask': 0.5, 'impliedVolatility': 0.5},
            {'strike': 90, 'lastPrice': 1.0, 'bid': 1.0, 'ask': 1.1, 'impliedVolatility': 0.5},
            {'strike': 95, 'lastPrice': 2.0, 'bid': 1.9, 'ask': 2.1, 'impliedVolatility': 0.5},
        ]
        mock_df = self._make_df(rows)
        results = self.scanner._scan_spreads('TEST', 100, '2026-04-30', 20, mock_df, 'put')
        for r in results:
            self.assertIn('roi', r)
            self.assertIn('estimated_delta', r)
            self.assertIn('score', r)
            self.assertIn('width', r)


if __name__ == '__main__':
    unittest.main()
