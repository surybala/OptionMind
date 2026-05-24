import unittest
import sys
import os
from unittest.mock import MagicMock

sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner


def _make_puts_df(rows):
    mock_df = MagicMock()
    mock_df.iterrows.return_value = enumerate(rows)
    return mock_df


class TestCSPScanning(unittest.TestCase):

    def setUp(self):
        self.config = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {'min_probability_of_expiry': 0.8},
            'strategies': {
                'covered_put': {'enabled': True, 'min_premium': 0.10},
            },
        }
        self.scanner = OptionScanner(self.config)

    # ── Valid CSP ─────────────────────────────────────────────────────────────

    def test_identifies_valid_otm_csp(self):
        rows = [{'strike': 80.0, 'lastPrice': 1.50, 'impliedVolatility': 0.2}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['strategy'], 'CSP')
        self.assertEqual(r['symbol'], 'AAPL')
        self.assertEqual(r['short_strike'], 80.0)
        self.assertIsNone(r['long_strike'])
        self.assertAlmostEqual(r['premium'], 1.50)

    def test_score_equals_premium_times_prob(self):
        rows = [{'strike': 80.0, 'lastPrice': 1.0, 'impliedVolatility': 0.2}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        if results:
            r = results[0]
            self.assertAlmostEqual(r['score'], round(r['premium'] * r['prob_win'], 4), places=4)

    def test_roi_equals_premium_over_strike(self):
        rows = [{'strike': 80.0, 'lastPrice': 1.0, 'impliedVolatility': 0.2}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        if results:
            r = results[0]
            self.assertAlmostEqual(r['roi'], 1.0 / 80.0, places=6)

    def test_max_loss_equals_strike_times_100(self):
        rows = [{'strike': 80.0, 'lastPrice': 1.0, 'impliedVolatility': 0.2}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        if results:
            self.assertEqual(results[0]['max_loss'], 80.0 * 100)

    # ── Filters: invalid legs ─────────────────────────────────────────────────

    def test_rejects_atm_put(self):
        # strike == current_price is not OTM for a put
        rows = [{'strike': 100.0, 'lastPrice': 5.0, 'impliedVolatility': 0.3}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 14, _make_puts_df(rows))
        self.assertEqual(len(results), 0)

    def test_rejects_itm_put(self):
        rows = [{'strike': 110.0, 'lastPrice': 12.0, 'impliedVolatility': 0.3}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 14, _make_puts_df(rows))
        self.assertEqual(len(results), 0)

    def test_rejects_below_min_premium(self):
        rows = [{'strike': 80.0, 'lastPrice': 0.05, 'impliedVolatility': 0.2}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        self.assertEqual(len(results), 0)

    def test_rejects_below_min_probability(self):
        # Near-the-money with high IV and short time → low prob of expiring OTM
        # strike=99, price=100, IV=2.0, days=3 → prob ≈ 0.48 < 0.8
        rows = [{'strike': 99.0, 'lastPrice': 2.0, 'impliedVolatility': 2.0}]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-03', 3, _make_puts_df(rows))
        self.assertEqual(len(results), 0)

    def test_multiple_rows_only_valid_pass(self):
        rows = [
            {'strike': 80.0, 'lastPrice': 1.0, 'impliedVolatility': 0.2},   # valid
            {'strike': 100.0, 'lastPrice': 5.0, 'impliedVolatility': 0.3},  # ITM/ATM → rejected
            {'strike': 75.0, 'lastPrice': 0.02, 'impliedVolatility': 0.2},  # below min premium
        ]
        results = self.scanner._scan_csp('AAPL', 100.0, '2026-04-30', 30, _make_puts_df(rows))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['short_strike'], 80.0)


if __name__ == '__main__':
    unittest.main()
