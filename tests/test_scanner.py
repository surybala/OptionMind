import unittest
import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing src.scanner
sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())

# Add the parent directory (project root) to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner

class TestOptionScanner(unittest.TestCase):

    def setUp(self):
        self.config = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {
                'min_probability_of_expiry': 0.8
            },
            'strategies': {
                'covered_put': {
                    'min_premium': 0.1
                }
            }
        }
        self.scanner = OptionScanner(self.config)

    def test_probability_deep_otm_put(self):
        # Current price = 100, Strike = 50 (Put is deeply OTM)
        # Should be very high probability of expiring worthless (success)
        prob = self.scanner.get_probability_of_expiry(
            current_price=100, strike=50, iv=0.2, days_to_expiry=30, option_type='put'
        )
        self.assertGreater(prob, 0.95, "Deep OTM put should have high probability of expiring worthless")

    def test_probability_atm_put(self):
        # Current price = 100, Strike = 100 (ATM)
        # Should be around 0.5
        prob = self.scanner.get_probability_of_expiry(
            current_price=100, strike=100, iv=0.2, days_to_expiry=30, option_type='put'
        )
        self.assertAlmostEqual(prob, 0.5, delta=0.05, msg="ATM put should have approx 0.5 probability")

    def test_probability_itm_put(self):
        # Current price = 50, Strike = 100 (Deep ITM Put)
        # Should be low probability of expiring worthless (success)
        prob = self.scanner.get_probability_of_expiry(
            current_price=50, strike=100, iv=0.2, days_to_expiry=30, option_type='put'
        )
        self.assertLess(prob, 0.05, "Deep ITM put should have low probability of expiring worthless")

    def test_probability_zero_iv(self):
        # IV = 0
        prob = self.scanner.get_probability_of_expiry(
            current_price=100, strike=90, iv=0, days_to_expiry=10, option_type='put'
        )
        self.assertEqual(prob, 0.99, "Zero IV OTM put should be 0.99")
        
        prob_itm = self.scanner.get_probability_of_expiry(
            current_price=90, strike=100, iv=0, days_to_expiry=10, option_type='put'
        )
        self.assertEqual(prob_itm, 0.01, "Zero IV ITM put should be 0.01")

if __name__ == '__main__':
    unittest.main()
