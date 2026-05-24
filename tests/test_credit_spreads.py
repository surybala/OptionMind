import unittest
import sys
import os
from unittest.mock import MagicMock

# 1. Mock external dependencies BEFORE importing the module under test
sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())

# 2. Add src to path so we can import scanner
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 3. Import the module under test
# Since pandas is mocked, the import inside scanner.py won't fail
from src.scanner import OptionScanner

class TestCreditSpreads(unittest.TestCase):

    def setUp(self):
        # Configuration for testing
        self.config = {
            'market_cap_min': 1e9,
            'expiry_days_max': 14,
            'risk_parameters': {
                'min_probability_of_expiry': 0.7
            },
            'strategies': {
                'put_credit_spread': {
                    'enabled': True,
                    'min_net_credit': 0.20,
                    'strike_width': 5,
                    'min_prob_profit': 0.60, # Lowering for test
                    'max_delta_short_leg': 0.5 
                },
                'call_credit_spread': {
                    'enabled': True,
                    'min_net_credit': 0.20,
                    'strike_width': 5,
                    'min_prob_profit': 0.60,
                    'max_delta_short_leg': 0.5 
                }
            }
        }
        self.scanner = OptionScanner(self.config)

    def _create_mock_df(self, rows):
        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.iterrows.return_value = enumerate(rows)
        
        # Mock column access: df['strike'] -> returns a list-like object
        mock_col = MagicMock()
        mock_col.tolist.return_value = [r['strike'] for r in rows]
        
        def getitem(name):
            if name == 'strike':
                return mock_col
            return MagicMock() # Fallback
            
        mock_df.__getitem__.side_effect = getitem
        return mock_df

    def test_put_credit_spread_identification(self):
        # Scenario:
        # Stock Price: 100
        # Short Put: Strike 90 (OTM), Bid 1.00
        # Long Put: Strike 85 (OTM), Ask 0.50
        # Net Credit: 1.00 - 0.50 = 0.50
        
        rows = [
            {'strike': 80, 'lastPrice': 0.2, 'bid': 0.1, 'ask': 0.3, 'impliedVolatility': 0.5},
            {'strike': 85, 'lastPrice': 0.5, 'bid': 0.4, 'ask': 0.5, 'impliedVolatility': 0.5}, # Long Leg
            {'strike': 90, 'lastPrice': 1.0, 'bid': 1.0, 'ask': 1.1, 'impliedVolatility': 0.5}, # Short Leg
            {'strike': 95, 'lastPrice': 2.0, 'bid': 1.9, 'ask': 2.1, 'impliedVolatility': 0.5},
            {'strike': 100, 'lastPrice': 5.0, 'bid': 4.9, 'ask': 5.1, 'impliedVolatility': 0.5}
        ]
        
        mock_df = self._create_mock_df(rows)
        
        # We invoke the private method directly
        results = self.scanner._scan_spreads('TEST', 100, '2026-03-20', 20, mock_df, 'put')
        
        found = False
        for r in results:
            if r['short_strike'] == 90 and r['long_strike'] == 85:
                found = True
                self.assertEqual(r['strategy'], 'PCS')
                self.assertAlmostEqual(r['premium'], 0.50, delta=0.01)
                self.assertEqual(r['width'], 5)
                self.assertAlmostEqual(r['max_loss'], 4.50, delta=0.01)
                self.assertIn('score', r)
        
        self.assertTrue(found, "Failed to identify valid Put Credit Spread 90/85")

    def test_call_credit_spread_identification(self):
        # Scenario:
        # Stock Price: 100
        # Short Call: Strike 110 (OTM), Bid 1.00
        # Long Call: Strike 115 (OTM), Ask 0.50
        # Net Credit: 0.50
        
        rows = [
            {'strike': 100, 'lastPrice': 5.0, 'bid': 4.9, 'ask': 5.1, 'impliedVolatility': 0.5},
            {'strike': 105, 'lastPrice': 2.0, 'bid': 1.9, 'ask': 2.1, 'impliedVolatility': 0.5},
            {'strike': 110, 'lastPrice': 1.0, 'bid': 1.0, 'ask': 1.1, 'impliedVolatility': 0.5}, # Short Leg
            {'strike': 115, 'lastPrice': 0.5, 'bid': 0.4, 'ask': 0.5, 'impliedVolatility': 0.5}, # Long Leg
            {'strike': 120, 'lastPrice': 0.2, 'bid': 0.1, 'ask': 0.3, 'impliedVolatility': 0.5}
        ]
        
        mock_df = self._create_mock_df(rows)
        
        results = self.scanner._scan_spreads('TEST', 100, '2026-03-20', 20, mock_df, 'call')
        
        found = False
        for r in results:
            if r['short_strike'] == 110 and r['long_strike'] == 115:
                found = True
                self.assertEqual(r['strategy'], 'CCS')
                self.assertAlmostEqual(r['premium'], 0.50, delta=0.01)
                
        self.assertTrue(found, "Failed to identify valid Call Credit Spread 110/115")

    def test_spread_filtering(self):
        # Scenario: Credit is too low
        # Short Put 90 Bid: 1.0
        # Long Put 85 Ask: 0.9
        # Net: 0.10 < 0.20 (min)
        
        rows = [
            {'strike': 85, 'lastPrice': 0.9, 'bid': 0.8, 'ask': 0.9, 'impliedVolatility': 0.5},
            {'strike': 90, 'lastPrice': 1.0, 'bid': 1.0, 'ask': 1.1, 'impliedVolatility': 0.5}
        ]
        
        mock_df = self._create_mock_df(rows)
        
        results = self.scanner._scan_spreads('TEST', 100, '2026-03-20', 20, mock_df, 'put')
        
        self.assertEqual(len(results), 0, "Should reject spread with insufficient credit")

if __name__ == '__main__':
    unittest.main()
