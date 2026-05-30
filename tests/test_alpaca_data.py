"""
test_alpaca_data.py
===================
Unit tests for src/alpaca_data.py.

All alpaca-py SDK sub-modules are stubbed at module level so no real SDK
installation or network calls are required.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# ── Stub every alpaca-py module so imports inside src.alpaca_data succeed ──────
_alpaca_stub = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.data',
    'alpaca.data.historical',
    'alpaca.data.requests',
    'alpaca.data.timeframe',
]:
    sys.modules.setdefault(_mod, _alpaca_stub)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alpaca_data import (
    AlpacaDataClient,
    OptionChain,
    _snapshot_to_row,
    make_alpaca_data_client,
)
from src.osi import parse_osi as _parse_osi


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_client() -> AlpacaDataClient:
    """
    Return an AlpacaDataClient with fresh, independent MagicMock internal clients.

    Without this override, every AlpacaDataClient() call returns the *same*
    StockHistoricalDataClient/OptionHistoricalDataClient mock instance (because
    all constructor calls use the shared _alpaca_stub singleton).  Sharing mocks
    across tests causes call-count assertions (assert_not_called, etc.) and
    side_effect state to bleed from one test into the next.

    By replacing _stock and _option with fresh MagicMock() instances here, every
    test starts with clean, independent state.
    """
    client = AlpacaDataClient('test_key', 'test_secret')
    client._stock = MagicMock()
    client._option = MagicMock()
    return client


def _make_snap(bid=1.0, ask=2.0, last=1.5, iv=0.3, oi=None) -> MagicMock:
    """Build a minimal OptionsSnapshot mock."""
    snap = MagicMock()
    snap.latest_quote.bid_price = bid
    snap.latest_quote.ask_price = ask
    snap.latest_trade.price = last
    snap.implied_volatility = iv
    snap.open_interest = oi
    return snap


def _make_chain_mock(data: dict) -> MagicMock:
    """Wrap a symbol→snapshot dict so AlpacaDataClient sees it as a chain response."""
    m = MagicMock()
    m.data = data
    return m


# ══════════════════════════════════════════════════════════════════════════════
# _parse_osi
# ══════════════════════════════════════════════════════════════════════════════

class TestParseOsi(unittest.TestCase):

    def test_valid_call_symbol(self):
        result = _parse_osi('AAPL240119C00200000')
        self.assertIsNotNone(result)
        self.assertEqual(result.option_type, 'call')
        self.assertAlmostEqual(result.strike, 200.0)
        self.assertEqual(result.expiration.isoformat(), '2024-01-19')
        self.assertEqual(result.underlying, 'AAPL')

    def test_valid_put_symbol(self):
        result = _parse_osi('SPY250117P00400000')
        self.assertEqual(result.option_type, 'put')
        self.assertAlmostEqual(result.strike, 400.0)
        self.assertEqual(result.expiration.isoformat(), '2025-01-17')

    def test_fractional_strike(self):
        # 185500 / 1000 = 185.5
        result = _parse_osi('AAPL240119C00185500')
        self.assertEqual(result.option_type, 'call')
        self.assertAlmostEqual(result.strike, 185.5)
        self.assertEqual(result.expiration.isoformat(), '2024-01-19')

    def test_multi_char_root(self):
        result = _parse_osi('TSLA250117C00500000')
        self.assertEqual(result.option_type, 'call')
        self.assertAlmostEqual(result.strike, 500.0)
        self.assertEqual(result.expiration.isoformat(), '2025-01-17')

    def test_year_2000_epoch(self):
        # yy=00 → 2000
        result = _parse_osi('AAPL000119C00100000')
        self.assertEqual(result.expiration.year, 2000)

    def test_year_2069_upper_boundary(self):
        # yy=69 → 2069 (not 1969)
        result = _parse_osi('SPY690117C00100000')
        self.assertEqual(result.expiration.year, 2069)

    def test_year_1970_lower_boundary(self):
        # yy=70 → 1970 (not 2070)
        result = _parse_osi('SPY700117C00100000')
        self.assertEqual(result.expiration.year, 1970)

    def test_invalid_symbol_returns_none(self):
        self.assertIsNone(_parse_osi('NOT_VALID'))
        self.assertIsNone(_parse_osi(''))
        self.assertIsNone(_parse_osi('AAPL'))

    def test_lowercase_not_matched(self):
        self.assertIsNone(_parse_osi('aapl240119c00200000'))

    def test_zero_strike(self):
        # Strike 00000000 = 0.0; still a valid parse
        result = _parse_osi('AAPL240119C00000000')
        self.assertEqual(result.option_type, 'call')
        self.assertAlmostEqual(result.strike, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# _snapshot_to_row
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshotToRow(unittest.TestCase):

    def test_normal_call_snapshot(self):
        snap = _make_snap(bid=1.0, ask=2.0, last=1.5, iv=0.3, oi=100)
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertIsNotNone(row)
        self.assertEqual(row['_opt_type'], 'call')
        self.assertAlmostEqual(row['strike'], 200.0)
        self.assertAlmostEqual(row['bid'], 1.0)
        self.assertAlmostEqual(row['ask'], 2.0)
        self.assertAlmostEqual(row['lastPrice'], 1.5)
        self.assertAlmostEqual(row['impliedVolatility'], 0.3)
        self.assertEqual(row['openInterest'], 100)

    def test_put_snapshot_sets_opt_type(self):
        snap = _make_snap()
        row = _snapshot_to_row('SPY250117P00400000', snap)
        self.assertEqual(row['_opt_type'], 'put')

    def test_zero_last_price_uses_midpoint(self):
        """When lastPrice is 0, fall back to (bid + ask) / 2."""
        snap = _make_snap(bid=1.0, ask=3.0, last=0.0)
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertAlmostEqual(row['lastPrice'], 2.0)

    def test_non_zero_last_price_not_overridden(self):
        snap = _make_snap(bid=1.0, ask=3.0, last=1.8)
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertAlmostEqual(row['lastPrice'], 1.8)

    def test_oi_present_used_directly(self):
        snap = _make_snap(oi=500)
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertEqual(row['openInterest'], 500)

    def test_oi_none_yields_sentinel_minus_one(self):
        """Alpaca OptionsSnapshot omits OI → _snapshot_to_row returns -1."""
        snap = MagicMock(spec=['latest_quote', 'latest_trade', 'implied_volatility'])
        snap.latest_quote.bid_price = 1.0
        snap.latest_quote.ask_price = 2.0
        snap.latest_trade.price = 1.5
        snap.implied_volatility = 0.3
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertEqual(row['openInterest'], -1)

    def test_invalid_osi_symbol_returns_none(self):
        row = _snapshot_to_row('INVALID_SYMBOL', MagicMock())
        self.assertIsNone(row)

    def test_no_quote_defaults_bid_ask_to_zero(self):
        snap = MagicMock()
        snap.latest_quote = None
        snap.latest_trade.price = 1.5
        snap.implied_volatility = 0.2
        snap.open_interest = None
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertAlmostEqual(row['bid'], 0.0)
        self.assertAlmostEqual(row['ask'], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_spot_prices / get_spot_price
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSpotPrices(unittest.TestCase):

    def test_returns_price_dict_for_multiple_symbols(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.return_value = {
            'AAPL': MagicMock(close=150.0),
            'MSFT': MagicMock(close=300.0),
        }
        result = client.get_spot_prices(['AAPL', 'MSFT'])
        self.assertAlmostEqual(result['AAPL'], 150.0)
        self.assertAlmostEqual(result['MSFT'], 300.0)

    def test_empty_symbol_list_returns_empty_dict(self):
        client = _make_client()
        self.assertEqual(client.get_spot_prices([]), {})
        client._stock.get_stock_latest_bar.assert_not_called()

    def test_sdk_exception_returns_empty_dict(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.side_effect = Exception('network error')
        self.assertEqual(client.get_spot_prices(['AAPL']), {})

    def test_bar_without_close_is_skipped(self):
        client = _make_client()
        bar = MagicMock(spec=[])  # no 'close' attribute
        client._stock.get_stock_latest_bar.return_value = {'AAPL': bar}
        result = client.get_spot_prices(['AAPL'])
        self.assertNotIn('AAPL', result)

    def test_get_spot_price_single_symbol(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.return_value = {'NVDA': MagicMock(close=800.0)}
        self.assertAlmostEqual(client.get_spot_price('NVDA'), 800.0)

    def test_get_spot_price_missing_symbol_returns_none(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.return_value = {}
        self.assertIsNone(client.get_spot_price('AAPL'))


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_bulk_history
# ══════════════════════════════════════════════════════════════════════════════

class TestGetBulkHistory(unittest.TestCase):

    def _bars(self, closes):
        return [MagicMock(close=c) for c in closes]

    def test_returns_series_per_symbol(self):
        client = _make_client()
        client._stock.get_stock_bars.return_value = MagicMock(
            data={'AAPL': self._bars(range(10, 20))}
        )
        result = client.get_bulk_history(['AAPL'], days=30)
        self.assertIn('AAPL', result)
        self.assertIsInstance(result['AAPL'], pd.Series)
        self.assertEqual(len(result['AAPL']), 10)

    def test_series_shorter_than_5_bars_is_excluded(self):
        client = _make_client()
        client._stock.get_stock_bars.return_value = MagicMock(
            data={'TINY': self._bars([1.0, 2.0, 3.0])}  # only 3 bars
        )
        result = client.get_bulk_history(['TINY'], days=30)
        self.assertNotIn('TINY', result)

    def test_multiple_symbols_returned(self):
        client = _make_client()
        client._stock.get_stock_bars.return_value = MagicMock(
            data={
                'AAPL': self._bars(range(10, 20)),
                'MSFT': self._bars(range(20, 30)),
            }
        )
        result = client.get_bulk_history(['AAPL', 'MSFT'], days=30)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)

    def test_empty_symbol_list_returns_empty_dict(self):
        client = _make_client()
        self.assertEqual(client.get_bulk_history([]), {})
        client._stock.get_stock_bars.assert_not_called()

    def test_sdk_exception_returns_empty_dict(self):
        client = _make_client()
        client._stock.get_stock_bars.side_effect = Exception('API error')
        result = client.get_bulk_history(['AAPL'], days=30)
        self.assertEqual(result, {})

    def test_series_close_values_match_bar_closes(self):
        client = _make_client()
        closes = [100.0, 101.5, 102.3, 99.8, 103.0, 104.2]
        client._stock.get_stock_bars.return_value = MagicMock(
            data={'AAPL': self._bars(closes)}
        )
        result = client.get_bulk_history(['AAPL'], days=30)
        for i, expected in enumerate(closes):
            self.assertAlmostEqual(result['AAPL'].iloc[i], expected)

    def test_large_batch_splits_into_chunks(self):
        """Requests for >100 symbols are split into ≤100-symbol chunks."""
        client = _make_client()
        client._stock.get_stock_bars.return_value = MagicMock(data={})
        syms = [f'SYM{i:03d}' for i in range(250)]
        client.get_bulk_history(syms, days=30)
        # 250 symbols → 3 chunks (100, 100, 50)
        self.assertEqual(client._stock.get_stock_bars.call_count, 3)


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_option_chain  (single expiry)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOptionChain(unittest.TestCase):

    def test_returns_option_chain_namedtuple(self):
        client = _make_client()
        data = {
            'AAPL240119C00200000': _make_snap(bid=1.0, ask=2.0, last=1.5, iv=0.3),
            'AAPL240119P00180000': _make_snap(bid=0.5, ask=1.0, last=0.7, iv=0.25),
        }
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertIsInstance(result, OptionChain)

    def test_puts_and_calls_are_dataframes(self):
        client = _make_client()
        data = {
            'AAPL240119C00200000': _make_snap(),
            'AAPL240119P00180000': _make_snap(),
        }
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertIsInstance(result.puts, pd.DataFrame)
        self.assertIsInstance(result.calls, pd.DataFrame)

    def test_calls_df_contains_correct_strike(self):
        client = _make_client()
        data = {'AAPL240119C00200000': _make_snap(bid=1.0, ask=2.0)}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertEqual(len(result.calls), 1)
        self.assertAlmostEqual(result.calls.iloc[0]['strike'], 200.0)

    def test_puts_df_contains_correct_strike(self):
        client = _make_client()
        data = {'AAPL240119P00180000': _make_snap()}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertEqual(len(result.puts), 1)
        self.assertAlmostEqual(result.puts.iloc[0]['strike'], 180.0)

    def test_required_columns_present(self):
        client = _make_client()
        data = {'AAPL240119C00200000': _make_snap()}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        for col in ['strike', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'openInterest']:
            self.assertIn(col, result.calls.columns)

    def test_empty_chain_returns_none(self):
        client = _make_client()
        client._option.get_option_chain.return_value = _make_chain_mock({})
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertIsNone(result)

    def test_sdk_exception_returns_none(self):
        client = _make_client()
        client._option.get_option_chain.side_effect = Exception('API error')
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertIsNone(result)

    def test_strikes_sorted_ascending(self):
        client = _make_client()
        data = {
            'AAPL240119P00200000': _make_snap(),   # higher strike
            'AAPL240119P00180000': _make_snap(),   # lower strike
        }
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        strikes = result.puts['strike'].tolist()
        self.assertEqual(strikes, sorted(strikes))

    def test_oi_sentinel_preserved_in_dataframe(self):
        """OI absent from snapshot (Alpaca omits it) → DataFrame stores -1 sentinel."""
        client = _make_client()
        # Use spec= so that 'open_interest' does NOT exist as an attribute.
        # MagicMock auto-creates attributes on access, so 'del snap.open_interest'
        # would be immediately re-created on the next access; spec is the right tool.
        snap = MagicMock(spec=['latest_quote', 'latest_trade', 'implied_volatility'])
        snap.latest_quote.bid_price = 1.0
        snap.latest_quote.ask_price = 2.0
        snap.latest_trade.price = 1.5
        snap.implied_volatility = 0.3
        data = {'AAPL240119C00200000': snap}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chain('AAPL', '2024-01-19')
        self.assertEqual(result.calls.iloc[0]['openInterest'], -1)


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_option_chains_for_range
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOptionChainsForRange(unittest.TestCase):

    def test_returns_dict_keyed_by_expiry(self):
        client = _make_client()
        data = {
            'AAPL240119C00200000': _make_snap(),
            'AAPL240119P00180000': _make_snap(),
        }
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        self.assertIn('2024-01-19', result)

    def test_each_entry_is_option_chain(self):
        client = _make_client()
        data = {'AAPL240119C00200000': _make_snap()}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        for chain in result.values():
            self.assertIsInstance(chain, OptionChain)

    def test_groups_multiple_expiries_separately(self):
        client = _make_client()
        data = {
            'AAPL240119C00200000': _make_snap(),   # expiry 2024-01-19
            'AAPL240119P00180000': _make_snap(),   # expiry 2024-01-19
            'AAPL240126C00205000': _make_snap(),   # expiry 2024-01-26
        }
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-02-01')
        self.assertIn('2024-01-19', result)
        self.assertIn('2024-01-26', result)
        # Jan 19 expiry should have 1 put and 1 call
        self.assertEqual(len(result['2024-01-19'].puts), 1)
        self.assertEqual(len(result['2024-01-19'].calls), 1)
        # Jan 26 expiry should have only 1 call and empty puts
        self.assertEqual(len(result['2024-01-26'].calls), 1)

    def test_empty_chain_returns_empty_dict(self):
        client = _make_client()
        client._option.get_option_chain.return_value = _make_chain_mock({})
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        self.assertEqual(result, {})

    def test_sdk_exception_returns_empty_dict(self):
        client = _make_client()
        client._option.get_option_chain.side_effect = Exception('API error')
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        self.assertEqual(result, {})

    def test_accepts_date_string_inputs(self):
        client = _make_client()
        client._option.get_option_chain.return_value = _make_chain_mock({})
        # Should not raise even with ISO string dates
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        self.assertIsInstance(result, dict)

    def test_calls_df_has_required_columns(self):
        client = _make_client()
        data = {'AAPL240119C00200000': _make_snap()}
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        chain = result['2024-01-19']
        for col in ['strike', 'bid', 'ask', 'lastPrice', 'volume', 'impliedVolatility', 'openInterest']:
            self.assertIn(col, chain.calls.columns)

    def test_puts_df_empty_when_no_puts_for_expiry(self):
        client = _make_client()
        data = {'AAPL240119C00200000': _make_snap()}  # only a call
        client._option.get_option_chain.return_value = _make_chain_mock(data)
        result = client.get_option_chains_for_range('AAPL', '2024-01-10', '2024-01-25')
        chain = result['2024-01-19']
        self.assertTrue(chain.puts.empty)
        self.assertEqual(len(chain.calls), 1)


# ══════════════════════════════════════════════════════════════════════════════
# make_alpaca_data_client  (factory)
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeAlpacaDataClient(unittest.TestCase):

    def test_missing_key_returns_none(self):
        result = make_alpaca_data_client({'alpaca': {'api_key': '', 'api_secret': 'sec'}})
        self.assertIsNone(result)

    def test_missing_secret_returns_none(self):
        result = make_alpaca_data_client({'alpaca': {'api_key': 'key', 'api_secret': ''}})
        self.assertIsNone(result)

    def test_both_missing_returns_none(self):
        result = make_alpaca_data_client({'alpaca': {'api_key': '', 'api_secret': ''}})
        self.assertIsNone(result)

    def test_empty_config_returns_none(self):
        result = make_alpaca_data_client({})
        self.assertIsNone(result)

    def test_valid_config_returns_client(self):
        result = make_alpaca_data_client({
            'alpaca': {'api_key': 'test_key', 'api_secret': 'test_secret'}
        })
        self.assertIsInstance(result, AlpacaDataClient)

    def test_env_vars_override_config(self):
        with patch.dict('os.environ', {'ALPACA_API_KEY': 'env_key', 'ALPACA_API_SECRET': 'env_sec'}):
            result = make_alpaca_data_client({'alpaca': {'api_key': '', 'api_secret': ''}})
        self.assertIsInstance(result, AlpacaDataClient)

    def test_env_var_key_with_blank_config_secret_returns_none(self):
        with patch.dict('os.environ', {'ALPACA_API_KEY': 'env_key'}, clear=False):
            # ALPACA_API_SECRET not set, config secret blank
            env = {k: v for k, v in __import__('os').environ.items()
                   if k != 'ALPACA_API_SECRET'}
            with patch.dict('os.environ', env, clear=True):
                result = make_alpaca_data_client({'alpaca': {'api_key': '', 'api_secret': ''}})
        self.assertIsNone(result)

    def test_constructor_exception_returns_none(self):
        with patch('src.alpaca_data.AlpacaDataClient', side_effect=ImportError('SDK missing')):
            result = make_alpaca_data_client({
                'alpaca': {'api_key': 'k', 'api_secret': 's'}
            })
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# _alpaca_retry  (HFT retry helper)
# ══════════════════════════════════════════════════════════════════════════════

from src.alpaca_data import _alpaca_retry


class TestAlpacaRetry(unittest.TestCase):

    def test_succeeds_on_first_attempt(self):
        fn = MagicMock(return_value=42)
        result = _alpaca_retry(fn, max_retries=3, base_delay=0.0)
        self.assertEqual(result, 42)
        fn.assert_called_once()

    def test_succeeds_on_third_attempt(self):
        fn = MagicMock(side_effect=[Exception('err'), Exception('err'), 99])
        result = _alpaca_retry(fn, max_retries=3, base_delay=0.0)
        self.assertEqual(result, 99)
        self.assertEqual(fn.call_count, 3)

    def test_raises_after_exhausted_retries(self):
        fn = MagicMock(side_effect=Exception('always fails'))
        with self.assertRaises(RuntimeError):
            _alpaca_retry(fn, max_retries=2, base_delay=0.0)
        self.assertEqual(fn.call_count, 3)  # 1 original + 2 retries

    def test_raises_wraps_original_exception(self):
        original = ValueError('root cause')
        fn = MagicMock(side_effect=original)
        with self.assertRaises(RuntimeError) as ctx:
            _alpaca_retry(fn, max_retries=1, base_delay=0.0)
        self.assertIs(ctx.exception.__cause__, original)

    def test_zero_retries_raises_immediately_on_failure(self):
        fn = MagicMock(side_effect=Exception('fail'))
        with self.assertRaises(RuntimeError):
            _alpaca_retry(fn, max_retries=0, base_delay=0.0)
        fn.assert_called_once()

    @patch('src.alpaca_data.time.sleep')
    def test_exponential_backoff_delays(self, mock_sleep):
        fn = MagicMock(side_effect=[Exception('a'), Exception('b'), 'ok'])
        _alpaca_retry(fn, max_retries=3, base_delay=2.0)
        # Delays: 2.0, 4.0
        mock_sleep.assert_any_call(2.0)
        mock_sleep.assert_any_call(4.0)
        self.assertEqual(mock_sleep.call_count, 2)


# ══════════════════════════════════════════════════════════════════════════════
# _snapshot_to_row — greeks extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshotToRowGreeks(unittest.TestCase):

    def _make_snap_with_greeks(self, delta=-0.15, gamma=0.02, theta=-0.05,
                                vega=0.10, rho=-0.01):
        snap = _make_snap(bid=1.0, ask=2.0, last=1.5, iv=0.3, oi=100)
        greeks = MagicMock()
        greeks.delta = delta
        greeks.gamma = gamma
        greeks.theta = theta
        greeks.vega  = vega
        greeks.rho   = rho
        snap.greeks = greeks
        return snap

    def test_greeks_extracted_from_snapshot(self):
        snap = self._make_snap_with_greeks()
        row = _snapshot_to_row('AAPL240119P00200000', snap)
        self.assertAlmostEqual(row['delta'], -0.15)
        self.assertAlmostEqual(row['gamma'],  0.02)
        self.assertAlmostEqual(row['theta'], -0.05)
        self.assertAlmostEqual(row['vega'],   0.10)
        self.assertAlmostEqual(row['rho'],   -0.01)

    def test_greeks_none_when_snapshot_has_no_greeks_attr(self):
        snap = MagicMock(spec=['latest_quote', 'latest_trade', 'implied_volatility'])
        snap.latest_quote.bid_price = 1.0
        snap.latest_quote.ask_price = 2.0
        snap.latest_trade.price = 1.5
        snap.implied_volatility = 0.3
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertIsNone(row['delta'])
        self.assertIsNone(row['gamma'])
        self.assertIsNone(row['theta'])

    def test_greek_none_value_becomes_none_in_row(self):
        snap = _make_snap()
        greeks = MagicMock()
        greeks.delta = None
        greeks.gamma = 0.02
        greeks.theta = None
        greeks.vega  = None
        greeks.rho   = None
        snap.greeks = greeks
        row = _snapshot_to_row('AAPL240119C00200000', snap)
        self.assertIsNone(row['delta'])
        self.assertAlmostEqual(row['gamma'], 0.02)
        self.assertIsNone(row['theta'])


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_option_snapshots  (HFT targeted fetch)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOptionSnapshots(unittest.TestCase):

    def _make_snap_with_greeks(self, delta=-0.10, gamma=0.03):
        snap = _make_snap(bid=1.0, ask=2.0, last=1.5, iv=0.25, oi=200)
        g = MagicMock()
        g.delta = delta; g.gamma = gamma
        g.theta = -0.04; g.vega = 0.08; g.rho = -0.005
        snap.greeks = g
        return snap

    def test_returns_dict_keyed_by_osi_symbol(self):
        client = _make_client()
        osi = 'AAPL240119P00180000'
        raw = MagicMock()
        raw.data = {osi: self._make_snap_with_greeks()}
        client._option.get_option_snapshot.return_value = raw
        result = client.get_option_snapshots([osi], max_retries=0)
        self.assertIn(osi, result)

    def test_row_contains_greek_fields(self):
        client = _make_client()
        osi = 'AAPL240119P00180000'
        raw = MagicMock()
        raw.data = {osi: self._make_snap_with_greeks(delta=-0.12, gamma=0.025)}
        client._option.get_option_snapshot.return_value = raw
        result = client.get_option_snapshots([osi], max_retries=0)
        row = result[osi]
        self.assertAlmostEqual(row['delta'], -0.12)
        self.assertAlmostEqual(row['gamma'],  0.025)

    def test_row_does_not_contain_internal_opt_type_key(self):
        client = _make_client()
        osi = 'AAPL240119C00200000'
        raw = MagicMock()
        raw.data = {osi: self._make_snap_with_greeks()}
        client._option.get_option_snapshot.return_value = raw
        result = client.get_option_snapshots([osi], max_retries=0)
        self.assertNotIn('_opt_type', result[osi])

    def test_empty_osi_list_returns_empty_dict_without_api_call(self):
        client = _make_client()
        result = client.get_option_snapshots([], max_retries=0)
        self.assertEqual(result, {})
        client._option.get_option_snapshot.assert_not_called()

    def test_raises_runtime_error_when_api_fails_after_retries(self):
        client = _make_client()
        client._option.get_option_snapshot.side_effect = Exception('network error')
        with self.assertRaises(RuntimeError):
            client.get_option_snapshots(['AAPL240119P00180000'], max_retries=1, base_delay=0.0)

    def test_raises_when_response_data_empty(self):
        client = _make_client()
        raw = MagicMock()
        raw.data = {}
        client._option.get_option_snapshot.return_value = raw
        with self.assertRaises(RuntimeError):
            client.get_option_snapshots(['AAPL240119P00180000'], max_retries=0)


# ══════════════════════════════════════════════════════════════════════════════
# AlpacaDataClient.get_spot_price_strict  (HFT strict spot price)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSpotPriceStrict(unittest.TestCase):

    def test_returns_price_when_available(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.return_value = {
            'AAPL': MagicMock(close=175.5)
        }
        result = client.get_spot_price_strict('AAPL', max_retries=0)
        self.assertAlmostEqual(result, 175.5)

    def test_raises_when_symbol_missing_from_response(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.return_value = {}
        with self.assertRaises(RuntimeError):
            client.get_spot_price_strict('AAPL', max_retries=0)

    def test_raises_after_retries_exhausted_on_api_error(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.side_effect = Exception('timeout')
        with self.assertRaises(RuntimeError):
            client.get_spot_price_strict('AAPL', max_retries=1, base_delay=0.0)

    def test_succeeds_on_second_attempt(self):
        client = _make_client()
        client._stock.get_stock_latest_bar.side_effect = [
            Exception('first fail'),
            {'MSFT': MagicMock(close=300.0)},
        ]
        result = client.get_spot_price_strict('MSFT', max_retries=2, base_delay=0.0)
        self.assertAlmostEqual(result, 300.0)


if __name__ == '__main__':
    unittest.main()
