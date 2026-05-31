"""
Tests for DataAdapter.fetch_alpaca_chains — HFT no-options handling.

Covers the behaviour introduced to prevent wasted retry time when Alpaca
returns an empty chain (ticker has no listed options):

1. Empty response → returns None immediately, no retry, no sleep.
2. Empty response → ticker is added to the session no-options cache.
3. Subsequent call for cached ticker → returns None without calling Alpaca.
4. Exception → still retries (transient network / rate-limit error).
5. Exception (exhausted) → returns None, ticker NOT added to no-options cache
   (a network error is not a confirmation of "no options").
6. Non-HFT path → empty response returns {}, no cache, no retry.
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, call, patch
from datetime import date

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.market_data.adapter import DataAdapter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_adapter(hft: bool = True, max_retries: int = 2, base_delay: float = 0.0,
                  client=None) -> DataAdapter:
    """Build a DataAdapter with controllable HFT mode and a mock client."""
    return DataAdapter(
        hist_cache      = {},
        chain_cache     = {},
        client_getter   = lambda: client,
        hft_mode_getter = lambda: hft,
        hft_config      = {
            'max_retries':               max_retries,
            'retry_base_delay_seconds':  base_delay,
        },
    )


def _today_target():
    """Return (today_date, target_date) far in the future for date args."""
    today  = date(2026, 3, 25)
    target = date(2026, 5, 15)
    return today, target


# ── HFT: empty response ────────────────────────────────────────────────────────

class TestHftEmptyResponse:

    def test_empty_chain_returns_none_immediately(self):
        """Alpaca returns {} → fetch_alpaca_chains returns None, no retry."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}
        adapter = _make_adapter(hft=True, max_retries=3, base_delay=0.0, client=client)
        today, target = _today_target()

        result = adapter.fetch_alpaca_chains('JKHY', today, target)

        assert result is None
        # Called exactly once — no retry
        assert client.get_option_chains_for_range.call_count == 1

    def test_empty_chain_does_not_sleep(self):
        """No sleep.time call should be made for empty-response path."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}
        adapter = _make_adapter(hft=True, max_retries=3, base_delay=5.0, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep') as mock_sleep:
            adapter.fetch_alpaca_chains('KDP', today, target)

        mock_sleep.assert_not_called()

    def test_empty_chain_adds_to_no_options_cache(self):
        """Empty response → symbol stored in _no_options_cache."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}
        adapter = _make_adapter(hft=True, client=client)
        today, target = _today_target()

        adapter.fetch_alpaca_chains('KEYS', today, target)

        assert 'KEYS' in adapter._no_options_cache

    def test_none_response_treated_same_as_empty(self):
        """get_option_chains_for_range returning None → also no retry."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = None
        adapter = _make_adapter(hft=True, max_retries=3, base_delay=0.0, client=client)
        today, target = _today_target()

        result = adapter.fetch_alpaca_chains('KIM', today, target)

        assert result is None
        assert client.get_option_chains_for_range.call_count == 1
        assert 'KIM' in adapter._no_options_cache


# ── HFT: no-options session cache ─────────────────────────────────────────────

class TestHftNoOptionsCache:

    def test_cached_ticker_skipped_without_api_call(self):
        """Second call for an already-empty ticker skips Alpaca entirely."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}
        adapter = _make_adapter(hft=True, client=client)
        today, target = _today_target()

        # First call populates cache
        adapter.fetch_alpaca_chains('KEY', today, target)
        assert client.get_option_chains_for_range.call_count == 1

        # Second call should be a no-op
        result = adapter.fetch_alpaca_chains('KEY', today, target)
        assert result is None
        assert client.get_option_chains_for_range.call_count == 1  # not called again

    def test_cache_is_per_instance(self):
        """Two separate adapters do not share the no-options cache."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}

        adapter_a = _make_adapter(hft=True, client=client)
        adapter_b = _make_adapter(hft=True, client=client)
        today, target = _today_target()

        adapter_a.fetch_alpaca_chains('KLAC', today, target)
        # adapter_b has its own empty cache — must still call Alpaca
        adapter_b.fetch_alpaca_chains('KLAC', today, target)

        assert client.get_option_chains_for_range.call_count == 2

    def test_successful_chain_not_added_to_cache(self):
        """A ticker that returns a real chain must NOT be in the cache."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {'2026-05-16': MagicMock()}
        adapter = _make_adapter(hft=True, client=client)
        today, target = _today_target()

        result = adapter.fetch_alpaca_chains('AAPL', today, target)

        assert result is not None
        assert 'AAPL' not in adapter._no_options_cache


# ── HFT: exception path still retries ─────────────────────────────────────────

class TestHftExceptionRetry:

    def test_exception_retries_up_to_max(self):
        """RuntimeError / network error → retried max_retries times."""
        client = MagicMock()
        client.get_option_chains_for_range.side_effect = ConnectionError("timeout")
        adapter = _make_adapter(hft=True, max_retries=2, base_delay=0.0, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep'):
            result = adapter.fetch_alpaca_chains('L', today, target)

        assert result is None
        # 1 initial + 2 retries = 3 total calls
        assert client.get_option_chains_for_range.call_count == 3

    def test_exception_does_not_cache_ticker(self):
        """A network error must NOT add the ticker to _no_options_cache."""
        client = MagicMock()
        client.get_option_chains_for_range.side_effect = OSError("network unreachable")
        adapter = _make_adapter(hft=True, max_retries=1, base_delay=0.0, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep'):
            adapter.fetch_alpaca_chains('MSFT', today, target)

        assert 'MSFT' not in adapter._no_options_cache

    def test_exception_sleeps_with_backoff(self):
        """Each retry sleeps with exponential backoff based on base_delay."""
        client = MagicMock()
        client.get_option_chains_for_range.side_effect = TimeoutError("timed out")
        adapter = _make_adapter(hft=True, max_retries=2, base_delay=1.0, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep') as mock_sleep:
            adapter.fetch_alpaca_chains('TSLA', today, target)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1.0, 2.0]  # base * 2^0, base * 2^1

    def test_succeeds_after_transient_error(self):
        """Retry succeeds on the second attempt — result returned, no cache."""
        client = MagicMock()
        good_chain = {'2026-05-16': MagicMock()}
        client.get_option_chains_for_range.side_effect = [
            ConnectionError("blip"),
            good_chain,
        ]
        adapter = _make_adapter(hft=True, max_retries=2, base_delay=0.0, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep'):
            result = adapter.fetch_alpaca_chains('SPY', today, target)

        assert result is good_chain
        assert 'SPY' not in adapter._no_options_cache
        assert client.get_option_chains_for_range.call_count == 2


# ── Non-HFT path unaffected ───────────────────────────────────────────────────

class TestNonHftPath:

    def test_non_hft_empty_returns_empty_dict(self):
        """Non-HFT: empty Alpaca response returns {} (not None), no cache."""
        client = MagicMock()
        client.get_option_chains_for_range.return_value = {}
        adapter = _make_adapter(hft=False, client=client)
        today, target = _today_target()

        result = adapter.fetch_alpaca_chains('JKHY', today, target)

        assert result == {}
        assert 'JKHY' not in adapter._no_options_cache

    def test_non_hft_exception_returns_empty_dict(self):
        """Non-HFT: Alpaca exception returns {} — caller falls back to yfinance."""
        client = MagicMock()
        client.get_option_chains_for_range.side_effect = Exception("boom")
        adapter = _make_adapter(hft=False, client=client)
        today, target = _today_target()

        result = adapter.fetch_alpaca_chains('KDP', today, target)

        assert result == {}

    def test_non_hft_no_retry_on_exception(self):
        """Non-HFT: Alpaca exception → single attempt only, no retry loop."""
        client = MagicMock()
        client.get_option_chains_for_range.side_effect = Exception("boom")
        adapter = _make_adapter(hft=False, client=client)
        today, target = _today_target()

        with patch('src.market_data.adapter.time.sleep') as mock_sleep:
            adapter.fetch_alpaca_chains('KEY', today, target)

        assert client.get_option_chains_for_range.call_count == 1
        mock_sleep.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# get_spot_price — HFT vs non-HFT routing
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSpotPrice:

    def test_hft_returns_alpaca_price(self):
        """HFT mode: Alpaca price returned directly."""
        client = MagicMock()
        client.get_spot_price.return_value = 185.50
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_price('AAPL')

        assert result == 185.50
        client.get_spot_price.assert_called_once_with('AAPL')

    def test_hft_returns_none_when_alpaca_fails(self):
        """HFT mode: no yfinance fallback — returns None on Alpaca failure."""
        client = MagicMock()
        client.get_spot_price.side_effect = ConnectionError("timeout")
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_price('AAPL')

        assert result is None

    def test_hft_returns_none_when_alpaca_returns_none(self):
        """HFT mode: Alpaca returns None → no fallback, return None."""
        client = MagicMock()
        client.get_spot_price.return_value = None
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_price('SPY')

        assert result is None

    def test_hft_returns_none_when_alpaca_returns_zero(self):
        """HFT mode: Alpaca returns 0 (invalid) → no fallback, return None."""
        client = MagicMock()
        client.get_spot_price.return_value = 0
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_price('SPY')

        assert result is None

    @patch('src.market_data.adapter.yf')
    def test_non_hft_falls_back_to_yfinance_on_alpaca_failure(self, mock_yf):
        """Non-HFT mode: Alpaca fails → yfinance fallback used."""
        client = MagicMock()
        client.get_spot_price.side_effect = ConnectionError("timeout")
        adapter = _make_adapter(hft=False, client=client)

        mock_info = MagicMock()
        mock_info.last_price = 190.25
        mock_yf.Ticker.return_value.fast_info = mock_info

        result = adapter.get_spot_price('AAPL')

        assert result == 190.25

    @patch('src.market_data.adapter.yf')
    def test_non_hft_uses_alpaca_first(self, mock_yf):
        """Non-HFT mode: Alpaca succeeds → yfinance not called."""
        client = MagicMock()
        client.get_spot_price.return_value = 450.0
        adapter = _make_adapter(hft=False, client=client)

        result = adapter.get_spot_price('SPY')

        assert result == 450.0
        mock_yf.Ticker.assert_not_called()

    @patch('src.market_data.adapter.yf')
    def test_hft_never_calls_yfinance(self, mock_yf):
        """HFT mode: yfinance must never be invoked regardless of Alpaca result."""
        client = MagicMock()
        client.get_spot_price.return_value = None
        adapter = _make_adapter(hft=True, client=client)

        adapter.get_spot_price('AAPL')

        mock_yf.Ticker.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# get_spot_prices — bulk fetch, HFT vs non-HFT
# ══════════════════════════════════════════════════════════════════════════════

class TestGetSpotPrices:

    def test_hft_returns_alpaca_bulk_prices(self):
        """HFT mode: bulk Alpaca call returns all prices."""
        client = MagicMock()
        client.get_spot_prices.return_value = {'SPY': 450.0, 'QQQ': 380.0}
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_prices(['SPY', 'QQQ'])

        assert result == {'SPY': 450.0, 'QQQ': 380.0}

    def test_hft_no_yfinance_fallback_for_missing(self):
        """HFT mode: missing symbols from Alpaca → NOT filled by yfinance."""
        client = MagicMock()
        client.get_spot_prices.return_value = {'SPY': 450.0}  # QQQ missing
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_prices(['SPY', 'QQQ'])

        assert 'QQQ' not in result
        assert result == {'SPY': 450.0}

    @patch('src.market_data.adapter.yf')
    def test_non_hft_fills_missing_via_yfinance(self, mock_yf):
        """Non-HFT mode: yfinance fills symbols missing from Alpaca."""
        client = MagicMock()
        client.get_spot_prices.return_value = {'SPY': 450.0}
        adapter = _make_adapter(hft=False, client=client)

        mock_info = MagicMock()
        mock_info.last_price = 380.0
        mock_yf.Ticker.return_value.fast_info = mock_info

        result = adapter.get_spot_prices(['SPY', 'QQQ'])

        assert result == {'SPY': 450.0, 'QQQ': 380.0}
        mock_yf.Ticker.assert_called_once_with('QQQ')

    def test_hft_returns_empty_dict_on_alpaca_exception(self):
        """HFT mode: Alpaca exception → empty dict, no fallback."""
        client = MagicMock()
        client.get_spot_prices.side_effect = RuntimeError("API down")
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_spot_prices(['SPY', 'QQQ'])

        assert result == {}


# ══════════════════════════════════════════════════════════════════════════════
# get_historical_close — HFT vs non-HFT
# ══════════════════════════════════════════════════════════════════════════════

class TestGetHistoricalClose:

    def test_hft_returns_none_when_alpaca_fails(self):
        """HFT mode: Alpaca exception → returns None, no yfinance."""
        client = MagicMock()
        client._stock.get_stock_bars.side_effect = RuntimeError("network error")
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_historical_close('AAPL', '2026-05-20', '2026-05-26')

        assert result is None

    @patch('src.market_data.adapter.yf')
    def test_hft_never_calls_yfinance(self, mock_yf):
        """HFT mode: yfinance must not be called even on Alpaca failure."""
        client = MagicMock()
        client._stock.get_stock_bars.side_effect = RuntimeError("down")
        adapter = _make_adapter(hft=True, client=client)

        adapter.get_historical_close('AAPL', '2026-05-20', '2026-05-26')

        mock_yf.Ticker.assert_not_called()

    @patch('src.market_data.adapter.yf')
    def test_non_hft_falls_back_to_yfinance(self, mock_yf):
        """Non-HFT mode: Alpaca fails → yfinance fallback returns close."""
        client = MagicMock()
        client._stock.get_stock_bars.side_effect = RuntimeError("down")
        adapter = _make_adapter(hft=False, client=client)

        mock_hist = MagicMock()
        mock_hist.empty = False
        mock_hist.__getitem__ = lambda self, key: MagicMock(iloc=MagicMock(__getitem__=lambda s, i: 185.0))
        mock_yf.Ticker.return_value.history.return_value = mock_hist

        result = adapter.get_historical_close('AAPL', '2026-05-20', '2026-05-26')

        assert result == 185.0

    def test_hft_returns_close_from_alpaca_bars(self):
        """HFT mode: Alpaca returns bar data → close price extracted."""
        client = MagicMock()
        bar = MagicMock()
        bar.close = 192.50
        bar_set = MagicMock()
        bar_set.data = {'AAPL': [bar]}
        client._stock.get_stock_bars.return_value = bar_set
        adapter = _make_adapter(hft=True, client=client)

        result = adapter.get_historical_close('AAPL', '2026-05-20', '2026-05-26')

        assert result == 192.50
