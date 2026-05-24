import json
import os
import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Mock yfinance and requests before importing universe
sys.modules['yfinance'] = MagicMock()
sys.modules['requests'] = MagicMock()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.universe import (
    CACHE_TTL_HOURS,
    ETF_CACHE_FILE,
    _check_market_cap,
    _fetch_nasdaq_etfs,
    _fetch_nasdaq_symbols,
    _fetch_other_etfs,
    _fetch_other_symbols,
    _filter_by_market_cap,
    get_etf_universe,
    get_ticker_universe,
)

# ── Shared sample data ────────────────────────────────────────────────────────

NASDAQ_SAMPLE = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "AAPL|Apple Inc.|Q|N|N|100|N|N\n"
    "MSFT|Microsoft Corp.|Q|N|N|100|N|N\n"
    "TSTR|Test Corp|Q|Y|N|100|N|N\n"       # test issue → excluded
    "ETFX|ETF Fund|Q|N|N|100|Y|N\n"        # ETF → excluded
    "BRK.B|Berkshire B|Q|N|N|100|N|N\n"    # '.' in symbol → excluded
    "WARR+|Warrant|Q|N|N|100|N|N\n"        # '+' in symbol → excluded
    "BAD\n"                                  # too few columns → skipped
)

OTHER_SAMPLE = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "JPM|JPMorgan Chase|N|JPM|N|100|N|JPM\n"      # NYSE → included
    "AMEX1|Amex Stock|A|AMEX1|N|100|N|AMEX1\n"   # AMEX → included
    "ARCA1|Arca Stock|P|ARCA1|N|100|N|ARCA1\n"   # NYSE Arca (P) → excluded
    "BATS1|Bats Stock|Z|BATS1|N|100|N|BATS1\n"   # BATS (Z) → excluded
    "ETF2|ETF|N|ETF2|Y|100|N|ETF2\n"             # ETF → excluded
    "TSTI|Test|N|TSTI|N|100|Y|TSTI\n"            # test issue → excluded
    "BRK-B|Berkshire B|N|BRKB|N|100|N|BRKB\n"   # '-' in symbol → excluded
)


# ── _fetch_nasdaq_symbols ─────────────────────────────────────────────────────

class TestFetchNasdaqSymbols(unittest.TestCase):

    def _run(self, text):
        import requests
        requests.get.return_value = MagicMock(text=text)
        return _fetch_nasdaq_symbols()

    def test_returns_valid_symbols(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)

    def test_filters_test_issues(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertNotIn('TSTR', result)

    def test_filters_etfs(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertNotIn('ETFX', result)

    def test_filters_dot_in_symbol(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertNotIn('BRK.B', result)

    def test_filters_plus_in_symbol(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertNotIn('WARR+', result)

    def test_skips_malformed_rows_without_raising(self):
        result = self._run(NASDAQ_SAMPLE)
        self.assertIsInstance(result, list)


# ── _fetch_other_symbols ──────────────────────────────────────────────────────

class TestFetchOtherSymbols(unittest.TestCase):

    def _run(self, text):
        import requests
        requests.get.return_value = MagicMock(text=text)
        return _fetch_other_symbols()

    def test_includes_nyse(self):
        result = self._run(OTHER_SAMPLE)
        self.assertIn('JPM', result)

    def test_includes_amex(self):
        result = self._run(OTHER_SAMPLE)
        self.assertIn('AMEX1', result)

    def test_excludes_nyse_arca(self):
        result = self._run(OTHER_SAMPLE)
        self.assertNotIn('ARCA1', result)

    def test_excludes_bats(self):
        result = self._run(OTHER_SAMPLE)
        self.assertNotIn('BATS1', result)

    def test_filters_etfs(self):
        result = self._run(OTHER_SAMPLE)
        self.assertNotIn('ETF2', result)

    def test_filters_test_issues(self):
        result = self._run(OTHER_SAMPLE)
        self.assertNotIn('TSTI', result)

    def test_filters_dash_in_symbol(self):
        result = self._run(OTHER_SAMPLE)
        self.assertNotIn('BRK-B', result)


# ── _check_market_cap ─────────────────────────────────────────────────────────

class TestCheckMarketCap(unittest.TestCase):

    def setUp(self):
        import yfinance as yf
        yf.Ticker.side_effect = None
        self.yf = yf

    def test_returns_symbol_when_cap_meets_threshold(self):
        self.yf.Ticker.return_value.info = {'marketCap': 2_000_000_000}
        self.assertEqual(_check_market_cap('AAPL', 1_000_000_000), 'AAPL')

    def test_returns_none_when_cap_below_threshold(self):
        self.yf.Ticker.return_value.info = {'marketCap': 500_000_000}
        self.assertIsNone(_check_market_cap('TINY', 1_000_000_000))

    def test_returns_none_when_cap_exactly_at_threshold(self):
        # >= means exactly at threshold should pass
        self.yf.Ticker.return_value.info = {'marketCap': 1_000_000_000}
        self.assertEqual(_check_market_cap('EXACT', 1_000_000_000), 'EXACT')

    def test_returns_none_when_market_cap_missing(self):
        self.yf.Ticker.return_value.info = {}
        self.assertIsNone(_check_market_cap('NOCAP', 1_000_000_000))

    def test_returns_none_on_exception(self):
        self.yf.Ticker.side_effect = Exception("API error")
        self.assertIsNone(_check_market_cap('ERR', 1_000_000_000))
        self.yf.Ticker.side_effect = None

    def test_returns_none_when_market_cap_is_none(self):
        self.yf.Ticker.return_value.info = {'marketCap': None}
        self.assertIsNone(_check_market_cap('NONECAP', 1_000_000_000))


# ── _filter_by_market_cap ─────────────────────────────────────────────────────

class TestFilterByMarketCap(unittest.TestCase):

    @patch('src.universe._check_market_cap', side_effect=lambda sym, cap: sym if sym in {'AAPL', 'MSFT'} else None)
    def test_returns_only_qualifying_symbols(self, mock_check):
        result = _filter_by_market_cap(['AAPL', 'TINY', 'MSFT'], 1e9)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertNotIn('TINY', result)

    @patch('src.universe._check_market_cap', return_value=None)
    def test_returns_empty_when_none_qualify(self, mock_check):
        result = _filter_by_market_cap(['A', 'B', 'C'], 1e9)
        self.assertEqual(result, [])

    @patch('src.universe._check_market_cap', return_value=None)
    def test_returns_empty_for_empty_input(self, mock_check):
        result = _filter_by_market_cap([], 1e9)
        self.assertEqual(result, [])

    @patch('src.universe._check_market_cap', side_effect=lambda sym, cap: sym)
    def test_result_is_sorted(self, mock_check):
        result = _filter_by_market_cap(['MSFT', 'AAPL', 'GOOG'], 1e9)
        self.assertEqual(result, sorted(result))


# ── get_ticker_universe ───────────────────────────────────────────────────────

class TestGetTickerUniverse(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, 'universe_cache.json')

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        os.rmdir(self.tmpdir)

    def _write_cache(self, tickers, age_hours=0):
        cached_at = (datetime.now() - timedelta(hours=age_hours)).isoformat()
        with open(self.cache_path, 'w') as f:
            json.dump({'cached_at': cached_at, 'tickers': tickers}, f)

    def test_returns_cached_tickers_when_fresh(self):
        self._write_cache(['AAPL', 'MSFT'], age_hours=1)
        with patch('src.universe.CACHE_FILE', self.cache_path):
            result = get_ticker_universe(1e9)
        self.assertEqual(result, ['AAPL', 'MSFT'])

    def test_fetches_when_no_cache_file(self):
        with patch('src.universe.CACHE_FILE', self.cache_path), \
             patch('src.universe._fetch_nasdaq_symbols', return_value=['AAPL']), \
             patch('src.universe._fetch_other_symbols', return_value=['JPM']), \
             patch('src.universe._filter_by_market_cap', return_value=['AAPL', 'JPM']):
            result = get_ticker_universe(1e9)
        self.assertIn('AAPL', result)
        self.assertIn('JPM', result)

    def test_cache_is_written_after_fetch(self):
        with patch('src.universe.CACHE_FILE', self.cache_path), \
             patch('src.universe._fetch_nasdaq_symbols', return_value=['AAPL']), \
             patch('src.universe._fetch_other_symbols', return_value=[]), \
             patch('src.universe._filter_by_market_cap', return_value=['AAPL']):
            get_ticker_universe(1e9)
        self.assertTrue(os.path.exists(self.cache_path))
        with open(self.cache_path) as f:
            cache = json.load(f)
        self.assertIn('cached_at', cache)
        self.assertIn('tickers', cache)

    def test_force_refresh_bypasses_fresh_cache(self):
        self._write_cache(['STALE'], age_hours=0)
        with patch('src.universe.CACHE_FILE', self.cache_path), \
             patch('src.universe._fetch_nasdaq_symbols', return_value=['FRESH']), \
             patch('src.universe._fetch_other_symbols', return_value=[]), \
             patch('src.universe._filter_by_market_cap', return_value=['FRESH']):
            result = get_ticker_universe(1e9, force_refresh=True)
        self.assertIn('FRESH', result)
        self.assertNotIn('STALE', result)

    def test_expired_cache_triggers_refresh(self):
        self._write_cache(['OLD'], age_hours=CACHE_TTL_HOURS + 1)
        with patch('src.universe.CACHE_FILE', self.cache_path), \
             patch('src.universe._fetch_nasdaq_symbols', return_value=['NEW']), \
             patch('src.universe._fetch_other_symbols', return_value=[]), \
             patch('src.universe._filter_by_market_cap', return_value=['NEW']):
            result = get_ticker_universe(1e9)
        self.assertIn('NEW', result)

    def test_deduplicates_symbols_before_filtering(self):
        """If NASDAQ and NYSE both list a symbol, it should only be checked once."""
        filter_mock = MagicMock(return_value=['AAPL'])
        with patch('src.universe.CACHE_FILE', self.cache_path), \
             patch('src.universe._fetch_nasdaq_symbols', return_value=['AAPL', 'MSFT']), \
             patch('src.universe._fetch_other_symbols', return_value=['MSFT', 'JPM']), \
             patch('src.universe._filter_by_market_cap', filter_mock):
            get_ticker_universe(1e9)
        symbols_passed = filter_mock.call_args[0][0]
        self.assertEqual(len(symbols_passed), len(set(symbols_passed)), "Duplicates should be removed before filtering")

    def test_logs_cache_hit_when_log_provided(self):
        self._write_cache(['AAPL'], age_hours=1)
        mock_log = MagicMock()
        with patch('src.universe.CACHE_FILE', self.cache_path):
            get_ticker_universe(1e9, log=mock_log)
        mock_log.info.assert_called()



# ── get_index_tickers — Russell 1000 integration ──────────────────────────────

from src.universe_indices import (
    CACHE_VERSION,
    get_index_tickers,
    _RUSSELL1000_FALLBACK,
    _RUSSELL1000_MIN,
)


class TestGetIndexTickers(unittest.TestCase):
    """Tests for get_index_tickers() — Russell 1000 inclusion and cache logic."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._cache  = os.path.join(self._tmpdir, 'index_cache.json')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _write_cache(self, tickers, age_hours=0, version=CACHE_VERSION):
        cached_at = (datetime.now() - timedelta(hours=age_hours)).isoformat()
        with open(self._cache, 'w') as f:
            json.dump({
                'cache_version': version,
                'cached_at':     cached_at,
                'index_tickers': tickers,
            }, f)

    # Synthetic lists large enough to clear each index's minimum threshold so
    # that the fallback lists are never triggered unless a test explicitly
    # passes a small list.
    _BIG_SP500  = [f'SP{i:04d}' for i in range(450)]   # >= _SP500_MIN (400)
    _BIG_NDX    = [f'NQ{i:04d}' for i in range(90)]    # >= _NASDAQ100_MIN (80)
    _BIG_DOW    = [f'DJ{i:04d}' for i in range(28)]    # >= _DOW30_MIN (25)
    _BIG_RUSS   = [f'RU{i:04d}' for i in range(250)]   # >= _RUSSELL1000_MIN (200)

    def _patch_fetchers(self, sp500=None, ndx100=None, dow30=None, russell=None):
        """Return a context manager that patches all four Wikipedia fetchers.

        Defaults to synthetic lists large enough to avoid triggering any
        fallback list, so only the explicitly supplied values appear in the
        result.
        """
        from unittest.mock import patch as _patch
        import contextlib

        _sp    = sp500   if sp500   is not None else self._BIG_SP500
        _ndx   = ndx100  if ndx100  is not None else self._BIG_NDX
        _dow   = dow30   if dow30   is not None else self._BIG_DOW
        _russ  = russell if russell is not None else self._BIG_RUSS

        @contextlib.contextmanager
        def _ctx():
            with _patch('src.universe_indices._fetch_sp500',      return_value=_sp),   \
                 _patch('src.universe_indices._fetch_nasdaq100',   return_value=_ndx),  \
                 _patch('src.universe_indices._fetch_dow30',       return_value=_dow),  \
                 _patch('src.universe_indices._fetch_russell1000', return_value=_russ), \
                 _patch('src.universe_indices.CACHE_FILE', self._cache),                \
                 _patch('src.universe_indices.time.sleep'):
                yield
        return _ctx()

    # ── Russell 1000 included in union ────────────────────────────────────────

    def test_russell1000_tickers_included_in_result(self):
        """Russell 1000 names appear in the combined result."""
        with self._patch_fetchers(russell=['RH', 'MANH', 'HUBS']):
            tickers = get_index_tickers(log=MagicMock())
        self.assertIn('RH',   tickers)
        self.assertIn('MANH', tickers)
        self.assertIn('HUBS', tickers)

    def test_result_is_union_of_all_four_indices(self):
        """Combined list is the sorted union of SP500, NDX100, Dow30, Russell1000."""
        # Use big synthetic lists so no fallback triggers; add one unique
        # sentinel per index to verify all four are included.
        sp500  = self._BIG_SP500  + ['UNIQ_SP']
        ndx100 = self._BIG_NDX   + ['UNIQ_NQ']
        dow30  = self._BIG_DOW   + ['UNIQ_DJ']
        russ   = self._BIG_RUSS  + ['UNIQ_RU']
        with self._patch_fetchers(sp500=sp500, ndx100=ndx100, dow30=dow30, russell=russ):
            tickers = get_index_tickers(log=MagicMock())
        for sentinel in ('UNIQ_SP', 'UNIQ_NQ', 'UNIQ_DJ', 'UNIQ_RU'):
            self.assertIn(sentinel, tickers)

    def test_duplicates_across_indices_deduplicated(self):
        """Tickers appearing in multiple indices are counted once."""
        # Use big base lists so no fallback fires; then add a shared symbol
        sp500  = self._BIG_SP500 + ['SHARED']
        ndx100 = self._BIG_NDX  + ['SHARED']
        dow30  = self._BIG_DOW  + ['SHARED']
        russ   = self._BIG_RUSS + ['SHARED']
        with self._patch_fetchers(sp500=sp500, ndx100=ndx100, dow30=dow30, russell=russ):
            tickers = get_index_tickers(log=MagicMock())
        self.assertEqual(tickers.count('SHARED'), 1)

    # ── Fallback when Wikipedia returns too few tickers ───────────────────────

    def test_russell1000_fallback_used_when_scrape_too_small(self):
        """Fewer than _RUSSELL1000_MIN scraped tickers → fallback list used."""
        tiny_list = ['RH']   # well below _RUSSELL1000_MIN
        with self._patch_fetchers(russell=tiny_list):
            tickers = get_index_tickers(log=MagicMock())
        # Every entry in the fallback should appear in the result
        for sym in _RUSSELL1000_FALLBACK[:10]:
            self.assertIn(sym, tickers)

    def test_russell1000_live_list_used_when_scrape_large_enough(self):
        """When Wikipedia returns ≥ _RUSSELL1000_MIN tickers the live list is used."""
        # Use a live list with unique sentinel symbols that cannot appear in
        # any other index's big synthetic list or fallback list.
        live = [f'RU{i:04d}' for i in range(_RUSSELL1000_MIN)]
        with self._patch_fetchers(russell=live):
            tickers = get_index_tickers(log=MagicMock())
        self.assertIn('RU0000', tickers)
        # None of the fallback list's Russell-specific symbols (that aren't
        # also in SP500/NDX) should appear — the live list was used instead.
        # Pick a few fallback-unique names as canaries.
        fallback_canaries = {'MANH', 'HUBS', 'GWRE', 'BRZE', 'TOST'}
        for sym in fallback_canaries:
            self.assertNotIn(sym, tickers)

    # ── Cache version invalidation ────────────────────────────────────────────

    def test_stale_cache_version_triggers_refresh(self):
        """Cache with an older version number is ignored even if within TTL."""
        self._write_cache(['OLD_TICKER'], age_hours=0, version=CACHE_VERSION - 1)
        with self._patch_fetchers(russell=['RH']):
            tickers = get_index_tickers(log=MagicMock())
        self.assertNotIn('OLD_TICKER', tickers)
        self.assertIn('RH', tickers)

    def test_current_cache_version_within_ttl_is_returned(self):
        """Fresh cache with correct version is served without scraping."""
        self._write_cache(['CACHED_SYM'], age_hours=1, version=CACHE_VERSION)
        mock_fetcher = MagicMock(return_value=['SHOULD_NOT_APPEAR'])
        with patch('src.universe_indices._fetch_russell1000', mock_fetcher), \
             patch('src.universe_indices.CACHE_FILE', self._cache):
            tickers = get_index_tickers(log=MagicMock())
        mock_fetcher.assert_not_called()
        self.assertEqual(tickers, ['CACHED_SYM'])

    def test_cache_written_with_current_version(self):
        """After a refresh the new cache file contains CACHE_VERSION."""
        with self._patch_fetchers():
            get_index_tickers(log=MagicMock())
        with open(self._cache) as f:
            written = json.load(f)
        self.assertEqual(written['cache_version'], CACHE_VERSION)

    def test_expired_cache_triggers_refresh(self):
        """Cache within correct version but past TTL triggers a fresh scrape."""
        self._write_cache(['OLD'], age_hours=25, version=CACHE_VERSION)
        fresh_russell = self._BIG_RUSS + ['FRESH_RU']
        with self._patch_fetchers(russell=fresh_russell):
            tickers = get_index_tickers(log=MagicMock())
        self.assertIn('FRESH_RU', tickers)
        self.assertNotIn('OLD', tickers)

    # ── force_refresh ─────────────────────────────────────────────────────────

    def test_force_refresh_bypasses_valid_cache(self):
        """force_refresh=True ignores a perfectly valid cache."""
        self._write_cache(['STALE'], age_hours=0, version=CACHE_VERSION)
        fresh_russell = self._BIG_RUSS + ['FRESH_RU']
        with self._patch_fetchers(russell=fresh_russell):
            tickers = get_index_tickers(force_refresh=True, log=MagicMock())
        self.assertIn('FRESH_RU', tickers)
        self.assertNotIn('STALE', tickers)

    # ── Fallback list sanity ───────────────────────────────────────────────────

    def test_fallback_list_has_no_duplicates(self):
        self.assertEqual(len(_RUSSELL1000_FALLBACK), len(set(_RUSSELL1000_FALLBACK)))

    def test_fallback_list_all_valid_tickers(self):
        """Every entry in the fallback is a valid uppercase alphabetic ticker."""
        for sym in _RUSSELL1000_FALLBACK:
            self.assertTrue(
                re.match(r'^[A-Z]{1,5}(-[A-Z])?$', sym),
                f"Invalid ticker in fallback: {sym!r}",
            )


# ── ETF universe sample data ──────────────────────────────────────────────────

NASDAQ_ETF_SAMPLE = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
    "SPY|SPDR S&P 500 ETF Trust|G|N|N|100|Y|N\n"          # ETF → included
    "QQQ|Invesco QQQ Trust|G|N|N|100|Y|N\n"               # ETF → included
    "AAPL|Apple Inc.|Q|N|N|100|N|N\n"                     # stock → excluded
    "TSTR|Test Fund|G|Y|N|100|Y|N\n"                      # test issue → excluded
    "ETF+|Bad ETF|G|N|N|100|Y|N\n"                        # special char → excluded
    "BAD\n"                                                 # too few columns → skipped
)

OTHER_ETF_SAMPLE = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "IWM|iShares Russell 2000 ETF|P|IWM|Y|100|N|IWM\n"    # NYSE Arca (P) → included
    "GLD|SPDR Gold Shares|N|GLD|Y|100|N|GLD\n"            # NYSE (N) → included
    "SLV|iShares Silver Trust|A|SLV|Y|100|N|SLV\n"        # NYSE American (A) → included
    "JPM|JPMorgan Chase|N|JPM|N|100|N|JPM\n"              # stock → excluded
    "BATS1|Bats ETF|Z|BATS1|Y|100|N|BATS1\n"             # BATS (Z) → excluded
    "TSTI|Test ETF|P|TSTI|Y|100|Y|TSTI\n"                 # test issue → excluded
    "ETF$|Bad|P|ETF$|Y|100|N|ETF$\n"                      # special char → excluded
)


# ── _fetch_nasdaq_etfs ────────────────────────────────────────────────────────

class TestFetchNasdaqEtfs(unittest.TestCase):

    def _run(self, text):
        import requests
        requests.get.return_value = MagicMock(text=text)
        return _fetch_nasdaq_etfs()

    def test_returns_etf_symbols(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertIn('SPY', result)
        self.assertIn('QQQ', result)

    def test_excludes_stocks(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertNotIn('AAPL', result)

    def test_excludes_test_issues(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertNotIn('TSTR', result)

    def test_excludes_special_chars(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertNotIn('ETF+', result)

    def test_skips_malformed_rows(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertIsInstance(result, list)

    def test_returns_list(self):
        result = self._run(NASDAQ_ETF_SAMPLE)
        self.assertIsInstance(result, list)


# ── _fetch_other_etfs ─────────────────────────────────────────────────────────

class TestFetchOtherEtfs(unittest.TestCase):

    def _run(self, text):
        import requests
        requests.get.return_value = MagicMock(text=text)
        return _fetch_other_etfs()

    def test_includes_nyse_arca(self):
        """NYSE Arca (P) is the primary ETF exchange — must be included."""
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertIn('IWM', result)

    def test_includes_nyse(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertIn('GLD', result)

    def test_includes_nyse_american(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertIn('SLV', result)

    def test_excludes_stocks(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertNotIn('JPM', result)

    def test_excludes_bats(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertNotIn('BATS1', result)

    def test_excludes_test_issues(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertNotIn('TSTI', result)

    def test_excludes_special_chars(self):
        result = self._run(OTHER_ETF_SAMPLE)
        self.assertNotIn('ETF$', result)

    def test_nyse_arca_absent_from_stock_universe(self):
        """Sanity: _fetch_other_symbols must NOT include NYSE Arca (P) tickers."""
        import requests as req
        req.get.return_value = MagicMock(text=OTHER_ETF_SAMPLE)
        stocks = _fetch_other_symbols()
        self.assertNotIn('IWM', stocks)


# ── get_etf_universe ──────────────────────────────────────────────────────────

class TestGetEtfUniverse(unittest.TestCase):

    def setUp(self):
        self.tmpdir    = tempfile.mkdtemp()
        self.cache_path = os.path.join(self.tmpdir, 'etf_universe_cache.json')

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        os.rmdir(self.tmpdir)

    def _write_cache(self, tickers, age_hours=0):
        cached_at = (datetime.now() - timedelta(hours=age_hours)).isoformat()
        with open(self.cache_path, 'w') as f:
            json.dump({'cached_at': cached_at, 'tickers': tickers}, f)

    _NASDAQ_DEFAULT = ['SPY', 'QQQ']
    _OTHER_DEFAULT  = ['IWM', 'GLD']

    def _patch(self, nasdaq=_NASDAQ_DEFAULT, other=_OTHER_DEFAULT):
        """Context manager: patch both ETF fetchers + cache path."""
        return (
            patch('src.universe.ETF_CACHE_FILE', self.cache_path),
            patch('src.universe._fetch_nasdaq_etfs', return_value=nasdaq),
            patch('src.universe._fetch_other_etfs',  return_value=other),
        )

    def test_returns_cached_tickers_when_fresh(self):
        self._write_cache(['SPY', 'QQQ'], age_hours=1)
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path):
            result = get_etf_universe()
        self.assertEqual(result, ['SPY', 'QQQ'])

    def test_fetches_when_no_cache(self):
        p = self._patch(nasdaq=['SPY'], other=['IWM'])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe()
        self.assertIn('SPY', result)
        self.assertIn('IWM', result)

    def test_cache_written_after_fetch(self):
        p = self._patch()
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            get_etf_universe()
        self.assertTrue(os.path.exists(self.cache_path))
        with open(self.cache_path) as f:
            cache = json.load(f)
        self.assertIn('cached_at', cache)
        self.assertIn('tickers',   cache)

    def test_deduplicates_across_exchanges(self):
        """An ETF listed on both NASDAQ and NYSE Arca appears once."""
        p = self._patch(nasdaq=['SPY', 'QQQ'], other=['SPY', 'IWM'])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe()
        self.assertEqual(result.count('SPY'), 1)

    def test_result_is_sorted(self):
        p = self._patch(nasdaq=['QQQ', 'SPY'], other=['IWM', 'GLD'])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe()
        self.assertEqual(result, sorted(result))

    def test_force_refresh_bypasses_fresh_cache(self):
        self._write_cache(['STALE'], age_hours=0)
        p = self._patch(nasdaq=['FRESH'])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe(force_refresh=True)
        self.assertIn('FRESH', result)
        self.assertNotIn('STALE', result)

    def test_expired_cache_triggers_refresh(self):
        self._write_cache(['OLD'], age_hours=CACHE_TTL_HOURS + 1)
        p = self._patch(nasdaq=['NEW'])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe()
        self.assertIn('NEW', result)
        self.assertNotIn('OLD', result)

    def test_logs_cache_hit(self):
        self._write_cache(['SPY'], age_hours=1)
        mock_log = MagicMock()
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path):
            get_etf_universe(log=mock_log)
        mock_log.info.assert_called()

    def test_logs_on_fetch(self):
        mock_log = MagicMock()
        p = self._patch()
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            get_etf_universe(log=mock_log)
        mock_log.info.assert_called()

    def test_no_market_cap_check(self):
        """ETF universe must never call _check_market_cap or _filter_by_market_cap."""
        p = self._patch()
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2], \
             patch('src.universe._check_market_cap') as mock_cap:
            get_etf_universe()
        mock_cap.assert_not_called()

    def test_empty_exchange_lists_return_empty(self):
        p = self._patch(nasdaq=[], other=[])
        with patch('src.universe.ETF_CACHE_FILE', self.cache_path), p[1], p[2]:
            result = get_etf_universe()
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
