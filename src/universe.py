import json
import logging
import os
import time
import yfinance as yf
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from src.etf_universe_presets import stable_etf_underlyings

_log = logging.getLogger('optionwheel')

NASDAQ_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL   = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
CACHE_FILE         = "data/universe_cache.json"
ETF_CACHE_FILE     = "data/etf_universe_cache.json"
CACHE_TTL_HOURS    = 48          # re-build every 48 h (constituents change slowly)
_SPECIAL_CHARS     = set("^$./+-")

# NYSE Arca (P) is excluded from the *stock* universe because it is primarily
# an ETF/closed-end-fund exchange.  It is included in the ETF universe.
_STOCK_EXCHANGES   = frozenset(('N', 'A'))        # NYSE, NYSE American / AMEX
_ETF_EXCHANGES     = frozenset(('N', 'A', 'P'))   # + NYSE Arca (where most ETFs list)


def _fetch_nasdaq_symbols():
    """
    Download NASDAQ-listed symbols and return plain-stock tickers.

    File columns (pipe-delimited):
    Symbol | Security Name | Market Category | Test Issue |
    Financial Status | Round Lot Size | ETF | NextShares
    """
    resp = requests.get(NASDAQ_LISTED_URL, timeout=15)
    resp.raise_for_status()

    symbols = []
    for line in resp.text.splitlines()[1:]:          # skip header row
        parts = line.split('|')
        if len(parts) < 7:
            continue
        symbol     = parts[0].strip()
        test_issue = parts[3].strip()
        etf        = parts[6].strip()

        if test_issue == 'Y' or etf == 'Y':
            continue
        if any(c in symbol for c in _SPECIAL_CHARS):
            continue
        symbols.append(symbol)

    return symbols


def _fetch_other_symbols():
    """
    Download NYSE/AMEX-listed symbols and return plain-stock tickers.

    File columns (pipe-delimited):
    ACT Symbol | Security Name | Exchange | CQS Symbol |
    ETF | Round Lot Size | Test Issue | NASDAQ Symbol
    """
    resp = requests.get(OTHER_LISTED_URL, timeout=15)
    resp.raise_for_status()

    symbols = []
    for line in resp.text.splitlines()[1:]:          # skip header row
        parts = line.split('|')
        if len(parts) < 7:
            continue
        symbol     = parts[0].strip()
        exchange   = parts[2].strip()
        etf        = parts[4].strip()
        test_issue = parts[6].strip()

        if test_issue == 'Y' or etf == 'Y':
            continue
        if exchange not in _STOCK_EXCHANGES:
            continue
        if any(c in symbol for c in _SPECIAL_CHARS):
            continue
        symbols.append(symbol)

    return symbols


def _fetch_nasdaq_etfs():
    """
    Download NASDAQ-listed ETF tickers.

    Uses the same nasdaqtrader.com feed as _fetch_nasdaq_symbols but keeps
    rows where ETF == 'Y' instead of filtering them out.

    File columns (pipe-delimited):
    Symbol | Security Name | Market Category | Test Issue |
    Financial Status | Round Lot Size | ETF | NextShares
    """
    resp = requests.get(NASDAQ_LISTED_URL, timeout=15)
    resp.raise_for_status()

    symbols = []
    for line in resp.text.splitlines()[1:]:
        parts = line.split('|')
        if len(parts) < 7:
            continue
        symbol     = parts[0].strip()
        test_issue = parts[3].strip()
        etf        = parts[6].strip()

        if test_issue == 'Y' or etf != 'Y':
            continue
        if any(c in symbol for c in _SPECIAL_CHARS):
            continue
        symbols.append(symbol)

    return symbols


def _fetch_other_etfs():
    """
    Download NYSE / NYSE American / NYSE Arca-listed ETF tickers.

    NYSE Arca (exchange code 'P') is the primary listing venue for the
    majority of US ETFs (SPY, QQQ, IWM, GLD, …) and is included here even
    though it is excluded from the plain-stock universe.

    File columns (pipe-delimited):
    ACT Symbol | Security Name | Exchange | CQS Symbol |
    ETF | Round Lot Size | Test Issue | NASDAQ Symbol
    """
    resp = requests.get(OTHER_LISTED_URL, timeout=15)
    resp.raise_for_status()

    symbols = []
    for line in resp.text.splitlines()[1:]:
        parts = line.split('|')
        if len(parts) < 7:
            continue
        symbol     = parts[0].strip()
        exchange   = parts[2].strip()
        etf        = parts[4].strip()
        test_issue = parts[6].strip()

        if test_issue == 'Y' or etf != 'Y':
            continue
        if exchange not in _ETF_EXCHANGES:
            continue
        if any(c in symbol for c in _SPECIAL_CHARS):
            continue
        symbols.append(symbol)

    return symbols


def get_etf_universe(force_refresh=False, log=None):
    """
    Return all ETF tickers listed on NASDAQ, NYSE, NYSE American, and NYSE Arca.

    No AUM / market-cap filter is applied — ETF liquidity is enforced at scan
    time via the chain_liquidity thresholds (min_open_interest, min_bid,
    max_spread_pct).  This keeps the fetch fast (no yfinance calls) and
    avoids incorrectly excluding newer or smaller-AUM ETFs that have liquid
    options chains.

    Results are cached in data/etf_universe_cache.json for 48 hours.
    Pass force_refresh=True to bypass the cache and rebuild.
    """
    if not force_refresh and os.path.exists(ETF_CACHE_FILE):
        with open(ETF_CACHE_FILE) as f:
            cache = json.load(f)
        expires_at = datetime.fromisoformat(cache['cached_at']) + timedelta(hours=CACHE_TTL_HOURS)
        if datetime.now() < expires_at:
            tickers = cache['tickers']
            if log:
                log.info(
                    "ETF universe loaded from cache: %d tickers "
                    "(expires %s)", len(tickers), expires_at.strftime('%Y-%m-%d %H:%M')
                )
            return tickers

    if log:
        log.info("Fetching ETF listings from nasdaqtrader.com ...")

    nasdaq_etfs = _fetch_nasdaq_etfs()
    other_etfs  = _fetch_other_etfs()
    all_etfs    = sorted(set(nasdaq_etfs + other_etfs))

    os.makedirs('data', exist_ok=True)
    with open(ETF_CACHE_FILE, 'w') as f:
        json.dump({'cached_at': datetime.now().isoformat(), 'tickers': all_etfs}, f, indent=2)

    if log:
        log.info("ETF universe built: %d tickers. Cached to %s.", len(all_etfs), ETF_CACHE_FILE)

    return all_etfs


def get_stable_etf_universe(log=None):
    """Return the curated stable ETF universe used by the live ML scanner."""
    tickers = stable_etf_underlyings()
    if log:
        log.info(
            "Stable ETF universe preset loaded: %d tickers "
            "(curated liquid non-leveraged ETFs for options selling)",
            len(tickers),
        )
    return tickers


def _check_market_cap(symbol, min_cap):
    """
    Return symbol if its market cap meets the threshold, else None.

    Uses fast_info.market_cap (lightweight endpoint — avoids the heavy
    /v8/finance/quoteSummary call that triggers Yahoo 401/429 rate-limits
    when thousands of tickers are checked in parallel).  Falls back to
    ticker.info if fast_info is unavailable or returns no data.

    Retries up to two more times (total 3 attempts) with exponential
    backoff.  HTTP 401/429 responses get a much longer sleep so the
    caller does not compound the rate-limit.
    """
    def _is_rl(exc):
        msg = str(exc).lower()
        return any(s in msg for s in ('401', '429', 'unauthorized', 'too many'))

    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            # fast_info is a lightweight endpoint — much less likely to 401
            cap = getattr(ticker.fast_info, 'market_cap', None)
            if not isinstance(cap, (int, float)):
                # fast_info returned None or a non-numeric (e.g. mock) —
                # fall back to the full info dict
                _log.debug("market cap: fast_info unavailable for %s — falling back to ticker.info", symbol)
                cap = ticker.info.get('marketCap') or 0
            return symbol if (cap or 0) >= min_cap else None
        except Exception as exc:
            if attempt < 2:
                wait = (30.0 if _is_rl(exc) else 0.5) * (2 ** attempt)
                time.sleep(wait)
    return None


# Throughput controls for _filter_by_market_cap
# fast_info is ~5× lighter than ticker.info so we can safely use more workers
# and a shorter inter-chunk pause without triggering Yahoo's rate limiter.
_MCAP_MAX_WORKERS  = 10   # concurrent yfinance threads
_MCAP_CHUNK_SIZE   = 200  # symbols per worker-pool batch
_MCAP_CHUNK_PAUSE  = 1.5  # seconds between consecutive batch chunks


def _filter_by_market_cap(symbols, min_cap, max_workers=_MCAP_MAX_WORKERS):
    """
    Parallel, chunked market-cap filter.

    Symbols are processed in chunks of _MCAP_CHUNK_SIZE.  After each chunk
    a brief pause prevents thundering-herd requests against yfinance's
    backend.  Returns a sorted list of qualifying symbols.
    """
    qualified = []
    for i in range(0, max(len(symbols), 1), _MCAP_CHUNK_SIZE):
        chunk = symbols[i : i + _MCAP_CHUNK_SIZE]
        if not chunk:
            break
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_check_market_cap, sym, min_cap): sym
                       for sym in chunk}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    qualified.append(result)
        # Pause between chunks to avoid overwhelming yfinance (skip after last)
        if i + _MCAP_CHUNK_SIZE < len(symbols):
            time.sleep(_MCAP_CHUNK_PAUSE)
    return sorted(qualified)


def get_ticker_universe(min_market_cap, force_refresh=False, log=None):
    """
    Return all NASDAQ/NYSE stock tickers with market cap >= min_market_cap.

    Results are cached in data/universe_cache.json for 48 hours.
    Pass force_refresh=True to bypass the cache and rebuild.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        expires_at = datetime.fromisoformat(cache['cached_at']) + timedelta(hours=CACHE_TTL_HOURS)
        if datetime.now() < expires_at:
            tickers = cache['tickers']
            if log:
                log.info(f"Universe loaded from cache: {len(tickers)} tickers (expires {expires_at.strftime('%Y-%m-%d %H:%M')})")
            return tickers

    if log:
        log.info("Fetching full NASDAQ/NYSE symbol lists from nasdaqtrader.com...")

    nasdaq_syms = _fetch_nasdaq_symbols()
    other_syms  = _fetch_other_symbols()
    all_symbols = sorted(set(nasdaq_syms + other_syms))

    if log:
        log.info(f"Candidate symbols: {len(all_symbols)}. Filtering by market cap >= ${min_market_cap:,.0f} (parallel)...")

    qualified = _filter_by_market_cap(all_symbols, min_market_cap)

    os.makedirs('data', exist_ok=True)
    with open(CACHE_FILE, 'w') as f:
        json.dump({'cached_at': datetime.now().isoformat(), 'tickers': qualified}, f, indent=2)

    if log:
        log.info(f"Universe built: {len(qualified)} tickers. Cached to {CACHE_FILE}.")

    return qualified
