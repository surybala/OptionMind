"""
AlpacaDataClient
================
Drop-in replacement for yfinance data calls in scanner.py,
position_monitor.py, and dashboard.py.

Provides
--------
get_bulk_history(symbols, days=35)   -> dict[str, pd.Series]
    Replaces: yf.download(tickers, period='30d')

get_spot_price(symbol)               -> float | None
get_spot_prices(symbols)             -> dict[str, float]
    Replaces: fast_info.last_price / _cached_hist.iloc[-1]

get_option_chain(symbol, expiry)     -> OptionChain | None
    Returns an object with .puts and .calls DataFrames (same column
    names as yfinance): strike, bid, ask, lastPrice,
    impliedVolatility, openInterest.
    Replaces: yf.Ticker(symbol).option_chain(expiry)

Uses alpaca-py's StockHistoricalDataClient and
OptionHistoricalDataClient (both authenticated — no 401/429 from
Yahoo Finance).  Falls back silently if credentials are absent or the
API call fails; the caller then uses yfinance as a fallback.

Notes
-----
* Open interest is not provided in the alpaca-py OptionsSnapshot
  model; openInterest is set to -1 ("unknown") so that the
  min_open_interest filter skips the check rather than rejecting
  all Alpaca-sourced picks.
* Historical bars are fetched in chunks of ≤ 100 tickers to stay
  well within Alpaca's bulk-request limits.
* The module creates the underlying HTTP clients lazily so that importing this
  module never fails even when alpaca-py is not installed.
"""
from __future__ import annotations

import logging
import re
import time
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

_log = logging.getLogger('optionwheel')

# ── Public namedtuple mirrors yfinance's option_chain return value ─────────────
OptionChain = namedtuple('OptionChain', ['puts', 'calls'])

_CHAIN_COLS = [
    'strike', 'bid', 'ask', 'lastPrice', 'volume', 'impliedVolatility', 'openInterest',
    # Greeks sourced directly from Alpaca (None when snapshot has no greeks field)
    'delta', 'gamma', 'theta', 'vega', 'rho',
]

# OSI option symbol: ROOT YYMMDD C|P STRIKE(8 digits, /1000 = dollars)
_OSI_RE = re.compile(r'^([A-Z./]+)(\d{6})([CP])(\d{8})$')

# Class-share symbols: our universe uses dash notation (LEN-B, BRK-B) sourced
# from Wikipedia via universe_indices._clean().  Alpaca's stock data endpoints
# expect dot notation (LEN.B, BRK.B).  These helpers translate between the two
# so the rest of the codebase never needs to know about the difference.
_DASH_CLASS_RE = re.compile(r'^([A-Z0-9]+)-([A-Z])$')


def _to_alpaca_sym(s: str) -> str:
    """LEN-B → LEN.B  (no-op for normal symbols)."""
    m = _DASH_CLASS_RE.match(s)
    return f"{m.group(1)}.{m.group(2)}" if m else s


def _parse_osi(symbol: str) -> Optional[tuple[str, float, str]]:
    """
    Parse an OSI option symbol into (option_type, strike_dollars, expiry_iso).

    Example: 'AAPL240119C00200000' -> ('call', 200.0, '2024-01-19')
    Returns None if the symbol does not match the expected format.
    """
    m = _OSI_RE.match(symbol)
    if not m:
        return None
    opt_type = 'call' if m.group(3) == 'C' else 'put'
    strike   = int(m.group(4)) / 1000.0

    # Parse YYMMDD expiry from the symbol
    d = m.group(2)                              # e.g. '240119'
    yy, mm, dd = int(d[0:2]), int(d[2:4]), int(d[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy  # handles Y2.1K: valid through 2069
    expiry_iso = f"{year:04d}-{mm:02d}-{dd:02d}"

    return opt_type, strike, expiry_iso


def _snapshot_to_row(symbol: str, snapshot) -> Optional[dict]:
    """
    Convert an alpaca-py OptionsSnapshot to a dict matching the
    yfinance option-chain DataFrame row format.
    """
    parsed = _parse_osi(symbol)
    if parsed is None:
        return None
    opt_type, strike, _expiry = parsed

    q = getattr(snapshot, 'latest_quote', None)
    bid  = float(getattr(q, 'bid_price', 0) or 0) if q else 0.0
    ask  = float(getattr(q, 'ask_price', 0) or 0) if q else 0.0

    t = getattr(snapshot, 'latest_trade', None)
    last = float(getattr(t, 'price', 0) or 0) if t else 0.0
    if last == 0.0 and bid > 0 and ask > 0:
        last = (bid + ask) / 2.0

    iv = float(getattr(snapshot, 'implied_volatility', 0) or 0)

    # open_interest is not in OptionsSnapshot — use -1 to signal
    # "unknown" so the min_open_interest filter skips this option.
    oi_raw = getattr(snapshot, 'open_interest', None)
    oi = int(oi_raw) if oi_raw is not None else -1

    # Greeks: available natively from Alpaca (alpaca-py >= 0.43.2).
    # Returns None per greek if the snapshot has no greeks object, so that
    # callers can distinguish "broker provided 0.0" from "not available".
    greeks = getattr(snapshot, 'greeks', None)
    def _g(attr: str) -> Optional[float]:
        if greeks is None:
            return None
        v = getattr(greeks, attr, None)
        return float(v) if v is not None else None

    return {
        '_opt_type':         opt_type,
        'strike':            strike,
        'bid':               bid,
        'ask':               ask,
        'lastPrice':         last,
        'volume':            0,    # Alpaca snapshots don't expose daily volume
        'impliedVolatility': iv,
        'openInterest':      oi,
        'delta':             _g('delta'),
        'gamma':             _g('gamma'),
        'theta':             _g('theta'),
        'vega':              _g('vega'),
        'rho':               _g('rho'),
    }


def _alpaca_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call ``fn()``; on exception retry with exponential back-off.

    Raises ``RuntimeError`` after all retries are exhausted — intentionally
    no silent fallback.  This is the contract for HFT mode: prefer a loud
    error over degraded data from a non-broker source.

    Delays (seconds): base_delay, 2×base_delay, 4×base_delay, …
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Alpaca call failed after {max_retries} retr{'y' if max_retries == 1 else 'ies'}: {exc}"
                ) from exc
            delay = base_delay * (2 ** attempt)
            _log.warning(
                "alpaca_retry: attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt + 1, max_retries, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)


class AlpacaDataClient:
    """
    Thin wrapper around alpaca-py's data clients providing helpers
    used by the scanner, position monitor, and dashboard.
    """

    def __init__(self, api_key: str, api_secret: str) -> None:
        from alpaca.data.historical import (
            OptionHistoricalDataClient,
            StockHistoricalDataClient,
        )
        self._stock  = StockHistoricalDataClient(api_key, api_secret)
        self._option = OptionHistoricalDataClient(api_key, api_secret)

    # ── Spot price ─────────────────────────────────────────────────────────────

    def get_spot_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Return the latest bar close price for each symbol.
        Uses StockLatestBarRequest — single bulk call.
        """
        if not symbols:
            return {}
        try:
            from alpaca.data.requests import StockLatestBarRequest
            # Alpaca uses dot notation (LEN.B); our universe uses dash (LEN-B).
            alpaca_to_orig = {_to_alpaca_sym(s): s for s in symbols}
            req  = StockLatestBarRequest(symbol_or_symbols=list(alpaca_to_orig.keys()))
            bars = self._stock.get_stock_latest_bar(req)
            result: dict[str, float] = {}
            for sym, bar in (bars.items() if hasattr(bars, 'items') else bars.data.items()):
                orig = alpaca_to_orig.get(sym, sym)
                c = getattr(bar, 'close', None)
                if c is not None:
                    result[orig] = float(c)
            return result
        except Exception:
            return {}

    def get_spot_price(self, symbol: str) -> Optional[float]:
        return self.get_spot_prices([symbol]).get(symbol)

    # ── Historical bars (for HV30 + batch price cache) ────────────────────────

    def get_bulk_history(
        self,
        symbols: list[str],
        days: int = 35,
    ) -> dict[str, pd.Series]:
        """
        Fetch *days* of daily close bars for all symbols.

        Returns a dict  symbol -> pd.Series(close prices, newest last)
        compatible with the ``_HIST_CACHE`` used by scanner.py.
        Splits into chunks of 100 to stay within Alpaca limits.
        """
        if not symbols:
            return {}
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            return {}

        start = datetime.now(timezone.utc) - timedelta(days=days + 7)
        result: dict[str, pd.Series] = {}
        chunk_size = 100

        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            # Alpaca uses dot notation (LEN.B) for class-share suffixes while
            # our universe stores dash notation (LEN-B).  Convert before sending.
            alpaca_to_orig = {_to_alpaca_sym(s): s for s in chunk}
            if not alpaca_to_orig:
                continue
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=list(alpaca_to_orig.keys()),
                    timeframe=TimeFrame.Day,
                    start=start,
                    # NOTE: do NOT set limit here — for multi-symbol requests
                    # Alpaca applies limit as the total number of bars across
                    # all symbols, so limit=45 with 100 tickers yields ~0 bars
                    # per ticker.  The start date already bounds the window.
                )
                bar_set = self._stock.get_stock_bars(req)
                data = bar_set.data if hasattr(bar_set, 'data') else bar_set
                for sym, bars in (data.items() if hasattr(data, 'items') else {}.items()):
                    closes = pd.Series([float(b.close) for b in bars])
                    if len(closes) >= 5:
                        # Map back from Alpaca dot-notation to original dash-notation
                        result[alpaca_to_orig.get(sym, sym)] = closes
            except Exception as _chunk_exc:
                # chunk failed — caller falls back to yfinance for these tickers
                import logging as _logging
                _logging.getLogger('optionwheel').warning(
                    "get_bulk_history: chunk %d-%d failed (%s: %s)",
                    i, i + chunk_size, type(_chunk_exc).__name__, _chunk_exc,
                )

        return result

    # ── Option chains ──────────────────────────────────────────────────────────

    def get_option_chain(
        self,
        symbol: str,
        expiry: str,
    ) -> Optional[OptionChain]:
        """
        Fetch the full option chain for *symbol* expiring on *expiry*
        (ISO-format date string, e.g. '2025-01-17').

        Returns an OptionChain namedtuple whose .puts and .calls are
        DataFrames with the same columns as yfinance option_chain().
        Returns None if the request fails or returns no data.
        """
        try:
            from alpaca.data.requests import OptionChainRequest
            from datetime import date as _date
            try:
                from alpaca.data.enums import OptionFeed
                _feed = OptionFeed.INDICATIVE
            except Exception:
                _feed = 'indicative'
            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date=_date.fromisoformat(expiry),
                feed=_feed,
            )
            chain = self._option.get_option_chain(req)
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger('optionwheel').warning(
                "get_option_chain: Alpaca request failed for %s/%s (%s: %s)",
                symbol, expiry, type(_exc).__name__, _exc,
            )
            return None

        if not chain:
            return None

        data = chain.data if hasattr(chain, 'data') else chain

        puts_rows: list[dict] = []
        calls_rows: list[dict] = []

        items = data.items() if hasattr(data, 'items') else chain.items()
        for opt_symbol, snapshot in items:
            row = _snapshot_to_row(opt_symbol, snapshot)
            if row is None:
                continue
            opt_type = row.pop('_opt_type')
            if opt_type == 'put':
                puts_rows.append(row)
            else:
                calls_rows.append(row)

        if not puts_rows and not calls_rows:
            return None

        # Guard: pd.DataFrame([]).sort_values('strike') raises KeyError when the
        # list is empty because the resulting DataFrame has no columns at all.
        # Use get_option_chains_for_range's pattern: fall back to a typed empty DF.
        puts_df = (
            pd.DataFrame(puts_rows)
            .sort_values('strike')
            .reset_index(drop=True)
            [_CHAIN_COLS]
            if puts_rows else pd.DataFrame(columns=_CHAIN_COLS)
        )
        calls_df = (
            pd.DataFrame(calls_rows)
            .sort_values('strike')
            .reset_index(drop=True)
            [_CHAIN_COLS]
            if calls_rows else pd.DataFrame(columns=_CHAIN_COLS)
        )

        return OptionChain(puts=puts_df, calls=calls_df)


    def get_option_chains_for_range(
        self,
        symbol: str,
        start_date,          # date or str  'YYYY-MM-DD'
        end_date,            # date or str  'YYYY-MM-DD'
    ) -> dict[str, OptionChain]:
        """
        Fetch ALL option contracts for *symbol* whose expiry falls in
        [start_date, end_date] in a **single API call**.

        Returns a dict  expiry_iso_str -> OptionChain(puts_df, calls_df).

        This replaces two yfinance calls per ticker:
          1. ``ticker.options``          — the expiry list
          2. ``ticker.option_chain(e)``  — per-expiry chain

        With this method the scanner makes exactly ONE Alpaca request per
        ticker (instead of 1 yfinance + 1-2 Alpaca requests) and receives
        all expiries at once, keyed by date for O(1) lookup.
        """
        from datetime import date as _date

        # Normalise to date objects
        if isinstance(start_date, str):
            start_date = _date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = _date.fromisoformat(end_date)

        try:
            from alpaca.data.requests import OptionChainRequest
            try:
                from alpaca.data.enums import OptionFeed
                _feed = OptionFeed.INDICATIVE
            except Exception:
                _feed = 'indicative'
            req = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date_gte=start_date,
                expiration_date_lte=end_date,
                feed=_feed,
            )
            chain = self._option.get_option_chain(req)
        except Exception as _exc:
            import logging as _logging
            _logging.getLogger('optionwheel').warning(
                "get_option_chains_for_range: Alpaca request failed for %s (%s: %s)",
                symbol, type(_exc).__name__, _exc,
            )
            return {}

        if not chain:
            return {}

        data = chain.data if hasattr(chain, 'data') else chain

        # Group snapshots by expiry date
        # key: expiry_iso  ->  {'puts': [...], 'calls': [...]}
        buckets: dict[str, dict[str, list]] = {}

        items = data.items() if hasattr(data, 'items') else chain.items()
        for opt_symbol, snapshot in items:
            parsed = _parse_osi(opt_symbol)
            if parsed is None:
                continue
            opt_type, strike, expiry_iso = parsed

            q   = getattr(snapshot, 'latest_quote', None)
            bid = float(getattr(q, 'bid_price', 0) or 0) if q else 0.0
            ask = float(getattr(q, 'ask_price', 0) or 0) if q else 0.0
            t   = getattr(snapshot, 'latest_trade', None)
            last = float(getattr(t, 'price', 0) or 0) if t else 0.0
            if last == 0.0 and bid > 0 and ask > 0:
                last = (bid + ask) / 2.0
            iv  = float(getattr(snapshot, 'implied_volatility', 0) or 0)
            oi_raw = getattr(snapshot, 'open_interest', None)
            oi  = int(oi_raw) if oi_raw is not None else -1

            # Greeks from broker (None if not provided)
            _greeks = getattr(snapshot, 'greeks', None)
            def _rg(attr: str) -> Optional[float]:
                if _greeks is None:
                    return None
                v = getattr(_greeks, attr, None)
                return float(v) if v is not None else None

            row = {
                'strike':            strike,
                'bid':               bid,
                'ask':               ask,
                'lastPrice':         last,
                'volume':            0,    # Alpaca range-chain snapshots don't expose daily volume
                'impliedVolatility': iv,
                'openInterest':      oi,
                'delta':             _rg('delta'),
                'gamma':             _rg('gamma'),
                'theta':             _rg('theta'),
                'vega':              _rg('vega'),
                'rho':               _rg('rho'),
            }

            bucket = buckets.setdefault(expiry_iso, {'puts': [], 'calls': []})
            bucket['puts' if opt_type == 'put' else 'calls'].append(row)

        # Build DataFrames for each expiry
        result: dict[str, OptionChain] = {}
        for expiry_iso, sides in buckets.items():
            puts_df  = (pd.DataFrame(sides['puts'])
                        .sort_values('strike')
                        .reset_index(drop=True)
                        [_CHAIN_COLS] if sides['puts'] else
                        pd.DataFrame(columns=_CHAIN_COLS))
            calls_df = (pd.DataFrame(sides['calls'])
                        .sort_values('strike')
                        .reset_index(drop=True)
                        [_CHAIN_COLS] if sides['calls'] else
                        pd.DataFrame(columns=_CHAIN_COLS))
            result[expiry_iso] = OptionChain(puts=puts_df, calls=calls_df)

        return result


    # ── HFT helpers (strict: raise on failure, no yfinance fallback) ──────────

    def get_option_snapshots(
        self,
        osi_symbols: list[str],
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> dict[str, dict]:
        """
        Fetch current snapshots for specific OSI option symbols.

        More efficient than ``get_option_chain()`` when monitoring a small
        set of open position legs — fetches only the contracts you hold.

        Returns dict[osi_symbol → row_dict] where each row has the same
        keys as ``_snapshot_to_row()`` (bid, ask, impliedVolatility,
        delta, gamma, theta, vega, rho, …).

        Raises ``RuntimeError`` after ``max_retries`` exhausted — no
        silent fallback (HFT contract: prefer loud errors over stale data).
        """
        if not osi_symbols:
            return {}

        def _fetch():
            from alpaca.data.requests import OptionSnapshotRequest
            req  = OptionSnapshotRequest(symbol_or_symbols=osi_symbols)
            raw  = self._option.get_option_snapshot(req)
            data = raw.data if hasattr(raw, 'data') else raw
            result: dict[str, dict] = {}
            items = data.items() if hasattr(data, 'items') else {}
            for sym, snap in items:
                row = _snapshot_to_row(sym, snap)
                if row is not None:
                    row.pop('_opt_type', None)
                    result[sym] = row
            if not result:
                raise RuntimeError(
                    f"get_option_snapshots: no data returned for {osi_symbols}"
                )
            return result

        return _alpaca_retry(_fetch, max_retries, base_delay)

    def get_spot_price_strict(
        self,
        symbol: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> float:
        """
        Return the latest bar close price for ``symbol``.

        Raises ``RuntimeError`` if the price cannot be fetched after
        ``max_retries`` attempts — no silent None return in HFT mode.
        """
        def _fetch():
            prices = self.get_spot_prices([symbol])
            if not prices or symbol not in prices:
                raise RuntimeError(f"No spot price returned for {symbol}")
            return prices[symbol]

        return _alpaca_retry(_fetch, max_retries, base_delay)

    def get_option_chain_strict(
        self,
        symbol: str,
        expiry: str,
        max_retries: int = 3,
        base_delay: float = 2.0,
    ) -> 'OptionChain':
        """
        Fetch the full option chain for ``symbol``/``expiry``.

        Raises ``RuntimeError`` after ``max_retries`` exhausted — no
        silent None return in HFT mode.
        """
        def _fetch():
            chain = self.get_option_chain(symbol, expiry)
            if chain is None:
                raise RuntimeError(
                    f"get_option_chain returned no data for {symbol}/{expiry}"
                )
            return chain

        return _alpaca_retry(_fetch, max_retries, base_delay)


# ── Factory ────────────────────────────────────────────────────────────────────

# Module-level cache: keyed by (api_key, api_secret) so the same credentials
# always return the same client instance without rebuilding the HTTP session on
# every call (important in the monitor daemon which calls this every 15 min).
_CLIENT_CACHE: dict[tuple[str, str], "AlpacaDataClient"] = {}


def make_alpaca_data_client(config: dict) -> Optional[AlpacaDataClient]:
    """
    Return a cached AlpacaDataClient built from env vars or config.
    Returns None if credentials are missing or alpaca-py is not installed.
    """
    import os
    api_key    = (os.environ.get('ALPACA_API_KEY')
                  or config.get('alpaca', {}).get('api_key', ''))
    api_secret = (os.environ.get('ALPACA_API_SECRET')
                  or config.get('alpaca', {}).get('api_secret', ''))
    if not api_key or not api_secret:
        return None
    cache_key = (api_key, api_secret)
    if cache_key not in _CLIENT_CACHE:
        try:
            _CLIENT_CACHE[cache_key] = AlpacaDataClient(api_key, api_secret)
        except Exception:
            return None
    return _CLIENT_CACHE[cache_key]
