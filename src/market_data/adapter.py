"""
market_data.adapter
===================

``DataAdapter`` — single class that consolidates all HFT / non-HFT branching
previously scattered through ``scanner.py`` and ``position_monitor.py``.

Design notes
------------
* ``client_getter`` and ``hft_mode_getter`` are **callables** (typically
  ``lambda: _ALPACA_CLIENT`` / ``lambda: _HFT_MODE``) so that
  ``@patch('src.scanner._ALPACA_CLIENT', None)`` in tests continues to
  work even after the ``OptionScanner`` (and its adapter) have already been
  constructed.

* This module **never** imports ``src.scanner`` at module level.  The
  lambdas close over the scanner's namespace, keeping the dependency
  graph acyclic.

* ``get_position_chain`` is the main entry point for ``PositionMonitor``; the
  scanner uses the lower-level helpers (``get_hft_spot``, ``get_history_df``,
  ``fetch_alpaca_chains``, ``fetch_chain_fallback``).
"""
from __future__ import annotations

import dataclasses
import logging
import time
from typing import Callable, Optional

import pandas as pd
import yfinance as yf

from .base import PositionChainResult

_log = logging.getLogger('optionwheel')


@dataclasses.dataclass
class HftPrefetch:
    """
    Pre-fetched HFT data for an entire monitoring cycle — populated once
    before the position loop by :meth:`DataAdapter.batch_prefetch_hft`.

    Collapses N×2 per-position Alpaca calls into exactly 2 bulk calls:
      - one ``OptionSnapshotRequest`` for all OSI legs across all positions
      - one ``StockLatestBarRequest`` for all unique underlying symbols
    """
    snapshots:     dict  # osi_symbol → row_dict
    spot_prices:   dict  # symbol     → float
    osi_maps:      dict  # pos_id     → {(strike, opt_type): osi_symbol}
    leg_specs_map: dict  # pos_id     → [(strike, opt_type, position_side)]


class DataAdapter:
    """
    Isolates HFT (Alpaca) vs non-HFT (yfinance) data-source selection.

    Parameters
    ----------
    hist_cache :
        Shared ``_HIST_CACHE`` dict (same object, not a copy).
    chain_cache :
        Shared ``_CHAIN_CACHE`` dict (same object, not a copy).
    client_getter :
        Zero-argument callable that returns the current Alpaca client (or
        ``None``).  Read at call time so patching the module-level name in
        tests takes effect.
    hft_mode_getter :
        Zero-argument callable that returns the current ``_HFT_MODE`` bool.
    hft_config :
        Dict with HFT-specific settings (``max_retries``,
        ``retry_base_delay_seconds``).
    """

    def __init__(
        self,
        hist_cache: dict,
        chain_cache: dict,
        client_getter: Callable,
        hft_mode_getter: Callable,
        hft_config: dict,
    ) -> None:
        self._hist      = hist_cache
        self._chain     = chain_cache
        self._get_client  = client_getter
        self._get_hft     = hft_mode_getter
        self._hft_cfg     = hft_config
        # Session-level cache of tickers Alpaca confirmed have no options.
        # Skipped immediately on future calls — no retry, no API hit.
        self._no_options_cache: set = set()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> 'DataAdapter':
        """
        Build a :class:`DataAdapter` from an application config dict.

        Reads ``hft_mode`` and ``hft`` keys; creates an Alpaca data client
        via :func:`~src.alpaca_data.make_alpaca_data_client` (returns ``None``
        when credentials are absent — non-HFT paths still work).

        Intended for callers that don't already hold a DataAdapter (e.g. the
        Flask dashboard).  :class:`~src.position_monitor.PositionMonitor`
        continues to construct the adapter directly so it can share caches.
        """
        from src.alpaca_data import make_alpaca_data_client
        client   = make_alpaca_data_client(config)
        hft_mode = bool(config.get('hft_mode', False))
        hft_cfg  = dict(config.get('hft', {}))
        return cls(
            hist_cache      = {},
            chain_cache     = {},
            client_getter   = lambda: client,
            hft_mode_getter = lambda: hft_mode,
            hft_config      = hft_cfg,
        )

    # ── Public helpers ────────────────────────────────────────────────────────

    def is_hft(self) -> bool:
        """Return the current HFT mode flag (read lazily)."""
        return bool(self._get_hft())

    def _client(self):
        """Return the current Alpaca client (read lazily)."""
        return self._get_client()

    def _max_retries(self) -> int:
        return int(self._hft_cfg.get('max_retries', 3))

    def _base_delay(self) -> float:
        return float(self._hft_cfg.get('retry_base_delay_seconds', 2.0))

    # ── Unified spot / historical helpers ────────────────────────────────────

    def get_spot_price(self, symbol: str) -> Optional[float]:
        """
        Return the current spot price for *symbol*.

        * **HFT**: Alpaca ``get_spot_price`` (no yfinance).
        * **Non-HFT**: Alpaca first, yfinance fallback.

        Returns ``None`` when no price can be determined.
        """
        client = self._client()
        if client is not None:
            try:
                price = client.get_spot_price(symbol)
                if price is not None and price > 0:
                    return price
            except Exception:
                pass
            if self._get_hft():
                return None

        # Non-HFT fallback
        try:
            info = yf.Ticker(symbol).fast_info
            for attr in ('last_price', 'regularMarketPrice'):
                raw = getattr(info, attr, None)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass
        return None

    def get_spot_prices(self, symbols: list[str]) -> dict[str, float]:
        """
        Return spot prices for multiple symbols in a single call where possible.

        * **HFT**: Alpaca bulk ``get_spot_prices`` only.
        * **Non-HFT**: Alpaca first, yfinance per-symbol fallback for misses.
        """
        result: dict[str, float] = {}
        client = self._client()
        if client is not None:
            try:
                result = client.get_spot_prices(symbols)
            except Exception:
                pass

        missing = [s for s in symbols if s not in result]
        if missing and not self._get_hft():
            for sym in missing:
                try:
                    info = yf.Ticker(sym).fast_info
                    for attr in ('last_price', 'regularMarketPrice'):
                        raw = getattr(info, attr, None)
                        if raw is None:
                            continue
                        v = float(raw)
                        if v > 0:
                            result[sym] = v
                            break
                except Exception:
                    pass
        return result

    def get_historical_close(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[float]:
        """
        Return the closing price nearest to *end_date* within the window.

        * **HFT**: Alpaca historical bars only.
        * **Non-HFT**: Alpaca first, yfinance fallback.

        Returns ``None`` when no price can be determined.
        """
        client = self._client()
        if client is not None:
            try:
                from datetime import date as _date
                from alpaca.data.requests import StockBarsRequest
                from alpaca.data.timeframe import TimeFrame
                req = StockBarsRequest(
                    symbol_or_symbols=[symbol],
                    timeframe=TimeFrame.Day,
                    start=_date.fromisoformat(start_date),
                    end=_date.fromisoformat(end_date),
                )
                bar_set = client._stock.get_stock_bars(req)
                data = bar_set.data if hasattr(bar_set, 'data') else bar_set
                items = data.get(symbol, []) if hasattr(data, 'get') else []
                if not items:
                    items = data.items() if hasattr(data, 'items') else []
                    for sym, bars in items:
                        if sym == symbol and bars:
                            return float(bars[-1].close)
                    if self._get_hft():
                        return None
                else:
                    return float(items[-1].close)
            except Exception:
                if self._get_hft():
                    return None

        # Non-HFT fallback
        try:
            hist = yf.Ticker(symbol).history(start=start_date, end=end_date)
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return None

    # ── Scanner helpers ───────────────────────────────────────────────────────

    def get_hft_spot(self, symbol: str) -> float:
        """
        Fetch the spot price for *symbol* via Alpaca (HFT path only).

        Raises
        ------
        RuntimeError
            If the Alpaca call fails after all retries.  The caller
            (``scan_ticker``) should catch this and return ``[]``.
        """
        client = self._client()
        if client is None:
            raise RuntimeError(
                "HFT mode: Alpaca client is None — cannot fetch spot price"
            )
        return client.get_spot_price_strict(
            symbol,
            max_retries=self._max_retries(),
            base_delay=self._base_delay(),
        )

    def get_history_df(
        self,
        symbol: str,
        ticker,
        cached_hist,
    ) -> pd.DataFrame:
        """
        Return a history DataFrame for ATR / HV30 computation.

        * **HFT**: wraps ``cached_hist`` (Alpaca Close-only pd.Series) into a
          one-column DataFrame.  If ``cached_hist`` is absent, returns an empty
          DataFrame (ATR will be 0).
        * **Non-HFT**: same wrapping when the batch cache is available; falls
          back to ``ticker.history(period='30d')`` otherwise.

        The returned DataFrame always has at least a ``'Close'`` column when
        non-empty.  Full OHLCV columns (``'High'``, ``'Low'``, ``'Close'``)
        are present only on the yfinance direct-fetch path.
        """
        if self._get_hft():
            if cached_hist is not None and len(cached_hist) >= 5:
                return pd.DataFrame({'Close': cached_hist})
            return pd.DataFrame()

        # Non-HFT
        try:
            if cached_hist is not None and len(cached_hist) >= 5:
                return pd.DataFrame({'Close': cached_hist})
            return _yf_retry(lambda: ticker.history(period='30d'))
        except Exception:
            return pd.DataFrame()

    def fetch_alpaca_chains(
        self,
        symbol: str,
        today,
        target_date,
    ) -> Optional[dict]:
        """
        Fetch all option chains for *symbol* in the ``[today, target_date]``
        window via Alpaca.

        Returns
        -------
        dict
            ``{expiry_str: chain}`` — non-empty means Alpaca returned data.
        {} (empty dict)
            Alpaca is unavailable or returned nothing (non-HFT can fall back
            to yfinance expirations).
        None
            **HFT only** — Alpaca failed after all retries; caller must abort
            this ticker (return ``[]``).
        """
        client = self._client()
        if client is None:
            if self._get_hft():
                # HFT without a client should have been caught at startup, but
                # guard here anyway.
                _log.error(
                    "HFT scan: Alpaca client is None for %s — skipping ticker", symbol
                )
                return None
            return {}

        hft = self._get_hft()
        max_r  = self._max_retries()
        base_d = self._base_delay()

        if hft:
            # Fast-path: skip tickers confirmed no-options during this session.
            if symbol in self._no_options_cache:
                _log.debug("HFT scan: %s has no options on Alpaca (cached) — skipping", symbol)
                return None

            # HFT: retry only on transient API errors (Exception), not empty responses.
            # An empty chain means Alpaca has no options for this ticker — retrying
            # won't help and wastes 14 s per ticker at max_retries=3/base_delay=2.
            for attempt in range(max_r + 1):
                try:
                    chains = (
                        client.get_option_chains_for_range(
                            symbol,
                            today.date() if hasattr(today, 'date') else today,
                            target_date.date() if hasattr(target_date, 'date') else target_date,
                        ) or {}
                    )
                    if chains:
                        return chains
                    # Empty is a definitive answer — Alpaca has no options for this
                    # ticker.  Cache it for the rest of the session and skip now.
                    self._no_options_cache.add(symbol)
                    _log.debug(
                        "HFT scan: Alpaca returned no options for %s — skipping (cached)",
                        symbol,
                    )
                    return None
                except Exception as exc:
                    if attempt == max_r:
                        _log.error(
                            "HFT scan: chain fetch exhausted for %s — skipping ticker: %s",
                            symbol, exc,
                        )
                        return None
                    _delay = base_d * (2 ** attempt)
                    _log.warning(
                        "HFT scan: chain retry %d/%d for %s in %.1fs (%s)",
                        attempt + 1, max_r, symbol, _delay, type(exc).__name__,
                    )
                    time.sleep(_delay)
            # Should not reach here but guard anyway
            return None
        else:
            # Non-HFT: single attempt; failure is fine (fall back to yfinance)
            try:
                return (
                    client.get_option_chains_for_range(
                        symbol,
                        today.date() if hasattr(today, 'date') else today,
                        target_date.date() if hasattr(target_date, 'date') else target_date,
                    ) or {}
                )
            except Exception:
                return {}

    def fetch_chain_fallback(self, symbol: str, expiry: str, ticker) -> Optional[object]:
        """
        Per-expiry fallback when Alpaca and the session cache both miss.

        * **HFT**: logs an error and returns ``None`` — no yfinance allowed.
        * **Non-HFT**: calls ``ticker.option_chain(expiry)`` via yfinance;
          returns ``None`` on failure.

        The caller should ``continue`` (skip this expiry) when ``None`` is
        returned.
        """
        if self._get_hft():
            _log.error(
                "HFT scan: no chain for %s/%s after Alpaca range fetch — skipping expiry",
                symbol, expiry,
            )
            return None

        # Non-HFT: yfinance fallback
        _log.debug(
            "option chain: Alpaca missing %s/%s — falling back to yfinance",
            symbol, expiry,
        )
        try:
            return _yf_retry(lambda e=expiry: ticker.option_chain(e))
        except Exception:
            return None

    # ── Position monitor helper ───────────────────────────────────────────────

    def get_position_chain(
        self,
        pos: dict,
        get_leg_specs_fn,
        build_osi_fn,
        prefetch: Optional[HftPrefetch] = None,
    ) -> PositionChainResult:
        """
        Fetch chain data for a single open position.

        Dispatches to the HFT (Alpaca snapshots) or non-HFT (yfinance chain)
        path depending on the current mode flag.

        Parameters
        ----------
        pos :
            Position dict from the database.
        get_leg_specs_fn :
            Bound method ``PositionMonitor._get_position_leg_specs`` — returns
            ``[(strike, opt_type, position_side), …]``.
        build_osi_fn :
            Bound static method ``PositionMonitor._build_osi_symbol``.

        Returns
        -------
        PositionChainResult
            ``put_map`` / ``call_map`` are ``None`` on non-HFT failure.

        Raises
        ------
        RuntimeError
            **HFT only** — on Alpaca failure after retries.  The caller
            (``PositionMonitor._check_position``) should let this propagate to
            ``_run_loop`` which catches and logs it.
        """
        if self._get_hft():
            if prefetch is not None:
                return self._get_position_chain_from_prefetch(
                    pos, get_leg_specs_fn, build_osi_fn, prefetch
                )
            return self._fetch_position_chain_hft(pos, get_leg_specs_fn, build_osi_fn)
        return self._fetch_position_chain_non_hft(pos)

    # ── Batched HFT pre-fetch (2 calls for all positions) ─────────────────────

    def batch_prefetch_hft(
        self,
        positions: list,
        get_leg_specs_fn,
        build_osi_fn,
    ) -> HftPrefetch:
        """
        Pre-fetch option snapshots + spot prices for *all* positions in exactly
        2 Alpaca calls — one ``OptionSnapshotRequest`` and one
        ``StockLatestBarRequest`` — instead of 2 × N sequential calls.

        Call this once at the start of each monitoring cycle (before the
        position loop) and pass the returned :class:`HftPrefetch` into
        :meth:`get_position_chain` via the *prefetch* keyword.

        Raises ``RuntimeError`` on Alpaca snapshot failure (consistent with
        the per-position HFT contract).  Spot-price failures are logged as
        warnings; individual positions will raise at check time when spot is
        missing.
        """
        client = self._client()
        if client is None:
            raise RuntimeError("HFT batch prefetch: Alpaca client is None")

        max_r  = self._max_retries()
        base_d = self._base_delay()

        # ── Step 1: Compute leg specs + OSI maps (CPU only, no API) ──────────
        leg_specs_map: dict = {}
        osi_maps: dict      = {}
        all_osis: list      = []
        for pos in positions:
            pos_id    = pos.get('id')
            leg_specs = get_leg_specs_fn(pos)
            leg_specs_map[pos_id] = leg_specs
            osi_map: dict = {}
            for strike, opt_type, _ in leg_specs:
                osi = build_osi_fn(pos['symbol'], pos['expiry'], strike, opt_type)
                osi_map[(strike, opt_type)] = osi
                all_osis.append(osi)
            osi_maps[pos_id] = osi_map

        # Deduplicate OSIs (positions may share legs) while preserving order
        seen: set       = set()
        unique_osis     = [o for o in all_osis if not (o in seen or seen.add(o))]

        # ── Step 2: ONE OptionSnapshotRequest for all legs ───────────────────
        # get_option_snapshots already wraps in _alpaca_retry internally.
        snapshots = client.get_option_snapshots(
            unique_osis, max_retries=max_r, base_delay=base_d
        )

        # ── Step 3: ONE StockLatestBarRequest for all unique underlyings ─────
        unique_symbols = list({pos['symbol'] for pos in positions})
        spot_prices: dict = {}
        for attempt in range(max_r + 1):
            spot_prices = client.get_spot_prices(unique_symbols)
            if spot_prices:
                break
            if attempt < max_r:
                delay = base_d * (2 ** attempt)
                _log.warning(
                    "[HFT] batch spot fetch attempt %d/%d empty — retrying in %.1fs",
                    attempt + 1, max_r, delay,
                )
                time.sleep(delay)

        missing_spots = [s for s in unique_symbols if s not in spot_prices]
        if missing_spots:
            _log.warning(
                "[HFT] batch prefetch: spot price unavailable for %s — "
                "affected positions will be skipped this cycle",
                missing_spots,
            )

        # Warn once if every snapshot came back with None greeks — this typically
        # means the market is closed or Alpaca hasn't priced these contracts yet.
        # Gamma-risk checks will be skipped for all positions this cycle; stop-loss
        # (price-based) remains active.
        if snapshots:
            no_greeks_count = sum(
                1 for row in snapshots.values()
                if row.get('delta') is None
            )
            if no_greeks_count == len(snapshots):
                _log.warning(
                    "[HFT] batch prefetch: ALL %d snapshots have no broker greeks "
                    "(market closed or Alpaca algo-trader tier required) — "
                    "gamma-risk checks disabled this cycle; stop-loss still active",
                    len(snapshots),
                )
            elif no_greeks_count > 0:
                _log.warning(
                    "[HFT] batch prefetch: %d/%d snapshots have no broker greeks — "
                    "gamma-risk check will be partial this cycle",
                    no_greeks_count, len(snapshots),
                )

        return HftPrefetch(
            snapshots=snapshots,
            spot_prices=spot_prices,
            osi_maps=osi_maps,
            leg_specs_map=leg_specs_map,
        )

    def _get_position_chain_from_prefetch(
        self,
        pos: dict,
        get_leg_specs_fn,
        build_osi_fn,
        prefetch: HftPrefetch,
    ) -> PositionChainResult:
        """
        Build a :class:`PositionChainResult` from pre-fetched data — no API
        calls.  Called by :meth:`get_position_chain` when *prefetch* is set.
        """
        pos_id    = pos.get('id')
        symbol    = pos['symbol']
        leg_specs = prefetch.leg_specs_map.get(pos_id) or get_leg_specs_fn(pos)
        osi_map   = prefetch.osi_maps.get(pos_id) or {}

        if not leg_specs:
            raise RuntimeError(
                f"HFT prefetch: no leg specs for position {pos_id} "
                f"(strategy={pos.get('type')}) — cannot build OSI symbols"
            )

        put_map:  dict = {}
        call_map: dict = {}
        osi_list: list = []
        for strike, opt_type, _ in leg_specs:
            osi = osi_map.get((strike, opt_type))
            if osi is None:
                continue
            osi_list.append(osi)
            row = prefetch.snapshots.get(osi)
            if row is None:
                _log.warning(
                    "[HFT] prefetch: snapshot missing for %s (pos %s)", osi, pos_id
                )
                continue
            if opt_type == 'put':
                put_map[strike] = row
            else:
                call_map[strike] = row

        spot = prefetch.spot_prices.get(symbol)
        if spot is None:
            raise RuntimeError(
                f"HFT prefetch: no spot price for {symbol} (pos {pos_id})"
            )

        return PositionChainResult(
            spot=spot,
            put_map=put_map,
            call_map=call_map,
            has_broker_greeks=True,
            snapshots=prefetch.snapshots,
            osi_map=osi_map,
            leg_specs=leg_specs,
        )

    # ── Internal HFT position chain ───────────────────────────────────────────

    def _fetch_position_chain_hft(
        self, pos: dict, get_leg_specs_fn, build_osi_fn
    ) -> PositionChainResult:
        """
        HFT variant — Alpaca snapshots for the specific OSI contracts only.

        Raises ``RuntimeError`` on Alpaca failure (no yfinance fallback).
        """
        client = self._client()
        if client is None:
            raise RuntimeError(
                "HFT mode requires Alpaca credentials "
                "(set ALPACA_API_KEY + ALPACA_API_SECRET)"
            )

        max_r  = self._max_retries()
        base_d = self._base_delay()

        symbol    = pos['symbol']
        expiry    = pos['expiry']
        leg_specs = get_leg_specs_fn(pos)

        if not leg_specs:
            raise RuntimeError(
                f"No leg specs for position {pos.get('id')} "
                f"(strategy={pos.get('type')}) — cannot build OSI symbols"
            )

        # Build OSI map: (strike, opt_type) → osi_symbol
        osi_map: dict = {}
        for strike, opt_type, _ in leg_specs:
            osi = build_osi_fn(symbol, expiry, strike, opt_type)
            osi_map[(strike, opt_type)] = osi

        osi_list = list(osi_map.values())

        # Fetch snapshots — raises RuntimeError on exhaustion
        snapshots = client.get_option_snapshots(
            osi_list, max_retries=max_r, base_delay=base_d
        )

        # Build put_map / call_map keyed by strike
        put_map:  dict = {}
        call_map: dict = {}
        for (strike, opt_type, _), osi in zip(leg_specs, osi_list):
            row = snapshots.get(osi)
            if row is None:
                _log.warning(
                    "[HFT] snapshot missing for %s (pos %s)", osi, pos.get('id')
                )
                continue
            if opt_type == 'put':
                put_map[strike] = row
            else:
                call_map[strike] = row

        # Spot price — raises RuntimeError on exhaustion
        spot = client.get_spot_price_strict(
            symbol, max_retries=max_r, base_delay=base_d
        )

        return PositionChainResult(
            spot=spot,
            put_map=put_map,
            call_map=call_map,
            has_broker_greeks=True,
            snapshots=snapshots,
            osi_map=osi_map,
            leg_specs=leg_specs,
        )

    # ── Internal non-HFT position chain ──────────────────────────────────────

    def _fetch_position_chain_non_hft(self, pos: dict) -> PositionChainResult:
        """
        Non-HFT variant — Alpaca chain first, yfinance fallback.

        Returns ``PositionChainResult(put_map=None, …)`` on failure.
        """
        symbol = pos['symbol']
        expiry = pos['expiry']
        _failure = PositionChainResult(
            spot=None, put_map=None, call_map=None, has_broker_greeks=False
        )

        try:
            chain = None
            # Try Alpaca first (client_getter may be make_alpaca_data_client(config)
            # for position_monitor, or the module-level _ALPACA_CLIENT for scanner)
            try:
                _alpaca = self._client()
                if _alpaca is not None:
                    chain = _alpaca.get_option_chain(symbol, expiry)
            except Exception:
                pass

            # Fall back to yfinance
            if chain is None:
                _log.warning(
                    "chain fetch: Alpaca returned no chain for %s/%s — falling back to yfinance",
                    symbol, expiry,
                )
                ticker = yf.Ticker(symbol)
                chain  = ticker.option_chain(expiry)

            put_map  = {row['strike']: row for _, row in chain.puts.iterrows()}
            call_map = {row['strike']: row for _, row in chain.calls.iterrows()}
        except Exception:
            return _failure

        # Fetch spot price (best-effort)
        spot: Optional[float] = None
        try:
            info = yf.Ticker(symbol).fast_info
            for attr in ('last_price', 'regularMarketPrice'):
                raw = getattr(info, attr, None)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                    if v > 0:
                        spot = v
                        break
                except (TypeError, ValueError):
                    continue
        except Exception:
            pass

        return PositionChainResult(
            spot=spot,
            put_map=put_map,
            call_map=call_map,
            has_broker_greeks=False,
        )


# ── Module-private yfinance retry helper ─────────────────────────────────────

_NORMAL_DELAYS = (0.5, 1.0, 2.0)
_RL_DELAYS     = (15.0, 45.0, 90.0)


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(s in msg for s in ('401', '429', 'unauthorized', 'too many requests')):
        return True
    resp = getattr(exc, 'response', None)
    return resp is not None and getattr(resp, 'status_code', 0) in (401, 429)


def _yf_retry(fn, retries: int = 3):
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                delays = _RL_DELAYS if _is_rate_limited(exc) else _NORMAL_DELAYS
                time.sleep(delays[attempt])
    raise last_exc  # type: ignore[misc]
