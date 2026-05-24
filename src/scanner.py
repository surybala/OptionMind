"""Legacy deterministic scanner.

This module is retained for historical reference and legacy tests only.
The app-facing scanner path is ``src.model_scanner.ModelScanner``. Do not add
new candidate-generation work here unless explicitly migrating legacy behavior.
"""

import logging
import math
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

_log = logging.getLogger('optionwheel')

# ── yfinance rate-limiting ────────────────────────────────────────────────────
# A BoundedSemaphore caps the number of threads that can be actively fetching
# from yfinance at the same time.  Combined with ThreadPoolExecutor in
# get_top_picks() this keeps us well under yfinance's informal rate limit
# without introducing mandatory per-request sleeps.
#
# Raised from 5 → 10: safe because the batch prefetch below removes the
# expensive ticker.history(30d) calls from inside the semaphore, cutting
# per-ticker hold time roughly in half and leaving headroom for higher
# concurrency on the remaining option-chain calls.
_MAX_CONCURRENT_SCANS = 10
_SCAN_SEMAPHORE       = threading.BoundedSemaphore(_MAX_CONCURRENT_SCANS)

# Retry delays (seconds) for individual yfinance calls inside scan_ticker.
# Normal errors:      0.5 s → 1.0 s → 2.0 s
# Rate-limit (401/429): 15 s → 45 s → 90 s
_NORMAL_DELAYS = (0.5, 1.0, 2.0)
_RL_DELAYS     = (15.0, 45.0, 90.0)

# ── Batch-prefetch caches ─────────────────────────────────────────────────────
# _HIST_CACHE  — populated by _batch_prefetch_history() before the parallel
#                scan so scan_ticker() can skip individual history(30d) calls.
#                Key: ticker symbol.  Value: Close price pd.Series.
# _CHAIN_CACHE — session-scoped option-chain cache (TTL = 5 min).  Saves
#                re-fetching on re-scans / retries within the same run.
#                Key: (symbol, expiry_str).  Value: (timestamp, chain).

_HIST_LOCK  = threading.Lock()
_HIST_CACHE: dict[str, pd.Series] = {}

_CHAIN_LOCK  = threading.Lock()
_CHAIN_CACHE: dict[tuple, tuple]  = {}
_CHAIN_TTL   = 300   # seconds — option IV moves slowly; 5 min reuse is safe

# ── Earnings calendar cache ────────────────────────────────────────────────────
# Earnings dates change at most weekly; a 24 h TTL avoids per-ticker yfinance
# calls inside the scan semaphore in HFT mode.
# Key: ticker symbol.  Value: (timestamp, frozenset_of_dates).
_EARNINGS_LOCK:  threading.Lock          = threading.Lock()
_EARNINGS_CACHE: dict[str, tuple]        = {}
_EARNINGS_TTL   = 86_400   # seconds (24 h)

# ── Optional Alpaca data client ───────────────────────────────────────────────
# Set by OptionScanner.__init__ when valid Alpaca credentials are available.
# When set, option chains and historical bars are fetched via Alpaca instead
# of yfinance — eliminating Yahoo Finance 401/429 rate-limit errors.
# Remains None when credentials are absent; all code paths fall back to yfinance.
_ALPACA_CLIENT = None   # type: Optional['src.alpaca_data.AlpacaDataClient']

# ── HFT mode ─────────────────────────────────────────────────────────────────
# Set by OptionScanner.__init__ from config.hft_mode.
# When True: no yfinance anywhere — retry Alpaca instead; use broker greeks
# for probability calculation instead of computing Black-Scholes N(d2).
_HFT_MODE:   bool = False
_HFT_CONFIG: dict = {}


def _batch_prefetch_history(tickers: list[str], period: str = '30d') -> None:
    """
    Pre-populate ``_HIST_CACHE`` with 30-day close-price history for all
    tickers before the parallel scan begins.

    Strategy (in priority order):
    1. **Alpaca** — when credentials are configured, fetches authenticated
       bulk bars via ``AlpacaDataClient.get_bulk_history()``.  No Yahoo
       Finance 401/429 issues; supports up to 100 tickers per request.
    2. **yfinance** — single ``yf.download()`` call as a fallback.  Can
       return misaligned / split-adjusted data for large batches (the price
       cross-validation in ``scan_ticker`` catches the most common cases).

    Falls back silently: any failure leaves ``_HIST_CACHE`` partially or
    fully empty; ``scan_ticker`` then uses its own individual fallbacks.
    """
    if not tickers:
        return

    # ── Try Alpaca first (authenticated, no rate-limit issues) ────────────────
    if _ALPACA_CLIENT is not None:
        try:
            days = int(period.rstrip('d')) + 5 if period.endswith('d') else 40
            hist = _ALPACA_CLIENT.get_bulk_history(tickers, days=days)
            if hist:
                with _HIST_LOCK:
                    _HIST_CACHE.update(hist)
                coverage = len(hist) / max(len(tickers), 1)
                if _HFT_MODE:
                    # HFT: no yfinance fallback — Alpaca is the only source.
                    # Any coverage is acceptable; partial history is fine for HV30.
                    if coverage < 0.20:
                        raise RuntimeError(
                            f"HFT mode: Alpaca bulk history returned only "
                            f"{coverage:.0%} coverage ({len(hist)}/{len(tickers)} tickers)"
                        )
                    return
                # Non-HFT: skip yfinance if coverage is good enough
                if coverage >= 0.80:
                    return
                # Partial coverage — fall through to yfinance for missing tickers.
                _log.warning(
                    "batch history: Alpaca coverage %.0f%% (<%d tickers) — "
                    "falling back to yfinance for missing symbols",
                    coverage * 100, len(tickers),
                )
        except Exception as _exc:
            if _HFT_MODE:
                raise RuntimeError(
                    f"HFT mode: Alpaca bulk history failed — not falling back to yfinance "
                    f"({type(_exc).__name__}: {_exc})"
                ) from _exc
            _log.warning(
                "batch history: Alpaca fetch failed (%s: %s) — falling back to yfinance",
                type(_exc).__name__, _exc,
            )
    elif _HFT_MODE:
        raise RuntimeError(
            "HFT mode: Alpaca client not available — cannot fetch history"
        )

    if _HFT_MODE:
        # Should not reach here but guard anyway
        return

    # ── yfinance bulk download fallback ───────────────────────────────────────
    try:
        raw = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            return

        # yf.download returns a MultiIndex DataFrame for multiple tickers:
        #   columns = (price_field, symbol)  e.g. ('Close', 'AAPL')
        # For a single ticker it returns flat columns ('Open', 'High', …).
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw['Close']          # DataFrame: dates × symbols
        else:
            # Single ticker — wrap into one-column DataFrame
            sym = tickers[0]
            close_df = raw[['Close']].rename(columns={'Close': sym})

        with _HIST_LOCK:
            for sym in close_df.columns:
                series = close_df[sym].dropna()
                if len(series) >= 5:
                    _HIST_CACHE[sym] = series
    except Exception:
        pass  # silent — scan_ticker falls back to individual calls


def _get_cached_chain(symbol: str, expiry: str):
    """Return a cached option chain or None if absent / expired."""
    with _CHAIN_LOCK:
        entry = _CHAIN_CACHE.get((symbol, expiry))
    if entry is None:
        return None
    ts, chain = entry
    return chain if (time.time() - ts) <= _CHAIN_TTL else None


def _put_cached_chain(symbol: str, expiry: str, chain) -> None:
    with _CHAIN_LOCK:
        _CHAIN_CACHE[(symbol, expiry)] = (time.time(), chain)


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if exc looks like an HTTP 401 or 429 from Yahoo Finance."""
    msg = str(exc).lower()
    if any(s in msg for s in ('401', '429', 'unauthorized', 'too many requests')):
        return True
    resp = getattr(exc, 'response', None)
    return resp is not None and getattr(resp, 'status_code', 0) in (401, 429)


def _yf_retry(fn, retries: int = 3):
    """
    Call fn() up to `retries` times.

    Normal transient errors back off 0.5 s / 1.0 s / 2.0 s.
    HTTP 401 / 429 (Yahoo rate-limit) back off 15 s / 45 s / 90 s so the
    caller avoids compounding the rate-limit by hammering the endpoint.
    """
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


def _row_oi_vol(row) -> tuple[int | None, int]:
    """
    Extract (open_interest, volume) from an option chain row.
    Returns (None, 0) when OI is unknown (-1 from Alpaca snapshots).
    """
    raw_oi  = row.get('openInterest') if hasattr(row, 'get') else getattr(row, 'openInterest', None)
    raw_vol = row.get('volume')       if hasattr(row, 'get') else getattr(row, 'volume',       None)
    oi  = None if (raw_oi  is None or raw_oi  < 0) else int(raw_oi)
    vol = 0    if (raw_vol is None or raw_vol < 0) else int(raw_vol)
    return oi, vol


class OptionScanner:
    def __init__(self, config):
        self.config = config
        self.min_market_cap = config.get('market_cap_min', 1e9)
        self.max_expiry_days = config.get('expiry_days_max', 14)

        # Load risk parameters
        risk_params = config.get('risk_parameters', {})
        self.min_prob = risk_params.get('min_probability_of_expiry', 0.8)

        # Load strategy parameters
        strategies = config.get('strategies', {})
        self.csp_params      = strategies.get('covered_put', {})
        self.pcs_params      = strategies.get('put_credit_spread', {})
        self.ccs_params      = strategies.get('call_credit_spread', {})
        self.ic_params       = strategies.get('iron_condor', {})
        self.ifly_params     = strategies.get('iron_butterfly', {})
        self.strangle_params = strategies.get('short_strangle', {})
        self.cc_params       = strategies.get('covered_call', {})

        # Chain-level liquidity pre-filter (applied once per expiry, before
        # any strategy scanner sees the data).  All three thresholds can be
        # overridden via config["chain_liquidity"]; set a value to 0 to disable
        # that individual check.
        liq = config.get('chain_liquidity', {})
        self._liq_min_bid    = float(liq.get('min_bid',            0.05))
        self._liq_min_oi     = int(  liq.get('min_open_interest',  10))
        self._liq_max_spread = float(liq.get('max_spread_pct',     0.80))

        # Minimum % OTM distance from spot (applied to short leg before delta check)
        _otm_cfg = config.get('min_otm_pct', {})
        self._min_otm_put  = float(_otm_cfg.get('put',  0.0))
        self._min_otm_call = float(_otm_cfg.get('call', 0.0))

        # ATR-based distance guard (require |spot - strike| >= multiplier × ATR(period))
        _atr_cfg = config.get('atr_distance', {})
        self._atr_enabled    = bool(_atr_cfg.get('enabled',    False))
        self._atr_period     = int( _atr_cfg.get('atr_period', 14))
        self._atr_multiplier = float(_atr_cfg.get('multiplier', 1.5))

        # Earnings exclusion: skip expiries that fall within an earnings window
        _earn_cfg = config.get('earnings_exclusion', {})
        self._earnings_enabled = bool(_earn_cfg.get('enabled',     False))
        self._earnings_buffer  = int( _earn_cfg.get('days_buffer', 2))

        # IV quality filters — gate the entire ticker scan on vol conditions
        _iv_cfg = config.get('iv_filters', {})
        self._iv_filters_enabled    = bool( _iv_cfg.get('enabled',            False))
        self._min_iv_rank           = float(_iv_cfg.get('min_iv_rank',         0.30))
        self._require_iv_premium    = bool( _iv_cfg.get('require_iv_premium', True))
        self._iv_premium_min_ratio  = float(_iv_cfg.get('iv_premium_min_ratio', 1.0))
        # History lookback for IV rank — needs 252 days to cover the rolling year
        self._iv_rank_history_days  = int(  _iv_cfg.get('history_days',        252))

        # Optional SentimentAnalyzer — injected by agent.py when enabled.
        # When None the scanner uses static config deltas with no adjustment.
        self.sentiment_analyzer = None

        # Initialise Alpaca data client if credentials are available.
        # This replaces yfinance for option chains and historical bars,
        # eliminating Yahoo Finance rate-limit (401/429) errors entirely.
        global _ALPACA_CLIENT
        try:
            from src.alpaca_data import make_alpaca_data_client
            _ALPACA_CLIENT = make_alpaca_data_client(config)
        except Exception:
            _ALPACA_CLIENT = None
        if _ALPACA_CLIENT is not None:
            _log.info("Alpaca data client initialised (scanner: bars + option chains).")
        else:
            _log.info("Alpaca data client unavailable — scanner will use yfinance.")

        # HFT mode — set module-level flags used by free functions
        global _HFT_MODE, _HFT_CONFIG
        _HFT_MODE   = bool(config.get('hft_mode', False))
        _HFT_CONFIG = dict(config.get('hft', {}))
        if _HFT_MODE:
            if _ALPACA_CLIENT is None:
                raise RuntimeError(
                    "hft_mode=true requires valid Alpaca credentials "
                    "(ALPACA_API_KEY + ALPACA_API_SECRET must be set)"
                )
            _log.info(
                "HFT mode ENABLED — scanner uses Alpaca exclusively "
                "(no yfinance fallback, broker greeks for probability)"
            )

        # VIX filter — current VIX level injected by agent.py at startup.
        # When set, IC scans are skipped if VIX >= ic_pause_threshold.
        self.current_vix: float | None = None
        vix_cfg = risk_params.get('vix_filter', {})
        self.vix_filter_enabled: bool  = bool(vix_cfg.get('enabled', False))
        self.vix_ic_pause_threshold: float = float(vix_cfg.get('ic_pause_threshold', 20.0))

        # ── Market data adapter (HFT / non-HFT isolation) ────────────────────────
        from src.market_data import DataAdapter
        self._data = DataAdapter(
            hist_cache      = _HIST_CACHE,
            chain_cache     = _CHAIN_CACHE,
            client_getter   = lambda: _ALPACA_CLIENT,
            hft_mode_getter = lambda: _HFT_MODE,
            hft_config      = _HFT_CONFIG,
        )

        # ── Strategy scanner instances ────────────────────────────────────────────
        from src.scan_strategies import (CspScanner, SpreadsScanner, IronCondorScanner,
                                          IronButterflyScanner, StrangleScanner, CoveredCallScanner)
        _common = dict(
            min_prob=self.min_prob,
            min_otm_put=self._min_otm_put,
            min_otm_call=self._min_otm_call,
            atr_enabled=self._atr_enabled,
            atr_multiplier=self._atr_multiplier,
            prob_otm_fn=self._prob_otm,
            row_oi_vol_fn=_row_oi_vol,
            dynamic_width_cfg=config.get('dynamic_width', {}),
        )
        self._csp_scanner      = CspScanner(self.csp_params,     **_common)
        self._spreads_scanner  = SpreadsScanner(self.pcs_params, self.ccs_params, **_common)
        self._ic_scanner       = IronCondorScanner(self.ic_params,    **_common)
        self._ifly_scanner     = IronButterflyScanner(self.ifly_params, **_common)
        self._strangle_scanner = StrangleScanner(self.strangle_params, **_common)
        self._cc_scanner       = CoveredCallScanner(self.cc_params,   **_common)

    # ── Probability engine ────────────────────────────────────────────────────

    def _apply_liquidity_filter(self, df: 'pd.DataFrame') -> 'pd.DataFrame':
        from src.scan_filters.liquidity import apply_liquidity_filter
        return apply_liquidity_filter(df, self._liq_min_bid, self._liq_min_oi,
                                      self._liq_max_spread)

    def get_probability_of_expiry(self, current_price, strike, iv, days_to_expiry, option_type='put'):
        """
        Estimate probability of expiring OTM using standard normal distribution.
        """
        if iv <= 0 or days_to_expiry <= 0:
            if option_type == 'put':
                return 0.99 if strike < current_price else 0.01
            else:
                return 0.99 if strike > current_price else 0.01

        # Time in years
        t = days_to_expiry / 365.0
        # Standard deviation move
        sigma = iv * math.sqrt(t)
        # d2 term from Black-Scholes
        d2 = (math.log(current_price / strike) + (0 - 0.5 * iv**2) * t) / (iv * math.sqrt(t))

        # Standard Normal CDF
        def normal_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        if option_type == 'put':
            # Probability OTM for Put (S_T > K) is N(d2)
            return normal_cdf(d2)
        else:
            # Probability OTM for Call (S_T < K) is N(-d2)
            return normal_cdf(-d2)

    def _prob_otm(
        self,
        row,
        current_price: float,
        strike: float,
        days: int,
        option_type: str,
    ) -> float:
        """
        Return P(option expires OTM).

        **HFT mode**: uses the broker-supplied delta from *row* (``row['delta']``).
        Because delta ≈ P(ITM) for both puts and calls, P(OTM) = 1 − |delta|.
        This avoids a local Black-Scholes N(d₂) call and uses the broker's own
        volatility surface, including skew and smile effects.

        **Non-HFT mode**: falls back to the existing Black-Scholes N(d₂) path
        via ``get_probability_of_expiry()``.
        """
        if _HFT_MODE:
            raw_delta = (
                row.get('delta') if hasattr(row, 'get') else getattr(row, 'delta', None)
            )
            if raw_delta is not None:
                return max(0.0, min(1.0, 1.0 - abs(float(raw_delta))))
            # Delta missing despite HFT mode — log WARNING and fall through to B-S
            _log.warning(
                "_prob_otm: broker delta missing for strike %.2f %s — falling back to B-S N(d2)",
                strike, option_type,
            )
        iv = (
            row.get('impliedVolatility') if hasattr(row, 'get')
            else getattr(row, 'impliedVolatility', 0)
        )
        return self.get_probability_of_expiry(
            current_price, strike, float(iv or 0), days, option_type
        )

    # ── Top-level scanner ─────────────────────────────────────────────────────

    def scan_ticker(self, ticker_symbol):
        """
        Fetch live option chain data for ticker_symbol and run all enabled
        strategies.  Each yfinance call is wrapped in _yf_retry (up to 3
        attempts with backoff) and the whole function runs under
        _SCAN_SEMAPHORE so at most _MAX_CONCURRENT_SCANS threads hit
        yfinance at the same time.

        Batch-prefetch fast-path
        ------------------------
        If get_top_picks() called _batch_prefetch_history() beforehand,
        ``_HIST_CACHE`` already contains the 30-day close series for this
        ticker.  In that case:
          • current_price  is read from the last row of the cache (no network)
          • HV30 history   is read from the cache   (no network)
        Only the option-chain calls (``ticker.options`` + per-expiry
        ``ticker.option_chain()``) still hit yfinance inside the semaphore,
        cutting per-ticker semaphore hold-time roughly in half.

        If self.sentiment_analyzer is set, per-ticker sentiment is computed
        once and used to adjust max_delta for PCS / CCS / IC legs:
          BULL → raise PCS delta ceiling, lower CCS delta ceiling
          BEAR → lower PCS delta ceiling, raise CCS delta ceiling
        """
        results = []
        market_cap_val: float | None = None
        try:
            # ── Phase 1: Read batch-prefetched history (zero network cost) ────
            with _HIST_LOCK:
                _cached_hist = _HIST_CACHE.get(ticker_symbol)

            current_price: float | None = self._read_price_from_cache(_cached_hist)

            # HFT mode: get spot price from Alpaca directly if not yet in cache.
            # Done outside the semaphore — Alpaca is authenticated and unlimited.
            if self._data.is_hft() and (current_price is None or current_price <= 0):
                try:
                    current_price = self._data.get_hft_spot(ticker_symbol)
                except RuntimeError as _exc:
                    _log.error(
                        "HFT scan: spot price failed for %s — skipping ticker: %s",
                        ticker_symbol, _exc,
                    )
                    return []

            with _SCAN_SEMAPHORE:
                if self._data.is_hft():
                    ticker = None
                    if not current_price:
                        return []
                else:
                    ticker = yf.Ticker(ticker_symbol)

                # ── Phase 2: Price confirmation + market-cap gate (non-HFT) ──
                if not self._data.is_hft():
                    current_price, market_cap_val = self._resolve_price_and_cap(
                        ticker_symbol, ticker, _cached_hist, current_price
                    )
                    if not current_price:
                        return []

                if not current_price:
                    return []

                # ── Phase 3: Sentiment-adjusted delta limits ──────────────────
                (sentiment,
                 pcs_delta_adj, ccs_delta_adj,
                 ic_put_delta_adj, ic_call_delta_adj) = self._compute_sentiment_deltas(
                    ticker_symbol
                )

                # ── Phase 4: History / ATR / HV30 / HV rank ─────────────────
                _hist_df = self._data.get_history_df(ticker_symbol, ticker, _cached_hist)
                (_ticker_atr, _hv30, _hv_rank,
                 _ic_vix_blocked,
                 _put_w_override,
                 _call_w_override) = self._compute_atr_and_hv30(_hist_df)

                # ── Phase 4b: IV rank gate ────────────────────────────────────
                if self._iv_filters_enabled and self._min_iv_rank > 0:
                    if _hv_rank < self._min_iv_rank:
                        _log.debug(
                            "[iv_filter] %s skipped: HV rank=%.2f < min=%.2f",
                            ticker_symbol, _hv_rank, self._min_iv_rank,
                        )
                        return []

                # ── Phase 5: Expiry list, Alpaca chains, earnings calendar ────
                today       = datetime.now()
                target_date = today + timedelta(days=self.max_expiry_days)
                _alpaca_chains, expirations, _earnings_dates = (
                    self._fetch_expirations_and_chains(
                        ticker_symbol, ticker, today, target_date
                    )
                )
                if _alpaca_chains is None:
                    return []

                # ── Phase 5b: IV premium gate (IV > HV30) ─────────────────────
                if self._iv_filters_enabled and self._require_iv_premium and _hv30 > 0:
                    _chain_iv = self._compute_atm_iv(
                        _alpaca_chains, expirations, current_price
                    )
                    if _chain_iv is not None:
                        if _chain_iv < _hv30 * self._iv_premium_min_ratio:
                            _log.debug(
                                "[iv_filter] %s skipped: ATM IV=%.3f < HV30=%.3f × %.1f",
                                ticker_symbol, _chain_iv, _hv30, self._iv_premium_min_ratio,
                            )
                            return []

                # ── Phase 6: Per-expiry strategy scan ────────────────────────
                for expiry in expirations:
                    results.extend(self._scan_one_expiry(
                        ticker_symbol, expiry, current_price,
                        today, target_date,
                        _alpaca_chains, _earnings_dates, ticker,
                        pcs_delta_adj, ccs_delta_adj,
                        ic_put_delta_adj, ic_call_delta_adj,
                        sentiment, _ticker_atr,
                        _ic_vix_blocked, _put_w_override, _call_w_override,
                    ))

            if market_cap_val is not None:
                for r in results:
                    r.setdefault('market_cap', market_cap_val)
            return results
        except Exception:
            return []

    # ── scan_ticker private helpers ───────────────────────────────────────────

    def _read_price_from_cache(self, cached_hist) -> 'float | None':
        """Return the last Close from the batch-prefetch cache, or None."""
        if cached_hist is not None and len(cached_hist) >= 1:
            try:
                return float(cached_hist.iloc[-1])
            except Exception:
                pass
        return None

    def _resolve_price_and_cap(
        self,
        ticker_symbol: str,
        ticker,
        cached_hist,
        current_price: 'float | None',
    ) -> 'tuple[float | None, float | None]':
        """
        (Non-HFT) Confirm / update *current_price* and read *market_cap*
        via yfinance ``fast_info`` / ``info``.

        Returns ``(price, market_cap)``.  Either value may be ``None``; a
        ``None`` price signals that the ticker should be skipped.
        """
        market_cap_val: 'float | None' = None
        skip_mcap = self.config.get('skip_market_cap_check_in_scanner', False)

        alpaca_price_available = (
            _ALPACA_CLIENT is not None
            and ticker_symbol in _HIST_CACHE
            and current_price is not None
            and current_price > 0
        )
        needs_fast_info = (
            not skip_mcap                             # market-cap check
            or current_price is None                  # no price yet
            or (not alpaca_price_available            # yfinance batch → validate
                and current_price > 0)
        )
        if needs_fast_info:
            try:
                fast = _yf_retry(lambda: ticker.fast_info)
                if not skip_mcap:
                    cap = getattr(fast, 'market_cap', None)
                    if isinstance(cap, (int, float)) and cap < self.min_market_cap:
                        return None, None   # caller returns []
                    if isinstance(cap, (int, float)) and cap > 0:
                        market_cap_val = cap

                raw_price = getattr(fast, 'last_price', None)
                if not isinstance(raw_price, (int, float)) or raw_price <= 0:
                    raw_price = getattr(fast, 'regular_market_previous_close', None)
                fi_price = raw_price if isinstance(raw_price, (int, float)) and raw_price > 0 else None

                if current_price is None or current_price <= 0:
                    current_price = fi_price
                elif fi_price is not None and fi_price > 0 and not alpaca_price_available:
                    lo = min(current_price, fi_price)
                    hi = max(current_price, fi_price)
                    if hi / lo > 2.0:
                        current_price = fi_price
            except Exception:
                pass

        if not current_price:
            try:
                info = _yf_retry(lambda: ticker.info)
                if not skip_mcap and info.get('marketCap', 0) < self.min_market_cap:
                    return None, None
                if market_cap_val is None:
                    mc = info.get('marketCap')
                    if isinstance(mc, (int, float)) and mc > 0:
                        market_cap_val = mc
                current_price = (info.get('currentPrice')
                                 or info.get('regularMarketPrice'))
            except Exception:
                try:
                    hist = _yf_retry(lambda: ticker.history(period='1d'))
                    if hist.empty:
                        return None, None
                    current_price = float(hist['Close'].iloc[-1])
                except Exception:
                    return None, None

        return current_price, market_cap_val

    def _compute_sentiment_deltas(self, ticker_symbol: str) -> tuple:
        """
        Run sentiment analysis and return adjusted delta limits.

        Returns ``(sentiment, pcs_delta, ccs_delta, ic_put_delta, ic_call_delta)``.
        """
        sa = self.sentiment_analyzer
        sentiment         = None
        pcs_delta_adj     = self.pcs_params.get('max_delta_short_leg', 0.30)
        ccs_delta_adj     = self.ccs_params.get('max_delta_short_leg', 0.30)
        ic_put_delta_adj  = self.ic_params.get('max_delta_short_leg', 0.25)
        ic_call_delta_adj = self.ic_params.get('max_delta_short_leg', 0.25)

        if sa is not None:
            sentiment         = sa.analyze(ticker_symbol)
            pcs_delta_adj     = sa.adjust_delta(pcs_delta_adj,     'PCS', sentiment)
            ccs_delta_adj     = sa.adjust_delta(ccs_delta_adj,     'CCS', sentiment)
            ic_put_delta_adj, ic_call_delta_adj = sa.adjust_ic_deltas(
                ic_put_delta_adj, ic_call_delta_adj, sentiment
            )

        return sentiment, pcs_delta_adj, ccs_delta_adj, ic_put_delta_adj, ic_call_delta_adj

    def _compute_atr_and_hv30(self, hist_df) -> tuple:
        """
        Compute ATR(14), HV30, HV rank, VIX block flag, and IC wing-width overrides.

        Returns ``(atr, hv30, hv_rank, ic_vix_blocked, put_width_override, call_width_override)``.

        *hv30*   — annualised 30-day realised volatility (0.25 = 25%).
        *hv_rank* — percentile rank of current HV30 within the full history
                    window (0.0–1.0).  Requires ≥60 rows for a meaningful
                    estimate; returns 0.5 (neutral) when insufficient data.
        """
        _ticker_atr = 0.0
        if self._atr_enabled:
            try:
                n = self._atr_period
                if hist_df is not None and not hist_df.empty:
                    if all(c in hist_df.columns for c in ('High', 'Low', 'Close')):
                        _prev_c = hist_df['Close'].shift(1)
                        _tr = pd.concat([
                            hist_df['High'] - hist_df['Low'],
                            (hist_df['High'] - _prev_c).abs(),
                            (hist_df['Low']  - _prev_c).abs(),
                        ], axis=1).max(axis=1)
                        _atr_s = _tr.rolling(n).mean()
                        _ticker_atr = (float(_atr_s.iloc[-1])
                                       if len(_atr_s) >= n else float(_tr.mean()))
                    else:
                        _ticker_atr = float(
                            hist_df['Close'].diff().abs().tail(n).mean() or 0.0)
            except Exception:
                _ticker_atr = 0.0

        # ── HV30 + HV rank ────────────────────────────────────────────────────
        _hv30    = 0.0
        _hv_rank = 0.5   # neutral fallback when insufficient history
        try:
            if hist_df is not None and not hist_df.empty:
                lr = np.log(hist_df['Close'] / hist_df['Close'].shift(1)).dropna()
                if len(lr) >= 5:
                    # Current HV30: annualised std of the most recent 30 log-returns
                    _hv30 = float(lr.tail(30).std() * math.sqrt(252))
                if len(lr) >= 60:
                    # Rolling HV30 over the full window → percentile rank of current value
                    rolling_hv = lr.rolling(30).std().dropna() * math.sqrt(252)
                    _min_hv = float(rolling_hv.min())
                    _max_hv = float(rolling_hv.max())
                    if _max_hv > _min_hv:
                        _hv_rank = float((_hv30 - _min_hv) / (_max_hv - _min_hv))
                        _hv_rank = max(0.0, min(1.0, _hv_rank))
        except Exception:
            pass

        _ic_vix_blocked = (
            self.vix_filter_enabled
            and self.current_vix is not None
            and self.current_vix >= self.vix_ic_pause_threshold
        )
        _put_w_override  = None
        _call_w_override = None
        if self.ic_params.get('enabled', False) and not _ic_vix_blocked:
            _high_iv_thresh = float(self.ic_params.get('high_iv_threshold', 0.40))
            try:
                if _hv30 >= _high_iv_thresh:
                    _put_w_override  = float(self.ic_params.get('wide_put_strike_width',  20))
                    _call_w_override = float(self.ic_params.get('wide_call_strike_width', 20))
            except Exception:
                pass

        return _ticker_atr, _hv30, _hv_rank, _ic_vix_blocked, _put_w_override, _call_w_override

    def _compute_atm_iv(
        self,
        alpaca_chains: dict,
        expirations: list,
        current_price: float,
        atm_band_pct: float = 0.05,
    ) -> 'float | None':
        """
        Return the median implied volatility of near-ATM options for the
        nearest available expiry.  Used by the IV premium gate to compare
        current option pricing against realised volatility (HV30).

        Looks at puts and calls within *atm_band_pct* (default 5%) of spot.
        Returns None if no IV data is available.
        """
        for expiry in expirations:
            chain = alpaca_chains.get(expiry)
            if chain is None:
                continue
            ivs = []
            lo, hi = current_price * (1 - atm_band_pct), current_price * (1 + atm_band_pct)
            for df in (chain.puts, chain.calls):
                try:
                    near = df[(df['strike'] >= lo) & (df['strike'] <= hi)]
                    for iv_val in near['impliedVolatility']:
                        v = float(iv_val)
                        if v > 0:
                            ivs.append(v)
                except Exception:
                    continue
            if ivs:
                ivs.sort()
                mid = len(ivs) // 2
                return ivs[mid] if len(ivs) % 2 else (ivs[mid - 1] + ivs[mid]) / 2
        return None

    def _fetch_expirations_and_chains(
        self,
        ticker_symbol: str,
        ticker,
        today,
        target_date,
    ) -> tuple:
        """
        Fetch Alpaca chain dict, expiration list, and earnings exclusion set.

        Returns ``(_alpaca_chains, expirations, earnings_dates)``.

        * ``_alpaca_chains is None`` — HFT Alpaca failure; caller must return ``[]``.
        * ``_alpaca_chains == {}``   — no Alpaca data; expirations from yfinance.
        * ``_alpaca_chains != {}``   — use Alpaca chains directly.
        """
        _alpaca_chains = self._data.fetch_alpaca_chains(
            ticker_symbol, today, target_date
        )
        if _alpaca_chains is None:
            return None, [], set()

        if _alpaca_chains:
            expirations = list(_alpaca_chains.keys())
        else:
            try:
                expirations = _yf_retry(lambda: ticker.options)
            except Exception:
                return {}, [], set()

        # Earnings exclusion: fetch calendar once per ticker, cached 24 h.
        # Use yfinance even in HFT mode (creates a lightweight temp Ticker).
        # Fail open: if calendar is unavailable, no expiries are blocked.
        _earnings_dates: set = set()
        if self._earnings_enabled:
            with _EARNINGS_LOCK:
                _cached_entry = _EARNINGS_CACHE.get(ticker_symbol)
                _now_ts = time.time()
                if _cached_entry is not None and (_now_ts - _cached_entry[0]) < _EARNINGS_TTL:
                    _earnings_dates = set(_cached_entry[1])
                else:
                    _fetched: set = set()
                    try:
                        _cal_ticker = ticker if ticker is not None else yf.Ticker(ticker_symbol)
                        _cal = _yf_retry(lambda: _cal_ticker.calendar)
                        _cal_dates = []
                        if isinstance(_cal, dict):
                            _cal_dates = _cal.get('Earnings Date', []) or []
                        elif _cal is not None and hasattr(_cal, 'loc'):
                            try:
                                _cal_dates = _cal.loc['Earnings Date'].dropna().tolist()
                            except KeyError:
                                pass
                        for _d in _cal_dates:
                            try:
                                _fetched.add(_d.date() if hasattr(_d, 'date') else _d)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    _EARNINGS_CACHE[ticker_symbol] = (_now_ts, frozenset(_fetched))
                    _earnings_dates = _fetched

        return _alpaca_chains, expirations, _earnings_dates

    def _scan_one_expiry(
        self,
        ticker_symbol: str,
        expiry: str,
        current_price: float,
        today,
        target_date,
        alpaca_chains: dict,
        earnings_dates: set,
        ticker,
        pcs_delta_adj: float,
        ccs_delta_adj: float,
        ic_put_delta_adj: float,
        ic_call_delta_adj: float,
        sentiment,
        ticker_atr: float,
        ic_vix_blocked: bool,
        put_w_override,
        call_w_override,
    ) -> list:
        """
        Scan all enabled strategies for a single *expiry* date.

        Returns a (possibly empty) list of candidate picks.
        """
        expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
        if not (today < expiry_dt <= target_date):
            return []
        days_to_expiry = (expiry_dt - today).days
        if days_to_expiry < 1:
            return []

        if self._earnings_enabled and earnings_dates:
            from src.scan_filters.earnings import should_skip_expiry
            if should_skip_expiry(expiry_dt, earnings_dates, self._earnings_buffer, today):
                _log.debug(
                    "earnings_exclusion: skipping %s/%s — earnings within window",
                    ticker_symbol, expiry,
                )
                return []

        chain = alpaca_chains.get(expiry) or _get_cached_chain(ticker_symbol, expiry)
        if chain is None:
            chain = self._data.fetch_chain_fallback(ticker_symbol, expiry, ticker)
        if chain is None:
            return []
        _put_cached_chain(ticker_symbol, expiry, chain)

        calls = self._apply_liquidity_filter(chain.calls)
        puts  = self._apply_liquidity_filter(chain.puts)

        results = []

        if self.csp_params.get('enabled', False):
            results.extend(self._scan_csp(
                ticker_symbol, current_price, expiry, days_to_expiry, puts,
                atr=ticker_atr))

        if self.pcs_params.get('enabled', False):
            results.extend(self._scan_spreads(
                ticker_symbol, current_price, expiry, days_to_expiry, puts, 'put',
                delta_override=pcs_delta_adj, sentiment=sentiment, atr=ticker_atr))

        if self.ccs_params.get('enabled', False):
            results.extend(self._scan_spreads(
                ticker_symbol, current_price, expiry, days_to_expiry, calls, 'call',
                delta_override=ccs_delta_adj, sentiment=sentiment, atr=ticker_atr))

        if self.ic_params.get('enabled', False) and not ic_vix_blocked:
            results.extend(self._scan_iron_condor(
                ticker_symbol, current_price, expiry, days_to_expiry, puts, calls,
                put_delta_override=ic_put_delta_adj,
                call_delta_override=ic_call_delta_adj,
                sentiment=sentiment,
                put_width_override=put_w_override,
                call_width_override=call_w_override,
                atr=ticker_atr))

        if self.ifly_params.get('enabled', False):
            results.extend(self._scan_iron_butterfly(
                ticker_symbol, current_price, expiry, days_to_expiry, puts, calls,
                atr=ticker_atr))

        if self.strangle_params.get('enabled', False):
            results.extend(self._scan_strangle(
                ticker_symbol, current_price, expiry, days_to_expiry, puts, calls,
                atr=ticker_atr))

        if self.cc_params.get('enabled', False):
            results.extend(self._scan_covered_call(
                ticker_symbol, current_price, expiry, days_to_expiry, calls,
                atr=ticker_atr))

        return results

    # ── Strategy 1: Cash Secured Put ──────────────────────────────────────────

    def _scan_csp(self, symbol, current_price, expiry, days, puts, atr=0.0):
        return self._csp_scanner.scan(symbol, current_price, expiry, days, puts, atr=atr)

    # ── Strategy 2 & 3: Put / Call Credit Spreads ─────────────────────────────

    def _scan_spreads(self, symbol, current_price, expiry, days, chain_df,
                      option_type, delta_override=None, sentiment=None, atr=0.0):
        return self._spreads_scanner.scan(symbol, current_price, expiry, days,
                                          chain_df, option_type,
                                          delta_override=delta_override,
                                          sentiment=sentiment, atr=atr)

    # ── Strategy 4: Iron Condor ───────────────────────────────────────────────

    def _scan_iron_condor(self, symbol, current_price, expiry, days, puts, calls,
                          put_delta_override=None, call_delta_override=None,
                          sentiment=None, put_width_override=None,
                          call_width_override=None, atr=0.0):
        return self._ic_scanner.scan(symbol, current_price, expiry, days, puts, calls,
                                      put_delta_override=put_delta_override,
                                      call_delta_override=call_delta_override,
                                      sentiment=sentiment,
                                      put_width_override=put_width_override,
                                      call_width_override=call_width_override,
                                      atr=atr)

    # ── Strategy 5: Iron Butterfly ────────────────────────────────────────────

    def _scan_iron_butterfly(self, symbol, current_price, expiry, days, puts, calls, atr=0.0):
        return self._ifly_scanner.scan(symbol, current_price, expiry, days, puts, calls, atr=atr)

    # ── Strategy 6: Short Strangle ────────────────────────────────────────────

    def _scan_strangle(self, symbol, current_price, expiry, days, puts, calls, atr=0.0):
        return self._strangle_scanner.scan(symbol, current_price, expiry, days, puts, calls, atr=atr)

    # ── Strategy 7: Covered Call ──────────────────────────────────────────────

    def _scan_covered_call(self, symbol, current_price, expiry, days, calls, atr=0.0):
        return self._cc_scanner.scan(symbol, current_price, expiry, days, calls, atr=atr)

    # ── Aggregator ────────────────────────────────────────────────────────────

    def get_top_picks(self, ticker_list, n=10):
        """
        Scan every ticker in parallel using a ThreadPoolExecutor.

        Worker count is intentionally larger than _MAX_CONCURRENT_SCANS so
        threads queue up smoothly behind the semaphore without starving the
        pool.  Results are collected in completion order and are
        deterministically sorted by score before truncation.

        Strategy diversity
        ------------------
        Results are distributed evenly across active strategies so a single
        strategy (e.g. CCS) cannot crowd out the others when it happens to
        score higher.  Each strategy receives floor(n / num_strategies) slots;
        any remainder slots are filled from the highest-scored picks across
        all strategies.

        IC allocation cap
        -----------------
        IC picks are further capped at floor(ic_allocation_pct × n) so condors
        cannot dominate even when they outscore spreads.

        Per-ticker cap
        --------------
        At most max_picks_per_ticker picks from the same symbol are allowed
        within each strategy's pool (reads from config key max_picks_per_ticker).
        """
        # ── Blacklist filter ──────────────────────────────────────────────────
        blacklist = {t.upper() for t in self.config.get('ticker_blacklist', [])}
        if blacklist:
            original_count = len(ticker_list)
            ticker_list = [t for t in ticker_list if t.upper() not in blacklist]
            skipped = original_count - len(ticker_list)
            if skipped:
                _log.info(
                    "[scanner] Blacklist: skipping %d ticker(s) — %s",
                    skipped,
                    sorted(blacklist),
                )

        # ── Phase 1: Batch-prefetch price history for ALL tickers ────────────
        # A single yf.download() call replaces ~N individual ticker.history()
        # calls that would otherwise each consume a semaphore slot and add
        # 0.5–1 s of network latency.  On 500 tickers this alone saves ~3 min.
        # When IV rank filtering is enabled we need a full year of history to
        # compute the rolling HV30 percentile rank.  Otherwise 30 days suffices.
        _hist_period = (f"{self._iv_rank_history_days}d"
                        if self._iv_filters_enabled else '30d')
        _batch_prefetch_history(ticker_list, period=_hist_period)

        # Warm the sentiment cache in the same way if the analyzer is active.
        if self.sentiment_analyzer is not None:
            self.sentiment_analyzer.batch_prefetch(ticker_list)

        # ── Phase 2: Parallel option-chain scan ───────────────────────────────
        all_results = []
        with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_SCANS * 2) as executor:
            futures = {executor.submit(self.scan_ticker, sym): sym
                       for sym in ticker_list}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        all_results.extend(res)
                except Exception:
                    pass   # scan_ticker already swallows exceptions; belt-and-suspenders

        if not all_results:
            return []

        # Sort globally by score (descending) — preserved inside each bucket
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)

        ic_pct         = float(self.ic_params.get('ic_allocation_pct', 1.0))
        max_ic_slots   = max(1, int(ic_pct * n))       # e.g. 0.50 * 10 = 5
        max_per_ticker = self.config.get('max_picks_per_ticker')  # None = unlimited

        # -- Step 1: build per-strategy pools, honouring per-ticker + IC caps --
        # Each pool holds at most n picks (IC: max_ic_slots), with at most
        # max_picks_per_ticker from any single symbol.
        pools:          dict[str, list[dict]]             = defaultdict(list)
        pool_ticker_ct: dict[str, dict[str, int]]         = defaultdict(lambda: defaultdict(int))

        for pick in all_results:
            strat = pick.get('strategy', '')
            sym   = pick.get('symbol',   '')
            pool_quota = max_ic_slots if (strat == 'IC' and ic_pct < 1.0) else n
            if len(pools[strat]) >= pool_quota:
                continue
            if max_per_ticker is not None and pool_ticker_ct[strat][sym] >= max_per_ticker:
                continue
            pools[strat].append(pick)
            pool_ticker_ct[strat][sym] += 1

        if not pools:
            return []

        # -- Step 2: per-strategy quotas: n divided evenly, IC capped ----------
        num_strats  = len(pools)
        per_strat_q = max(1, n // num_strats)

        # -- Step 3: take top per_strat_q from each strategy -------------------
        # Track the pool pointer (next unused index) for each strategy so
        # step 4 can continue exactly where step 3 left off — no id() tricks
        # needed, which avoids false deduplication when tests mock the same
        # dict object for multiple tickers.
        selected: list[dict] = []
        pool_ptr: dict[str, int] = {}
        for strat, group in sorted(pools.items()):   # sorted for determinism
            q = min(per_strat_q, max_ic_slots) if strat == 'IC' else per_strat_q
            selected.extend(group[:q])
            pool_ptr[strat] = q   # index of first unused pick in this pool

        # -- Step 4: fill remaining n slots from per-strategy pool leftovers --
        remaining = n - len(selected)
        if remaining > 0:
            # Collect picks not yet taken (pool[ptr:] for each strategy)
            extras: list[dict] = []
            for strat, group in sorted(pools.items()):
                extras.extend(group[pool_ptr[strat]:])
            extras.sort(key=lambda x: x.get('score', 0), reverse=True)

            ic_in_selected   = sum(1 for p in selected if p.get('strategy') == 'IC')
            global_ticker_ct: dict[str, int] = defaultdict(int)
            for p in selected:
                global_ticker_ct[p.get('symbol', '')] += 1

            for pick in extras:
                if remaining <= 0:
                    break
                sym   = pick.get('symbol',   '')
                strat = pick.get('strategy', '')
                if max_per_ticker is not None and global_ticker_ct[sym] >= max_per_ticker:
                    continue
                if strat == 'IC' and ic_pct < 1.0 and ic_in_selected >= max_ic_slots:
                    continue
                selected.append(pick)
                global_ticker_ct[sym] += 1
                if strat == 'IC':
                    ic_in_selected += 1
                remaining -= 1

        # Final sort by score and trim to n
        selected.sort(key=lambda x: x.get('score', 0), reverse=True)
        return selected[:n]
