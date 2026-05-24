"""
SentimentAnalyzer
=================

Analyses recent price-action of any ticker to detect bull / bear / neutral
momentum and return per-strategy delta-adjustment factors.

Signals (each scored -1 → +1)
------------------------------
  RSI-14       : >60 = bullish, <40 = bearish
  Price vs SMA-20 : pct above/below the 20-day simple moving average
  5-day momentum : simple 5-day price return

Weighted composite score → sentiment label + strength (0–1).

Delta adjustment for spread scanning
-------------------------------------
  BULL  →  PCS max_delta × (1 + skew × strength)   allow slightly more delta
            CCS max_delta × (1 − skew × strength)   require tighter/safer strikes
  BEAR  →  PCS max_delta × (1 − skew × strength)
            CCS max_delta × (1 + skew × strength)
  NEUTRAL → unchanged

Configuration (all optional, under 'sentiment' key in config.json):
  enabled         : true / false (default true)
  lookback_days   : history window for SMA/momentum (default 20)
  rsi_period      : RSI smoothing period (default 14)
  bull_threshold  : composite score floor for BULL (default 0.20)
  bear_threshold  : composite score floor for BEAR (default 0.20)
  skew_factor     : fraction of base_delta to shift per unit of strength (default 0.30)
  max_skew        : hard cap on total adjustment fraction (default 0.50)
  weight_rsi      : RSI signal weight (default 0.35)
  weight_sma      : SMA signal weight (default 0.35)
  weight_momentum : momentum signal weight (default 0.30)
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import pandas as pd
import yfinance as yf

# ── Session-scoped cache ──────────────────────────────────────────────────────
_CACHE_LOCK        = threading.Lock()
_SENTIMENT_CACHE:  dict[str, dict] = {}
_CACHE_TTL_SECONDS = 3_600   # 1 hour — re-use within a session


class SentimentAnalyzer:
    """Analyse price-action momentum and expose delta-adjustment helpers."""

    def __init__(self, config: dict):
        cfg = config.get('sentiment', {})
        self.enabled        = cfg.get('enabled',         True)
        self.lookback_days  = cfg.get('lookback_days',   20)
        self.rsi_period     = cfg.get('rsi_period',      14)
        self.bull_threshold = cfg.get('bull_threshold',  0.20)
        self.bear_threshold = cfg.get('bear_threshold',  0.20)
        self.skew_factor      = cfg.get('skew_factor',       0.30)
        self.max_skew         = cfg.get('max_skew',          0.50)
        # Top-N allocation skew: how aggressively to reallocate picks between
        # PCS and CCS based on aggregate sentiment.  E.g. 0.50 = up to ±50%
        # of the base top_n can be shifted.  Default: same as skew_factor.
        self.top_n_skew_factor = cfg.get('top_n_skew_factor', self.skew_factor)

        # Normalised weights (coerce to sum=1)
        w_rsi  = cfg.get('weight_rsi',      0.35)
        w_sma  = cfg.get('weight_sma',      0.35)
        w_mom  = cfg.get('weight_momentum', 0.30)
        total  = w_rsi + w_sma + w_mom or 1.0
        self._w = {
            'rsi':      w_rsi  / total,
            'sma':      w_sma  / total,
            'momentum': w_mom  / total,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, symbol: str) -> dict:
        """
        Return a sentiment dict for *symbol*.

        Keys: symbol, sentiment ('BULL'|'BEAR'|'NEUTRAL'), strength (0–1),
              score (-1 → +1), rsi (float), sma_pct (%), momentum (%)

        Returns NEUTRAL on any data error so callers never need to guard.
        """
        if not self.enabled:
            return self._neutral(symbol)

        # Session cache — avoid re-fetching the same ticker within one run
        with _CACHE_LOCK:
            cached = _SENTIMENT_CACHE.get(symbol)
            if cached and (time.time() - cached.get('_ts', 0)) < _CACHE_TTL_SECONDS:
                return cached

        try:
            result = self._compute(symbol)
        except Exception:
            result = self._neutral(symbol)

        result['_ts'] = time.time()
        with _CACHE_LOCK:
            _SENTIMENT_CACHE[symbol] = result
        return result

    def adjust_delta(
        self,
        base_delta: float,
        strategy: str,
        sentiment_result: dict,
    ) -> float:
        """
        Adjust *base_delta* (max_delta_short_leg) for the given strategy type.

        BULL  PCS → raise ceiling  (bullish bias supports puts staying OTM)
        BULL  CCS → lower ceiling  (bearish calls more risky in bull mkt)
        BEAR  PCS → lower ceiling
        BEAR  CCS → raise ceiling
        IC    put/call leg → caller should use adjust_ic_deltas()

        Result is clamped to [0.05, 0.50] and within ±max_skew of base.
        """
        if not self.enabled:
            return base_delta

        sentiment = sentiment_result.get('sentiment', 'NEUTRAL')
        strength  = float(sentiment_result.get('strength', 0.0))
        adj_frac  = self.skew_factor * strength      # 0 … skew_factor

        if sentiment == 'NEUTRAL' or adj_frac == 0.0:
            return base_delta

        # +1 = raise delta ceiling (allow more premium),  -1 = lower (be tighter)
        direction_map: dict[str, int]
        if sentiment == 'BULL':
            direction_map = {'PCS': +1, 'CCS': -1}
        else:  # BEAR
            direction_map = {'PCS': -1, 'CCS': +1}

        direction = direction_map.get(strategy.upper(), 0)
        if direction == 0:
            return base_delta

        adjusted = base_delta * (1.0 + direction * adj_frac)
        lo = base_delta * (1.0 - self.max_skew)
        hi = base_delta * (1.0 + self.max_skew)
        return round(max(0.05, min(0.50, max(lo, min(hi, adjusted)))), 4)

    def adjust_ic_deltas(
        self,
        put_delta:  float,
        call_delta: float,
        sentiment_result: dict,
    ) -> tuple[float, float]:
        """Return (adjusted_put_delta, adjusted_call_delta) for Iron Condor."""
        return (
            self.adjust_delta(put_delta,  'PCS', sentiment_result),
            self.adjust_delta(call_delta, 'CCS', sentiment_result),
        )

    def strategy_top_n(
        self,
        base_n:           int,
        aggregate_score:  float,
        strategy_limits:  dict | None = None,
    ) -> dict[str, int]:
        """
        Return a per-strategy top-N allocation dict, adjusting PCS and CCS
        counts based on the aggregate period sentiment.

        Logic
        -----
        The *aggregate_score* is a composite in [-1, +1]:
          positive (BULL) → add picks to PCS, remove from CCS
          negative (BEAR) → add picks to CCS, remove from PCS
          0 (NEUTRAL)     → both strategies get the base quota

        The adjustment magnitude is ``top_n_skew_factor × |score|``, clamped
        so that neither strategy ever falls below 1 pick or exceeds 3× base_n.

        For every other strategy (IC, IFLY, etc.) the allocation is unchanged
        (= base_n) unless an explicit override is provided in *strategy_limits*.

        Parameters
        ----------
        base_n           : default top-N quota for every strategy
        aggregate_score  : float in [-1, +1]; positive = bull, negative = bear
        strategy_limits  : optional dict of {strategy: n} overrides; entries
                           NOT in this dict fall back to the computed value.

        Returns
        -------
        dict mapping strategy name → top-N quota (e.g. {'PCS': 13, 'CCS': 7, ...})
        """
        if not self.enabled or base_n <= 0:
            return {}   # empty → caller uses base_n for all strategies

        # Clamp score to [-1, +1]
        score = max(-1.0, min(1.0, float(aggregate_score)))
        skew  = self.top_n_skew_factor * abs(score)

        if score > 0:       # BULL
            pcs_n = round(base_n * (1.0 + skew))
            ccs_n = round(base_n * (1.0 - skew))
        elif score < 0:     # BEAR
            pcs_n = round(base_n * (1.0 - skew))
            ccs_n = round(base_n * (1.0 + skew))
        else:               # NEUTRAL
            pcs_n = ccs_n = base_n

        # Floor at 1, ceil at 3× base_n
        lo, hi = 1, max(1, base_n * 3)
        pcs_n  = max(lo, min(hi, pcs_n))
        ccs_n  = max(lo, min(hi, ccs_n))

        result = {'PCS': pcs_n, 'CCS': ccs_n}
        if strategy_limits:
            result.update(strategy_limits)
        return result

    # ── Batch pre-warm ────────────────────────────────────────────────────────

    def batch_prefetch(self, tickers: list[str]) -> None:
        """
        Download price history for *all* tickers in a single ``yf.download()``
        call, then compute and cache the sentiment result for each one.

        After this runs, ``analyze()`` returns instantly from cache for every
        ticker in the list — no per-ticker yfinance calls during the scan.

        Called automatically by ``OptionScanner.get_top_picks()`` when a
        ``sentiment_analyzer`` is attached.
        """
        if not self.enabled or not tickers:
            return

        period = f"{self.lookback_days + 15}d"
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

            # Normalise to a DataFrame of Close prices keyed by symbol.
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw['Close']          # dates × symbols
            else:
                sym = tickers[0] if len(tickers) == 1 else None
                if sym is None:
                    return
                close_df = raw[['Close']].rename(columns={'Close': sym})

            ts = time.time()
            for sym in close_df.columns:
                closes = close_df[sym].dropna()
                if len(closes) < max(self.rsi_period + 1, 6):
                    continue
                try:
                    result = self._compute_from_series(sym, closes)
                    result['_ts'] = ts
                    with _CACHE_LOCK:
                        _SENTIMENT_CACHE[sym] = result
                except Exception:
                    pass
        except Exception:
            pass   # silent — per-ticker analyze() fallback handles misses

    # ── Internal computation ──────────────────────────────────────────────────

    def _compute(self, symbol: str) -> dict:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period=f"{self.lookback_days + 15}d")
        closes = hist['Close'].dropna()
        return self._compute_from_series(symbol, closes)

    def _compute_from_series(self, symbol: str, closes: pd.Series) -> dict:
        """Compute RSI / SMA / momentum signals from a pre-fetched Close series."""
        if len(closes) < max(self.rsi_period + 1, 6):
            return self._neutral(symbol)

        # ── RSI signal ────────────────────────────────────────────────────────
        rsi = self._rsi(closes, self.rsi_period)
        if rsi > 60:
            rsi_signal = min((rsi - 60) / 40.0, 1.0)       # +1.0 at RSI=100
        elif rsi < 40:
            rsi_signal = -min((40 - rsi) / 40.0, 1.0)      # -1.0 at RSI=0
        else:
            rsi_signal = (rsi - 50) / 50.0 * 0.4            # tiny linear band

        # ── SMA-20 signal ─────────────────────────────────────────────────────
        n_sma  = min(20, len(closes))
        sma20  = closes.iloc[-n_sma:].mean()
        sma_pct = (closes.iloc[-1] - sma20) / sma20 if sma20 > 0 else 0.0
        # ±5% ↔ ±1.0
        sma_signal = max(-1.0, min(1.0, sma_pct * 20.0))

        # ── 5-day momentum ────────────────────────────────────────────────────
        mom_pct = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] \
                  if len(closes) >= 6 else 0.0
        # ±6.7% ↔ ±1.0
        momentum_signal = max(-1.0, min(1.0, mom_pct * 15.0))

        # ── Weighted composite ────────────────────────────────────────────────
        score = (
            rsi_signal      * self._w['rsi'] +
            sma_signal      * self._w['sma'] +
            momentum_signal * self._w['momentum']
        )

        if score > self.bull_threshold:
            sentiment = 'BULL'
            strength  = min(score, 1.0)
        elif score < -self.bear_threshold:
            sentiment = 'BEAR'
            strength  = min(-score, 1.0)
        else:
            sentiment = 'NEUTRAL'
            strength  = 0.0

        return {
            'symbol':    symbol,
            'sentiment': sentiment,
            'strength':  round(strength,  3),
            'score':     round(score,     4),
            'rsi':       round(rsi,       1),
            'sma_pct':   round(sma_pct  * 100, 2),
            'momentum':  round(mom_pct  * 100, 2),
        }

    @staticmethod
    def _rsi(closes, period: int = 14) -> float:
        """Wilder's smoothed RSI."""
        deltas = closes.diff().dropna()
        if len(deltas) < period:
            return 50.0
        gains  = deltas.clip(lower=0)
        losses = (-deltas).clip(lower=0)
        avg_g  = gains.iloc[:period].mean()
        avg_l  = losses.iloc[:period].mean()
        for i in range(period, len(deltas)):
            avg_g = (avg_g * (period - 1) + gains.iloc[i])  / period
            avg_l = (avg_l * (period - 1) + losses.iloc[i]) / period
        if avg_l == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + avg_g / avg_l))

    @staticmethod
    def _neutral(symbol: str) -> dict:
        return {
            'symbol':    symbol,
            'sentiment': 'NEUTRAL',
            'strength':  0.0,
            'score':     0.0,
            'rsi':       50.0,
            'sma_pct':   0.0,
            'momentum':  0.0,
        }
