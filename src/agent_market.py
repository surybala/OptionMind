"""
agent_market.py
===============

Market-data helpers for the agent: VIX fetching (multi-source with
file cache), regime evaluation, and position reconciliation.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor
from src.position_reconciler import PositionReconciler
from src.regime import RegimeResult, RegimeService
from src.utils import get_logger

log = get_logger()


# ── VIX cache settings ──────────────────────────────────────────────────────

_VIX_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'vix_cache.json',
)
_VIX_CACHE_TTL_HOURS   = 4   # fresh cache: reuse without any network call
_VIX_STALE_MAX_HOURS   = 24  # stale-but-usable: last resort if every source fails


def fetch_vix(config: dict) -> Optional[float]:
    """
    Return the current CBOE VIX level using a layered, rate-limit-aware strategy:

      1. File cache  (data/vix_cache.json, TTL=4 h) — zero network calls
      2. Alpaca data API  — 'VIX' symbol (no API key for free tier, but works
                            when Alpaca credentials are configured)
      3. yfinance strategy A — Ticker.fast_info  (lighter request path)
      4. yfinance strategy B — Ticker.history(period='5d')
      5. yfinance strategy C — yf.download()  (different HTTP path, less throttled)
      6. Stale cache  (up to 24 h old) — better than skipping the VIX filter
         entirely when all live sources are temporarily rate-limited

    A fresh fetch result is always written back to the cache.
    """
    # ── 1. Fresh cache ─────────────────────────────────────────────────────────
    def _load_cache():
        try:
            with open(_VIX_CACHE_PATH) as _f:
                return json.load(_f)
        except Exception:
            return None

    cached = _load_cache()
    if cached:
        try:
            age = datetime.now() - datetime.fromisoformat(cached['fetched_at'])
            if age < timedelta(hours=_VIX_CACHE_TTL_HOURS):
                log.info(
                    f"VIX={cached['value']:.1f} (from cache, "
                    f"{int(age.total_seconds() / 60)}m old)"
                )
                return float(cached['value'])
        except Exception:
            pass

    # ── 2–5. Live sources ──────────────────────────────────────────────────────
    vix: Optional[float] = None

    # Source 2: Alpaca — works when credentials are configured; Alpaca serves
    # 'VIX' (no caret) as an index symbol on their market-data feed.
    try:
        from src.alpaca_data import make_alpaca_data_client
        _alpaca = make_alpaca_data_client(config)
        if _alpaca is not None:
            for _sym in ('VIX', '^VIX'):
                try:
                    _val = _alpaca.get_spot_price(_sym)
                    if _val is not None and float(_val) > 0:
                        vix = float(_val)
                        log.info(f"VIX={vix:.1f} (via Alpaca, symbol={_sym!r})")
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # Source 3: yfinance fast_info (lighter endpoint, separate rate-limit bucket)
    if vix is None:
        try:
            import yfinance as _yf
            _fi = _yf.Ticker('^VIX').fast_info
            _v  = getattr(_fi, 'last_price', None) or getattr(_fi, 'regularMarketPrice', None)
            if _v and float(_v) > 0:
                vix = float(_v)
                log.info(f"VIX={vix:.1f} (via yfinance fast_info)")
        except Exception:
            pass

    # Source 4: yfinance history with extended period
    if vix is None:
        try:
            import yfinance as _yf
            _h = _yf.Ticker('^VIX').history(period='5d')
            if not _h.empty:
                vix = float(_h['Close'].iloc[-1])
                log.info(f"VIX={vix:.1f} (via yfinance history)")
        except Exception:
            pass

    # Source 5: yf.download — hits a different HTTP endpoint, less throttled
    if vix is None:
        try:
            import yfinance as _yf
            _dl = _yf.download('^VIX', period='5d', progress=False, auto_adjust=True)
            if not _dl.empty:
                _col = _dl['Close']
                if hasattr(_col, 'iloc'):
                    vix = float(_col.iloc[-1])
                    log.info(f"VIX={vix:.1f} (via yfinance download)")
        except Exception:
            pass

    # ── Write fresh result to cache ────────────────────────────────────────────
    if vix is not None:
        try:
            os.makedirs(os.path.dirname(_VIX_CACHE_PATH), exist_ok=True)
            with open(_VIX_CACHE_PATH, 'w') as _f:
                json.dump({'value': vix, 'fetched_at': datetime.now().isoformat()}, _f)
        except Exception:
            pass
        return vix

    # ── 6. Stale cache (last resort) ───────────────────────────────────────────
    if cached:
        try:
            age = datetime.now() - datetime.fromisoformat(cached['fetched_at'])
            if age < timedelta(hours=_VIX_STALE_MAX_HOURS):
                log.warning(
                    f"All VIX sources failed; using stale cache "
                    f"VIX={cached['value']:.1f} ({int(age.total_seconds()/3600)}h old)"
                )
                return float(cached['value'])
        except Exception:
            pass

    return None


# ── Regime evaluation ────────────────────────────────────────────────────────

def _fetch_regime_history(symbol: str, period: str = '90d') -> list[float]:
    """Fetch adjusted closes for regime classification; empty list on failure."""
    try:
        import yfinance as yf
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist is None or getattr(hist, 'empty', True):
            return []
        close = hist.get('Close')
        if close is None:
            return []
        return [float(x) for x in close.dropna().tolist()]
    except Exception:
        return []


def evaluate_regime_filter(
    config: dict,
    current_vix: Optional[float] = None,
) -> RegimeResult:
    svc = RegimeService(config)
    cfg = config.get('risk_parameters', {}).get('regime_filter', {})
    if not cfg.get('enabled', False):
        return svc.evaluate(vix_current=current_vix)

    trend_cfg = cfg.get('trend', {})
    trend_symbol = str(trend_cfg.get('symbol', 'SPY') or 'SPY')
    vix_history = _fetch_regime_history('^VIX')
    spy_history = _fetch_regime_history(trend_symbol)
    result = svc.evaluate(
        vix_current=current_vix,
        vix_history=vix_history,
        spy_history=spy_history,
    )
    reason = '; '.join(result.reasons) if result.reasons else 'no reason supplied'
    log.info(
        "Regime filter: %s — qty %.0f%%, top-N %.0f%% (%s)",
        result.label,
        result.quantity_multiplier * 100,
        result.top_n_multiplier * 100,
        reason,
    )
    return result


# ── Position reconciliation ──────────────────────────────────────────────────

def reconcile_positions_before_budget(
    db: TradeDatabase,
    executor: AlpacaExecutor,
    config: dict,
) -> None:
    """
    Sync local position state with Alpaca before dedup/capital accounting.

    Profit-take and manual exits can happen in another process, or the prior
    agent run can die after submitting a close order.  The budget gate should
    use the freshest local state possible, so run the lightweight reconciler
    before reading open positions.
    """
    sched_cfg = config.get('monitor_schedule', {})
    grace_min = int(sched_cfg.get('ghost_grace_minutes', 10))
    summary = PositionReconciler(
        db, executor, ghost_grace_minutes=grace_min
    ).run()
    if summary.get('error'):
        log.warning(
            "[agent] Reconciliation skipped before capital accounting: %s",
            summary['error'],
        )
        return

    pending = summary.get('pending_closes', {})
    executed = summary.get('executed', {})
    if isinstance(pending, dict) and isinstance(executed, dict):
        confirmed = int(pending.get('confirmed', 0) or 0)
        ghost_closed = int(executed.get('ghost_closed', 0) or 0)
        if confirmed or ghost_closed:
            log.info(
                "[agent] Reconciled %d pending close(s), %d externally closed "
                "position(s) before budget accounting.",
                confirmed, ghost_closed,
            )
