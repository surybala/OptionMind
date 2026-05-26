"""
OptionMind Agent — Main Entry Point
===================================

Runs the ML scanner hook, applies deterministic risk controls, presents a
ranked plan of model-suggested trades, and optionally executes approved picks
via Alpaca.

Execution modes
---------------
  approve   (default)  Show the plan table, ask for user approval, then
                        execute only the approved subset.
  auto                 Execute automatically any pick whose prob_win
                        exceeds auto_execute_prob in config.json. Model
                        candidates can populate prob_win from
                        probability_of_profit.
                        (useful for scheduled / headless runs).
  scan-only            Score and print the plan but never execute anything.
                        Picks are also saved to data/pending_picks.json.

Trade safety
------------
  --dry-run  (default) Simulate every order — nothing reaches Alpaca.
  --live               Submit real orders to Alpaca.  Requires valid
                       api_key / api_secret in config.json.

Usage examples
--------------
  # Interactive approve mode (dry-run by default):
  python agent.py

  # Interactive approve mode, submit real orders when approved:
  python agent.py --live

  # Auto-execute high-confidence picks (live):
  python agent.py --mode auto --live

  # Score only — no execution, no prompts:
  python agent.py --mode scan-only

  # Use full index universe instead of the default 10-ticker sample:
  python agent.py --universe index

  # Override max capital per period:
  python agent.py --max-capital 30000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor
from src.notifier import EmailNotifier
from src.capital import capital_for_position as _capital_for_position
from src.position_reconciler import PositionReconciler
from src.position_monitor import PositionMonitor
from src.portfolio_risk import PortfolioRiskService
from src.regime import RegimeResult, RegimeService
from src.utils import get_logger, load_config

log = get_logger()

SCAN_AUDIT_PATH = os.path.join('data', 'model_candidates.json')


# ── Pick formatting ────────────────────────────────────────────────────────────

def _capital_for_pick(pick: dict) -> float:
    """Estimate the capital requirement for a model candidate."""
    strat = pick.get('strategy', '')
    price = pick.get('current_price', 0.0) or 0.0

    if strat == 'CSP':
        return (pick.get('short_strike') or 0.0) * 100
    if strat in ('PCS', 'CCS'):
        ss = pick.get('short_strike') or pick.get('short_put') or pick.get('short_call') or 0.0
        ls = pick.get('long_strike')  or pick.get('long_put')  or pick.get('long_call')  or 0.0
        return abs(ss - ls) * 100
    if strat == 'IC':
        sp = pick.get('short_put',  0.0) or 0.0
        lp = pick.get('long_put',   0.0) or 0.0
        sc = pick.get('short_call', 0.0) or 0.0
        lc = pick.get('long_call',  0.0) or 0.0
        return max(abs(sp - lp), abs(sc - lc)) * 100
    if strat == 'IFLY':
        put_wing  = pick.get('put_wing', 0.0) or abs((pick.get('short_put', 0) or 0) - (pick.get('long_put', 0) or 0))
        call_wing = pick.get('call_wing', 0.0) or abs((pick.get('long_call', 0) or 0) - (pick.get('short_call', 0) or 0))
        return max(put_wing, call_wing) * 100
    if strat == 'CC':
        return price * 100
    if strat == 'STRANGLE':
        return price * 0.20 * 100
    return 0.0


def _strategy_sides(strategy: str) -> set[str]:
    """Return directional risk side(s) for a strategy."""
    strat = (strategy or '').upper()
    if strat in ('CSP', 'PCS'):
        return {'put'}
    if strat in ('CC', 'CCS'):
        return {'call'}
    if strat in ('IC', 'IFLY', 'STRANGLE'):
        return {'put', 'call'}
    return set()


def _pick_width(pick: dict) -> float:
    strat = (pick.get('strategy') or '').upper()
    if strat in ('PCS', 'CCS'):
        ss = pick.get('short_strike') or pick.get('short_put') or pick.get('short_call') or 0.0
        ls = pick.get('long_strike') or pick.get('long_put') or pick.get('long_call') or 0.0
        return abs(float(ss or 0) - float(ls or 0))
    if strat in ('IC', 'IFLY'):
        sp = float(pick.get('short_put') or 0)
        lp = float(pick.get('long_put') or 0)
        sc = float(pick.get('short_call') or 0)
        lc = float(pick.get('long_call') or 0)
        put_wing = float(pick.get('put_wing') or abs(sp - lp))
        call_wing = float(pick.get('call_wing') or abs(lc - sc))
        return max(put_wing, call_wing)
    if strat in ('CSP', 'STRANGLE'):
        return float(pick.get('short_strike') or pick.get('short_put') or 0)
    if strat == 'CC':
        return float(pick.get('current_price') or pick.get('short_strike') or 0)
    return 0.0


def _max_loss_per_contract_for_pick(pick: dict) -> float:
    """Return estimated max loss per contract in dollars."""
    premium = max(0.0, float(pick.get('premium') or 0))
    width = _pick_width(pick)
    if width <= 0:
        return 0.0
    return max(0.0, round((width - premium) * 100, 2))


def _max_loss_multiple_for_pick(pick: dict) -> float:
    credit = max(0.0, float(pick.get('premium') or 0) * 100)
    if credit <= 0:
        return float('inf')
    return round(_max_loss_per_contract_for_pick(pick) / credit, 4)


def _pick_key(pick: dict) -> tuple:
    """Stable identity for one model candidate across risk-gate lists."""
    def _num(value) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    strat = (pick.get('strategy') or '').upper()
    return (
        str(pick.get('symbol') or '').upper(),
        strat,
        str(pick.get('expiry') or ''),
        _num(pick.get('short_strike') or pick.get('short_put')),
        _num(pick.get('long_strike') or pick.get('long_put')),
        _num(pick.get('short_call')),
        _num(pick.get('long_call')),
    )


def _mispricing_score_for_pick(pick: dict) -> float:
    """
    Practical model score adapter for existing risk/audit displays.

    New candidates should provide ``model_score``. Legacy ``score`` remains
    accepted only so old fixtures and execution code can share the same shape.
    """
    try:
        return round(float(pick.get('mispricing_score', pick.get('score', 0.0)) or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def _annotate_mispricing_scores(picks: list[dict]) -> list[dict]:
    for pick in picks:
        pick['mispricing_score'] = _mispricing_score_for_pick(pick)
        pick.setdefault(
            'mispricing_score_basis',
            'ML model score: expected utility / P&L-centered inference',
        )
    return picks


def _capture_rejections(
    before: list[dict],
    after: list[dict],
    gate: str,
    reason,
    rejected: list[dict],
) -> None:
    kept = {_pick_key(p) for p in after}
    seen = {_pick_key(p) for p in rejected}
    for pick in before:
        key = _pick_key(pick)
        if key in kept or key in seen:
            continue
        item = dict(pick)
        item['filtered_stage'] = gate
        item['reject_reason'] = reason(pick) if callable(reason) else str(reason)
        item['mispricing_score'] = _mispricing_score_for_pick(item)
        rejected.append(item)


def _pick_audit_row(pick: dict, status: str) -> dict:
    quantity = int(pick.get('quantity') or 1)
    premium = float(pick.get('premium') or 0.0)
    return {
        'status': status,
        'symbol': pick.get('symbol'),
        'strategy': pick.get('strategy'),
        'expiry': pick.get('expiry'),
        'legs': _legs_str(pick),
        'quantity': quantity,
        'premium': round(premium, 4),
        'total_credit': round(premium * 100 * quantity, 2),
        'prob_win': pick.get('prob_win'),
        'roi': pick.get('roi'),
        'score': pick.get('score'),
        'mispricing_score': _mispricing_score_for_pick(pick),
        'mispricing_score_basis': pick.get('mispricing_score_basis'),
        'current_price': pick.get('current_price'),
        'max_loss_multiple': pick.get('max_loss_multiple'),
        'max_loss_per_contract': pick.get('max_loss_per_contract'),
        'filtered_stage': pick.get('filtered_stage'),
        'reject_reason': pick.get('reject_reason'),
    }


def _write_scan_audit(
    selected: list[dict],
    rejected: list[dict],
    *,
    db: TradeDatabase | None = None,
    path: str = SCAN_AUDIT_PATH,
    max_rejected: int = 25,
    scanner_type: str = 'ml',
) -> None:
    """Persist latest candidate plan/rejections for the dashboard."""
    if db is not None and scanner_type == 'ml':
        _record_model_decisions(db, selected, rejected)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    selected_rows = [_pick_audit_row(p, 'SELECTED') for p in selected]
    selected_floor = min(
        (row['mispricing_score'] for row in selected_rows),
        default=0.0,
    )
    interesting_rejected = [
        p for p in rejected
        if _mispricing_score_for_pick(p) >= selected_floor
    ]
    if not interesting_rejected:
        interesting_rejected = list(rejected)
    interesting_rejected = sorted(
        interesting_rejected,
        key=lambda p: _mispricing_score_for_pick(p),
        reverse=True,
    )[:max_rejected]
    rejected_rows = [_pick_audit_row(p, 'REJECTED') for p in interesting_rejected]
    if scanner_type == 'ml':
        score_basis = (
            'Higher means the ML inference layer ranked the candidate as more '
            'attractive; deterministic risk gates still apply.'
        )
    else:
        score_basis = (
            'Deterministic scanner score (legacy heuristic); '
            'higher is more attractive per the rule-based ranker.'
        )
    payload = {
        'generated_at': datetime.now().isoformat(),
        'scanner': scanner_type,
        'score_basis': score_basis,
        'selected': selected_rows,
        'rejected': rejected_rows,
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)


def _record_model_predictions(db: TradeDatabase, picks: list[dict]) -> None:
    """Append model prediction facts and stamp ids back onto candidate dicts."""
    for pick in picks:
        if pick.get('model_prediction_id'):
            continue
        try:
            pick['model_prediction_id'] = db.record_model_prediction(pick)
        except Exception as exc:
            log.warning(
                "[ledger] Failed to record model prediction for %s %s: %s",
                pick.get('strategy'), pick.get('symbol'), exc,
            )


def _record_model_decisions(
    db: TradeDatabase,
    selected: list[dict],
    rejected: list[dict],
) -> None:
    """Append final pass/reject decisions for the current scan."""
    for rank, pick in enumerate(selected, start=1):
        if pick.get('model_decision_id'):
            continue
        prediction_id = pick.get('model_prediction_id')
        if prediction_id is None:
            try:
                prediction_id = db.record_model_prediction(pick)
                pick['model_prediction_id'] = prediction_id
            except Exception as exc:
                log.warning("[ledger] Failed to backfill selected prediction: %s", exc)
                continue
        try:
            pick['model_decision_id'] = db.record_model_decision(
                prediction_id=prediction_id,
                decision='SELECTED',
                selected_rank=rank,
                quantity=pick.get('quantity', 1),
                raw_decision=pick,
            )
        except Exception as exc:
            log.warning("[ledger] Failed to record selected decision: %s", exc)

    for pick in rejected:
        if pick.get('model_decision_id'):
            continue
        prediction_id = pick.get('model_prediction_id')
        if prediction_id is None:
            try:
                prediction_id = db.record_model_prediction(pick)
                pick['model_prediction_id'] = prediction_id
            except Exception as exc:
                log.warning("[ledger] Failed to backfill rejected prediction: %s", exc)
                continue
        try:
            pick['model_decision_id'] = db.record_model_decision(
                prediction_id=prediction_id,
                decision='REJECTED',
                risk_gate=pick.get('filtered_stage'),
                reject_reason=pick.get('reject_reason'),
                quantity=pick.get('quantity', 1),
                raw_decision=pick,
            )
        except Exception as exc:
            log.warning("[ledger] Failed to record rejected decision: %s", exc)


def _max_loss_for_position(pos: dict) -> float:
    """Estimate max remaining strategy loss for an open DB position."""
    strat = (pos.get('type') or '').upper()
    legs = pos.get('legs') or {}
    if isinstance(legs, str):
        try:
            legs = json.loads(legs) or {}
        except Exception:
            legs = {}
    premium = max(0.0, float(pos.get('premium') or 0))
    strike = float(pos.get('strike') or 0)
    contracts = int(pos.get('contracts') or 1)

    width = 0.0
    if strat in ('PCS', 'CCS'):
        ss = legs.get('short_strike') or legs.get('short_put') or legs.get('short_call') or strike
        ls = legs.get('long_strike') or legs.get('long_put') or legs.get('long_call') or 0
        width = abs(float(ss or 0) - float(ls or 0))
    elif strat in ('IC', 'IFLY'):
        sp = float(legs.get('short_put') or 0)
        lp = float(legs.get('long_put') or 0)
        sc = float(legs.get('short_call') or 0)
        lc = float(legs.get('long_call') or 0)
        width = max(abs(sp - lp), abs(sc - lc))
    elif strat in ('CSP', 'STRANGLE'):
        width = float(legs.get('short_strike') or legs.get('short_put') or strike or 0)
    elif strat == 'CC':
        width = float(legs.get('short_strike') or legs.get('short_call') or strike or 0)

    return max(0.0, round((width - premium) * 100 * contracts, 2))


def _directional_exposure(open_positions: list[dict]) -> dict[str, float]:
    exposure = {'put': 0.0, 'call': 0.0}
    for pos in open_positions:
        sides = _strategy_sides(pos.get('type', ''))
        if not sides:
            continue
        loss = _max_loss_for_position(pos)
        share = loss / len(sides)
        for side in sides:
            exposure[side] += share
    return {k: round(v, 2) for k, v in exposure.items()}


def _filter_max_loss_multiple(picks: list[dict], config: dict) -> list[dict]:
    cfg = config.get('risk_parameters', {}).get('max_loss_multiple', {})
    if not cfg.get('enabled', True):
        return picks
    default_limit = float(cfg.get('default', cfg.get('limit', 6.0)))
    by_strategy = cfg.get('by_strategy', {})
    kept: list[dict] = []
    rejected = 0
    for pick in picks:
        strat = (pick.get('strategy') or '').upper()
        limit = float(by_strategy.get(strat, default_limit))
        multiple = _max_loss_multiple_for_pick(pick)
        pick['max_loss_multiple'] = multiple
        pick['max_loss_per_contract'] = _max_loss_per_contract_for_pick(pick)
        if multiple <= limit:
            kept.append(pick)
        else:
            rejected += 1
            log.info(
                "Max-loss multiple filter: rejected %s %s %.2fx > %.2fx "
                "(credit=$%.2f, max_loss=$%.2f/contract).",
                strat, pick.get('symbol'), multiple, limit,
                float(pick.get('premium') or 0) * 100,
                pick['max_loss_per_contract'],
            )
    if rejected:
        log.info("Max-loss multiple filter: kept %d/%d pick(s).", len(kept), len(picks))
    return kept


def _apply_directional_exposure_caps(
    picks: list[dict],
    open_positions: list[dict],
    config: dict,
    account_capital: Optional[float],
) -> list[dict]:
    cfg = config.get('risk_parameters', {}).get('directional_exposure_caps', {})
    if not cfg.get('enabled', True) or not account_capital:
        return picks
    try:
        account_capital = float(account_capital)
    except (TypeError, ValueError):
        log.warning(
            "Directional cap disabled: account_capital/max_capital_per_period is not numeric."
        )
        return picks

    min_side_cap = float(cfg.get('min_side_cap_dollars', 0.0) or 0.0)
    put_limit = max(
        float(cfg.get('put', cfg.get('max_put_pct', 0.04))) * account_capital,
        float(cfg.get('min_put_cap_dollars', min_side_cap) or 0.0),
    )
    call_limit = max(
        float(cfg.get('call', cfg.get('max_call_pct', 0.04))) * account_capital,
        float(cfg.get('min_call_cap_dollars', min_side_cap) or 0.0),
    )
    limits = {'put': put_limit, 'call': call_limit}
    used = _directional_exposure(open_positions)
    capped: list[dict] = []

    for pick in sorted(picks, key=lambda x: x.get('score', 0.0), reverse=True):
        sides = _strategy_sides(pick.get('strategy', ''))
        if not sides:
            capped.append(pick)
            continue
        per_contract_loss = _max_loss_per_contract_for_pick(pick)
        if per_contract_loss <= 0:
            continue
        requested_qty = int(pick.get('quantity') or 1)
        per_side_loss = per_contract_loss / len(sides)
        side_cap_qty = requested_qty
        for side in sides:
            remaining = limits[side] - used.get(side, 0.0)
            side_cap_qty = min(side_cap_qty, int(remaining // per_side_loss))
        if side_cap_qty <= 0:
            log.info(
                "Directional cap: rejected %s %s; side exposure used=%s limits=%s.",
                pick.get('strategy'), pick.get('symbol'), used, limits,
            )
            continue
        if side_cap_qty < requested_qty:
            log.info(
                "Directional cap: reduced %s %s quantity %d → %d.",
                pick.get('strategy'), pick.get('symbol'), requested_qty, side_cap_qty,
            )
        pick['quantity'] = side_cap_qty
        for side in sides:
            used[side] = round(used.get(side, 0.0) + per_side_loss * side_cap_qty, 2)
        capped.append(pick)

    log.info(
        "Directional exposure after sizing: put=$%.0f/$%.0f, call=$%.0f/$%.0f.",
        used.get('put', 0.0), put_limit, used.get('call', 0.0), call_limit,
    )
    return capped


def _apply_portfolio_gamma_risk(
    picks: list[dict],
    capital_positions: list[dict],
    config: dict,
    account_capital: Optional[float],
    monitor: PositionMonitor,
) -> list[dict]:
    svc = PortfolioRiskService(
        config,
        position_risk_service=getattr(monitor, '_risk_service', None),
    )
    if not svc.enabled():
        return picks
    filtered = svc.filter_picks(picks, capital_positions, account_capital)
    if len(filtered) != len(picks):
        log.info("Portfolio gamma gate: kept %d/%d pick(s).", len(filtered), len(picks))
    return filtered


def _reconcile_positions_before_budget(
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


_VIX_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'vix_cache.json'
)
_VIX_CACHE_TTL_HOURS   = 4   # fresh cache: reuse without any network call
_VIX_STALE_MAX_HOURS   = 24  # stale-but-usable: last resort if every source fails


def _fetch_vix(config: dict, log) -> Optional[float]:
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
    from datetime import timedelta

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


def _evaluate_regime_filter(
    config: dict,
    log,
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


def _apply_regime_quantity_multiplier(
    picks: list[dict],
    regime: Optional[RegimeResult],
) -> list[dict]:
    if regime is None or regime.quantity_multiplier >= 1.0:
        return picks
    if regime.quantity_multiplier <= 0:
        return []

    adjusted: list[dict] = []
    for pick in picks:
        qty = max(1, int(pick.get('quantity') or 1))
        new_qty = max(1, int(qty * regime.quantity_multiplier))
        pick['quantity'] = min(qty, new_qty)
        pick['regime'] = regime.label
        pick['regime_quantity_multiplier'] = regime.quantity_multiplier
        adjusted.append(pick)
    return adjusted


def _fmt_mcap(v) -> str:
    """Format a market-cap value as '1.2T', '45.3B', '850M', or '—'."""
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        return '—'
    if v >= 1e12:
        return f'{v/1e12:.1f}T'
    if v >= 1e9:
        return f'{v/1e9:.1f}B'
    if v >= 1e6:
        return f'{v/1e6:.0f}M'
    return f'{v:,.0f}'


def _legs_from_pick(pick: dict) -> dict:
    """
    Extract leg strikes from a model candidate dict into a flat dict suitable
    for storage in the database's 'legs' JSON column.
    Used by PositionMonitor to price the position for stop-loss checks.
    Extra keys (market_cap, short_oi, short_volume) are stored alongside
    the strikes so the dashboard can display them without a live re-fetch.
    """
    strat = pick.get('strategy', '')
    if strat in ('PCS', 'CCS'):
        legs = {
            'short_strike': pick.get('short_strike'),
            'long_strike':  pick.get('long_strike'),
        }
    elif strat in ('IC', 'IFLY'):
        legs = {
            'short_put':  pick.get('short_put'),
            'long_put':   pick.get('long_put'),
            'short_call': pick.get('short_call'),
            'long_call':  pick.get('long_call'),
        }
    elif strat == 'CSP':
        legs = {'short_strike': pick.get('short_strike')}
    elif strat == 'CC':
        legs = {'short_strike': pick.get('short_strike') or pick.get('short_call')}
    elif strat == 'STRANGLE':
        legs = {
            'short_put':  pick.get('short_put'),
            'short_call': pick.get('short_call'),
        }
    else:
        legs = {}
    # Persist display metadata so the dashboard doesn't need a live re-fetch
    if pick.get('market_cap') is not None:
        legs['market_cap'] = pick['market_cap']
    if pick.get('short_oi') is not None:
        legs['short_oi'] = pick['short_oi']
    if 'short_volume' in pick:
        legs['short_volume'] = pick['short_volume']
    if pick.get('mispricing_score') is not None:
        legs['mispricing_score'] = pick['mispricing_score']
        legs['mispricing_score_basis'] = pick.get('mispricing_score_basis')
    return legs


def _legs_str(pick: dict) -> str:
    """Compact leg description for the plan table."""
    strat = pick.get('strategy', '')
    if strat == 'CSP':
        return f"{pick.get('short_strike')}P"
    if strat == 'PCS':
        return f"{pick.get('short_put') or pick.get('short_strike')}/{pick.get('long_put') or pick.get('long_strike')} P"
    if strat == 'CCS':
        return f"{pick.get('short_call') or pick.get('short_strike')}/{pick.get('long_call') or pick.get('long_strike')} C"
    if strat == 'IC':
        return (f"{pick.get('long_put')}/{pick.get('short_put')} P  "
                f"{pick.get('short_call')}/{pick.get('long_call')} C")
    if strat == 'IFLY':
        return (f"{pick.get('long_put')}/{pick.get('short_put')}(ATM)"
                f"/{pick.get('long_call')}")
    if strat == 'STRANGLE':
        return f"{pick.get('short_put')}P / {pick.get('short_call')}C"
    if strat == 'CC':
        return f"{pick.get('short_strike') or pick.get('short_call')}C"
    return '?'


def _print_open_positions(open_positions: list[dict], monitor) -> None:
    """
    Print a formatted table of all open positions with live mark prices and
    unrealized P&L fetched via PositionMonitor._get_current_mark().
    """
    import datetime as _dt

    today = _dt.date.today()

    print()
    print("=" * 106)
    print("  OPEN POSITIONS — CURRENT P&L")
    print(f"  As of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {len(open_positions)} position(s)")
    print("=" * 106)

    hdr  = f"  {'#':>3}  {'Strat':<6}  {'Symbol':<6}  {'Entry':<10}  {'Expiry':<10}  {'DTE':>4}  "
    hdr += f"{'Spot':>8}  {'Entry$':>8}  {'Mark$':>8}  {'Unreal P&L':>12}  {'P&L%':>7}  {'Status'}"
    print(hdr)
    print("  " + "-" * 102)

    total_pnl = 0.0
    priced    = 0

    for i, pos in enumerate(open_positions, start=1):
        strat   = pos.get('type', '?')
        symbol  = pos.get('symbol', '?')
        entry   = (pos.get('timestamp') or '')[:10]
        expiry  = pos.get('expiry', '?')
        premium = float(pos.get('premium', 0) or 0)
        status  = pos.get('status', '?')

        try:
            exp_date = _dt.date.fromisoformat(expiry)
            dte      = (exp_date - today).days
            dte_str  = f"{dte}d"
        except Exception:
            dte_str  = "?"

        # Fetch current underlying spot
        try:
            import yfinance as _yf
            _hist = _yf.Ticker(symbol).history(period='2d')
            spot_str = f"${float(_hist['Close'].iloc[-1]):>7.2f}" if not _hist.empty else f"{'N/A':>8}"
        except Exception:
            spot_str = f"{'N/A':>8}"

        # Fetch current option mark
        current_mark = monitor._get_current_mark(pos)

        if current_mark is not None:
            unreal_pnl = (premium - current_mark) * 100
            pnl_pct    = (premium - current_mark) / premium * 100 if premium > 0 else 0.0
            total_pnl += unreal_pnl
            priced    += 1
            pnl_str    = f"${unreal_pnl:>+10,.2f}"
            pct_str    = f"{pnl_pct:>+6.1f}%"
            mark_str   = f"${current_mark:>7.4f}"
        else:
            pnl_str  = f"{'N/A':>11}"
            pct_str  = f"{'N/A':>7}"
            mark_str = f"{'N/A':>8}"

        entry_str = f"${premium * 100:>7.2f}"

        print(
            f"  {i:>3}. {strat:<6}  {symbol:<6}  {entry:<10}  {expiry:<10}  {dte_str:>4}  "
            f"{spot_str}  {entry_str}  {mark_str}  {pnl_str}  {pct_str}  {status}"
        )

    print("  " + "-" * 102)
    if priced > 0:
        sign = "+" if total_pnl >= 0 else ""
        print(f"  Total unrealized P&L ({priced}/{len(open_positions)} priced): "
              f"${sign}{total_pnl:,.2f}")
    else:
        print(f"  Could not price any positions (market may be closed or data unavailable).")
    print("=" * 106)
    print()


def _print_plan(picks: list[dict], capital_budget: Optional[float] = None) -> None:
    """
    Print the trading plan as a formatted table.

    Columns: #  Strategy  Symbol  Price  Expiry  Legs  Qty  Credit  Capital  Prob  ROI  Score  MCap  OI  Vol
    """
    total_capital = sum(_capital_for_pick(p) * p.get('quantity', 1) for p in picks)
    total_credit  = sum(p.get('premium', 0) * 100 * p.get('quantity', 1) for p in picks)

    print()
    print("=" * 150)
    print("  OPTION WHEEL — TRADING PLAN")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(picks)} picks  |  Total premium: ${total_credit:,.2f}  |  "
          f"Total capital required: ${total_capital:,.0f}"
          + (f"  (budget: ${capital_budget:,.0f})" if capital_budget else ""))
    print("=" * 150)

    hdr = f"{'#':>3}  {'Strat':<6}  {'Symbol':<6}  {'Price':>9}  {'Expiry':<10}  {'Legs':<32}  "
    hdr += f"{'Qty':>4}  {'Credit':>7}  {'Capital':>9}  {'Prob':>6}  {'ROI':>6}  {'Score':>6}  "
    hdr += f"{'MCap':>7}  {'OI':>7}  {'Vol':>7}"
    print(hdr)
    print("-" * 150)

    for i, p in enumerate(picks, start=1):
        strat   = p.get('strategy', '?')
        symbol  = p.get('symbol', '?')
        price   = p.get('current_price')
        expiry  = p.get('expiry', '?')
        legs    = _legs_str(p)
        qty     = p.get('quantity', 1)
        credit  = p.get('premium', 0) * 100 * qty   # total dollars
        capital = _capital_for_pick(p) * qty         # total capital
        prob    = p.get('prob_win', 0)
        roi     = p.get('roi', 0)
        score   = p.get('score', 0)
        mcap    = _fmt_mcap(p.get('market_cap'))
        oi      = p.get('short_oi')
        vol     = p.get('short_volume', 0)

        price_str = f"${price:>7.2f}" if price is not None else f"{'N/A':>8}"
        oi_str  = f"{oi:>7,}" if oi is not None else f"{'—':>7}"
        vol_str = f"{vol:>7,}" if vol else f"{'—':>7}"

        print(
            f"{i:>3}. {strat:<6}  {symbol:<6}  {price_str}  {expiry:<10}  {legs:<32}  "
            f"{qty:>4}  ${credit:>6.2f}  ${capital:>8,.0f}  "
            f"{prob:>5.1%}  {roi:>5.1%}  {score:>6.3f}  "
            f"{mcap:>7}  {oi_str}  {vol_str}"
        )

    print("-" * 150)
    print(f"{'TOTAL':<72}  {'':>4}  {'':>7}  ${total_capital:>8,.0f}")
    print("=" * 150)
    print()


# ── Approval gate ──────────────────────────────────────────────────────────────

def _approval_gate(picks: list[dict]):
    """
    Interactively ask the user which picks to approve.

    Input options
    -------------
    a          Approve ALL picks
    n          Reject ALL picks (exit without executing)
    q          Quit the agent entirely
    replan     Discard this plan and request fresh model candidates
    1,3,5      Approve picks by comma-separated number
    1-5        Approve a range of picks
    1,3-5,8   Mix of individual numbers and ranges

    Returns the approved subset of picks, or the string 'REPLAN'.
    """
    if not sys.stdin.isatty():
        # Non-interactive (piped / CI) — safe default: reject all
        print("stdin is not a TTY — running in non-interactive mode.")
        print("Use --mode auto or pipe approval input to run headlessly.")
        return []

    print("  Enter the numbers of the trades to approve, 'a' for all, 'n' for none,")
    print("  or 'replan' to discard this plan and request fresh model candidates.")
    print("  Examples:  a   |   n   |   1,3,5   |   1-5   |   2,4-7,10   |   replan")
    print()

    while True:
        try:
            raw = input("  Your selection: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return []

        if raw in ('q', 'quit'):
            print("Quitting.")
            sys.exit(0)

        if raw in ('n', 'none', ''):
            print("  No picks approved — nothing will be executed.")
            return []

        if raw in ('a', 'all'):
            print(f"  All {len(picks)} picks approved.")
            return picks

        if raw in ('replan', 'rescan', 'retry'):
            print("  REPLAN requested — will request fresh model candidates...")
            return 'REPLAN'

        # Parse individual numbers and ranges
        approved_indices: set[int] = set()
        valid = True
        for part in raw.split(','):
            part = part.strip()
            if '-' in part:
                lo_s, _, hi_s = part.partition('-')
                try:
                    lo, hi = int(lo_s.strip()), int(hi_s.strip())
                    approved_indices.update(range(lo, hi + 1))
                except ValueError:
                    print(f"  Invalid range '{part}'. Try again.")
                    valid = False
                    break
            else:
                try:
                    approved_indices.add(int(part))
                except ValueError:
                    print(f"  Invalid number '{part}'. Try again.")
                    valid = False
                    break

        if not valid:
            continue

        # Map 1-based display numbers back to 0-based list indices
        out_of_range = [n for n in approved_indices if n < 1 or n > len(picks)]
        if out_of_range:
            print(f"  Number(s) out of range (1-{len(picks)}): {out_of_range}. Try again.")
            continue

        approved = [picks[n - 1] for n in sorted(approved_indices)]
        print(f"  {len(approved)} pick(s) approved: {sorted(approved_indices)}")
        return approved


def _confirm_execution(approved: list[dict], dry_run: bool) -> bool:
    """Ask for a final 'yes' before submitting orders."""
    if not sys.stdin.isatty():
        return False

    mode_label = "[DRY RUN]" if dry_run else "[LIVE - REAL MONEY]"
    total_cap  = sum(_capital_for_pick(p) for p in approved)
    total_prem = sum(p.get('premium', 0) * 100 for p in approved)

    print()
    print(f"  {mode_label}  About to submit {len(approved)} order(s).")
    print(f"  Capital required: ${total_cap:,.0f}   |   Premium collected: ${total_prem:,.2f}")
    if not dry_run:
        print("  WARNING: This will submit REAL orders to Alpaca!")
    print()

    try:
        answer = input("  Confirm? [yes / no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False

    return answer in ('yes', 'y')


# ── Execution helpers ──────────────────────────────────────────────────────────

def _execute_picks(
    approved: list[dict],
    executor: AlpacaExecutor,
    db: TradeDatabase,
    dry_run: bool,
) -> list[tuple[dict, Optional[str]]]:
    """Submit each approved pick and log it to the database.

    Returns a list of (pick, order_id) tuples — order_id is None on failure.
    """
    results: list[tuple[dict, Optional[str]]] = []
    print()
    for i, pick in enumerate(approved, start=1):
        strat  = pick.get('strategy', '?')
        symbol = pick.get('symbol', '?')
        expiry = pick.get('expiry', '?')
        short  = (pick.get('short_strike')
                  or pick.get('short_put')
                  or pick.get('short_call')
                  or 0)
        prem   = pick.get('premium', 0)
        prob   = pick.get('prob_win', 0)

        print(f"  [{i}/{len(approved)}] {strat} {symbol} {expiry}  "
              f"credit=${prem:.2f}  prob={prob:.1%} ...", end=" ", flush=True)

        # ── Phase 1: pre-insert PENDING row before touching the broker ─────────
        # If the process dies between broker submission and DB write the row
        # is already present (PENDING) and will be visible for reconciliation
        # rather than silently lost.
        if dry_run:
            # Dry-run: log immediately as DRY_RUN (no broker call, no two-phase)
            trade_id = db.log_trade(
                symbol, expiry, short, strat, prem, prob,
                'DRY_RUN', 'DRY_RUN',
                legs=_legs_from_pick(pick),
                contracts=pick.get('quantity', 1),
                model_prediction_id=pick.get('model_prediction_id'),
                model_decision_id=pick.get('model_decision_id'),
            )
            print(f"OK  (DRY_RUN)")
            results.append((pick, 'DRY_RUN'))
            continue

        trade_id = db.log_trade(
            symbol, expiry, short, strat, prem, prob,
            'PENDING', None,
            legs=_legs_from_pick(pick),
            contracts=pick.get('quantity', 1),
            model_prediction_id=pick.get('model_prediction_id'),
            model_decision_id=pick.get('model_decision_id'),
        )

        # ── Phase 2: submit to broker ──────────────────────────────────────────
        order_id = None
        try:
            order_id = executor.execute_pick(pick, dry_run=False, amount=pick.get('quantity', 1))
        except Exception as exc:
            log.error("FAILED  (%s: %s)", type(exc).__name__, exc, exc_info=True)

        if not order_id:
            db.void_trade(trade_id)
            log.warning("[agent] Broker submission failed for %s %s — PENDING row voided.",
                        strat, symbol)
            print("FAILED")
            results.append((pick, None))
            continue

        # ── Phase 3: record accepted order, then confirm only after fill ──────
        db.record_open_order(trade_id, order_id)

        # Poll Alpaca for the actual fill price before marking EXECUTED so
        # that stop-loss calculations use the real entry premium rather than
        # the (potentially hours-old) scanned value.  If Alpaca reports the
        # order as canceled/expired, void the record so the monitor does not
        # treat it as an open position.
        fill, canceled = executor.get_fill_price(order_id)
        if canceled:
            db.void_trade(trade_id)
            log.warning("[agent] Order %s for %s %s was not filled — "
                        "DB entry voided.", order_id, strat, symbol)
            print(f"NOT FILLED  ({order_id})")
            results.append((pick, None))
            continue
        if fill is None:
            log.warning(
                "[agent] Order %s for %s %s accepted but fill is unconfirmed — "
                "leaving DB row PENDING for reconciliation.",
                order_id, strat, symbol,
            )
            print(f"PENDING  ({order_id})")
            results.append((pick, order_id))
            continue
        if fill is not None:
            # Alpaca returns filled_avg_price as a negative value for
            # net-credit MLEG orders (PCS/CCS/IC/IFLY/STRANGLE).
            # Normalise to a positive credit before storing.
            fill = abs(fill)
        db.confirm_open(trade_id, order_id)
        if abs(fill - prem) > 0.001:
            db.update_premium(trade_id, fill)
            pick['premium'] = fill
            log.info("[agent] Fill price for %s %s: $%.2f (scanned $%.2f)",
                     strat, symbol, fill, prem)

        print(f"OK  ({order_id})")

        results.append((pick, order_id))

    print()
    return results


# ── Universe helpers ───────────────────────────────────────────────────────────

def _count_enabled_strategies(config: dict) -> int:
    """Return the number of enabled strategies in config (minimum 1)."""
    strategies = config.get('strategies', {})
    count = sum(1 for v in strategies.values() if v.get('enabled', False))
    return max(count, 1)


def _load_tickers(args, config: dict) -> list[str]:
    """Return the ticker list based on --universe / --tickers CLI args."""
    if args.tickers:
        return args.tickers

    # CLI flag > config["universe"] > "etf" (hard default)
    universe = args.universe or config.get('universe', 'etf')

    if universe == 'etf':
        from src.universe import get_etf_universe
        print("Universe: loading all NASDAQ / NYSE / NYSE Arca ETFs ...")
        tickers = get_etf_universe(force_refresh=args.refresh_universe, log=log)
        print(f"Universe: {len(tickers)} ETF tickers ready")
        return tickers

    if universe == 'index':
        from src.universe_indices import get_index_tickers
        print("Universe: loading S&P 500 + NASDAQ-100 + Dow 30 constituents...")
        tickers = get_index_tickers(force_refresh=args.refresh_universe)
        print(f"Universe: {len(tickers)} index tickers ready")
        return tickers

    # default / full: fall through to get_ticker_universe (NASDAQ/NYSE by market cap)
    from src.universe import get_ticker_universe
    min_cap = config.get('market_cap_min', 1_000_000_000)
    print(f"Universe: loading NASDAQ/NYSE tickers with market cap >= ${min_cap:,.0f} ...")
    tickers = get_ticker_universe(min_cap, log=log)
    print(f"Universe: {len(tickers)} tickers loaded")
    return tickers


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='agent.py',
        description='OptionMind Agent — score model candidates, approve, and execute option trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mode',
        choices=['approve', 'auto', 'scan-only'],
        default=None,        # None = read from config
        metavar='MODE',
        help=(
            'Execution mode: '
            '"approve" (default) — show plan, ask for approval, execute approved picks; '
            '"auto" — execute picks above auto_execute_prob without prompting; '
            '"scan-only" — print model-ranked plan and save to data/pending_picks.json, never execute.'
        ),
    )
    parser.add_argument(
        '--dry-run', action='store_true', dest='dry_run',
        help='Simulate orders without submitting to Alpaca (default when --live is absent).',
    )
    parser.add_argument(
        '--live', action='store_true', dest='live',
        help='Submit REAL orders to Alpaca.  Requires valid credentials in config.json.',
    )
    parser.add_argument(
        '--universe',
        choices=['etf', 'index', 'default', 'full'],
        default=None,           # None = read from config["universe"], fallback "etf"
        metavar='SRC',
        help=(
            '"etf" (default) — all ETFs on NASDAQ / NYSE / NYSE Arca; '
            '"index" — S&P 500 + NASDAQ-100 + Dow 30 (~540 tickers); '
            '"default" / "full" — NASDAQ/NYSE stocks by market cap.'
        ),
    )
    parser.add_argument(
        '--refresh-universe', action='store_true', dest='refresh_universe',
        help='Force-refresh the cached index constituent list (only with --universe index).',
    )
    parser.add_argument(
        '--tickers', nargs='+', metavar='SYM', default=None,
        help='Explicit ticker list (overrides --universe).',
    )
    parser.add_argument(
        '--top-n', type=int, default=None, metavar='N', dest='top_n',
        help=(
            'Total number of model candidates to surface '
            '(overrides config ml_scanner.top_n).'
        ),
    )
    parser.add_argument(
        '--max-capital', type=float, default=None, metavar='DOLLARS', dest='max_capital',
        help='Cap total capital per period in dollars (overrides config max_capital_per_period).',
    )
    parser.add_argument(
        '--config', default='config.json', metavar='PATH',
        help='Path to config.json (default: config.json).',
    )
    parser.add_argument(
        '--db', default='data/trades.db', metavar='PATH', dest='db_path',
        help='Path to the SQLite trades database (default: data/trades.db).',
    )
    parser.add_argument(
        '--scanner',
        choices=['ml', 'legacy'],
        default=None,       # None = read from config["scanner"], fallback "ml"
        dest='scanner',
        metavar='TYPE',
        help=(
            'Scanner to use: "ml" (default) — ML inference engine with the champion '
            'registry model (LivePaperInferenceProvider); '
            '"legacy" — deterministic rule-based scanner (useful for side-by-side '
            'comparison with an already-running legacy instance).'
        ),
    )

    # ── Manual close options (short-circuit the normal candidate flow) ────────
    close_group = parser.add_argument_group(
        'manual close',
        'Close open positions immediately without requesting model candidates.',
    )
    close_group.add_argument(
        '--close', nargs='+', type=int, metavar='ID', dest='close_ids',
        help='Close one or more open positions by trade ID (e.g. --close 7 12 15).',
    )
    close_group.add_argument(
        '--close-all', action='store_true', dest='close_all',
        help='Close ALL currently open positions.',
    )
    close_group.add_argument(
        '--list-open', action='store_true', dest='list_open',
        help='Print all open positions and exit (no scan, no close).',
    )
    parser.add_argument(
        '--daemon', action='store_true',
        help=(
            'Run continuously, waking at the configured schedule.run_time each day '
            '(weekdays only by default). Fully headless — approval via email only; '
            'if email is not configured or times out, the cycle is skipped.'
        ),
    )
    parser.add_argument(
        '--log-file', default=None, metavar='PATH', dest='log_file',
        help='Write logs to this file with rotation (10 MB × 5 backups). '
             'Default: stdout only.',
    )
    return parser.parse_args()


# ── Manual close helpers ───────────────────────────────────────────────────────

def _print_positions_table(positions: list[dict]) -> None:
    """Print a compact table of open positions (ID, strat, symbol, expiry, premium)."""
    if not positions:
        print("  No open positions found.")
        return
    print(f"  {'ID':>4}  {'Strat':<6}  {'Symbol':<7}  {'Entry':<10}  {'Expiry':<10}  "
          f"{'Premium':>8}  Status")
    print("  " + "-" * 65)
    for p in positions:
        prem = float(p.get('premium', 0) or 0)
        print(f"  {p['id']:>4}  {p.get('type','?'):<6}  {p.get('symbol','?'):<7}  "
              f"{(p.get('timestamp') or '')[:10]:<10}  {p.get('expiry','?'):<10}  "
              f"${prem * 100:>7.2f}  {p.get('status','?')}")
    print()


def _close_one(pos: dict, executor: AlpacaExecutor, db: TradeDatabase,
               monitor: PositionMonitor, dry_run: bool) -> bool:
    """
    Price and close a single position.  Returns True on success.
    Prints a one-line result summary.
    """
    trade_id = pos['id']
    symbol   = pos.get('symbol', '?')
    strat    = pos.get('type',   '?')

    # Try to get a live mark for a sensible limit price
    current_mark = None
    try:
        # conservative=True: use ask for short legs (buy-to-close) and bid for
        # long legs (sell-to-close) so realized P&L reflects actual fill costs.
        current_mark = monitor._get_current_mark(pos, conservative=True)
    except Exception:
        pass

    limit_px = round(current_mark, 2) if current_mark is not None else None

    contracts = int(pos.get('contracts') or 1)
    premium  = float(pos.get('premium', 0) or 0)
    pnl      = round((premium - current_mark) * 100 * contracts, 2) if current_mark is not None else 0.0
    from src.position_lifecycle import PositionLifecycleService
    result = PositionLifecycleService(db, executor).close_position(
        pos,
        limit_price=limit_px,
        pnl=pnl,
        dry_run=dry_run,
        reason='MANUAL_CLOSE',
    )
    if not result.success:
        log.error("[%s] %s %s — close failed: %s", trade_id, strat, symbol, result.error)
        return False

    tag      = ' [DRY RUN]' if dry_run else ''
    pnl_sign = '+' if pnl >= 0 else ''
    mark_str = f"${current_mark:.4f}" if current_mark is not None else 'N/A'
    action = 'closed' if result.status == 'CLOSED' else 'close submitted'
    pending = ' (already pending)' if result.already_pending else ''
    print(f"  ✓  [{trade_id}] {strat} {symbol} {action}{tag}{pending}  "
          f"mark={mark_str}  est P&L={pnl_sign}${pnl:,.2f}  order={result.order_id or '—'}")
    return True


def _run_close_command(args, db: TradeDatabase, executor: AlpacaExecutor,
                       monitor: PositionMonitor, dry_run: bool) -> None:
    """
    Handle --list-open, --close <ids>, and --close-all.
    Exits after completing the requested action.
    """
    open_positions = db.get_open_positions()

    # ── --list-open ───────────────────────────────────────────────────────────
    if args.list_open:
        print()
        print("=" * 72)
        print(f"  OPEN POSITIONS  ({len(open_positions)} found)")
        print("=" * 72)
        _print_positions_table(open_positions)
        print("=" * 72)
        sys.exit(0)

    # ── --close <ids> ─────────────────────────────────────────────────────────
    if args.close_ids:
        by_id = {p['id']: p for p in open_positions}
        missing = [i for i in args.close_ids if i not in by_id]
        if missing:
            print(f"  [WARN] Trade ID(s) not found among open positions: {missing}")

        targets = [by_id[i] for i in args.close_ids if i in by_id]
        if not targets:
            print("  Nothing to close.")
            sys.exit(1)

        tag = ' [DRY RUN]' if dry_run else ' [LIVE]'
        print()
        print("=" * 72)
        print(f"  MANUAL CLOSE{tag}  —  {len(targets)} position(s)")
        print("=" * 72)
        _print_positions_table(targets)

        # Confirmation prompt (skip in dry-run)
        if not dry_run:
            ans = input(f"  Close {len(targets)} position(s) with REAL orders? [y/N] ").strip().lower()
            if ans != 'y':
                print("  Aborted.")
                sys.exit(0)

        ok = sum(_close_one(p, executor, db, monitor, dry_run) for p in targets)
        print()
        print(f"  Closed {ok}/{len(targets)} position(s).")
        print("=" * 72)
        sys.exit(0 if ok == len(targets) else 1)

    # ── --close-all ───────────────────────────────────────────────────────────
    if args.close_all:
        if not open_positions:
            print("  No open positions to close.")
            sys.exit(0)

        tag = ' [DRY RUN]' if dry_run else ' [LIVE]'
        print()
        print("=" * 72)
        print(f"  CLOSE ALL POSITIONS{tag}  —  {len(open_positions)} position(s)")
        print("=" * 72)
        _print_positions_table(open_positions)

        if not dry_run:
            ans = input(f"  Close ALL {len(open_positions)} position(s) with REAL orders? [y/N] ").strip().lower()
            if ans != 'y':
                print("  Aborted.")
                sys.exit(0)

        ok = sum(_close_one(p, executor, db, monitor, dry_run) for p in open_positions)
        total_pnl = 0.0  # already printed per-line above
        print()
        print(f"  Closed {ok}/{len(open_positions)} position(s).")
        print("=" * 72)
        sys.exit(0 if ok == len(open_positions) else 1)


# ── Daemon scheduler ───────────────────────────────────────────────────────────

def _next_run_dt(run_time: str, tz_name: str, weekdays_only: bool) -> datetime:
    """
    Return the next wall-clock datetime at which the agent should fire.

    ``run_time`` is a 24-h "HH:MM" string.  The result is always in the
    future (at least 1 second from now).  Tries zoneinfo (stdlib, Python 3.9+)
    then pytz, then falls back to local system time.
    """
    tz = None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        try:
            import pytz
            tz = pytz.timezone(tz_name)
        except Exception:
            pass  # last resort: system local time

    hh, mm = (int(p) for p in run_time.split(':'))
    now = datetime.now(tz)

    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)

    if weekdays_only:
        # 0=Monday … 4=Friday  5=Saturday  6=Sunday
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)

    return candidate


def run_daemon(args) -> None:
    """
    Loop forever, waking at the configured schedule time each day to call
    _run_once().  Fully headless — if email is not configured or the
    approval reply times out, the cycle is skipped rather than blocking.
    """
    config     = load_config(args.config)
    sched_cfg  = config.get('schedule', {})
    run_time   = sched_cfg.get('run_time', '09:35')
    tz_name    = sched_cfg.get('timezone', 'US/Eastern')
    weekdays   = bool(sched_cfg.get('weekdays_only', True))

    log = get_logger('optionwheel')
    log.info(
        "[daemon] Starting — will fire at %s %s%s",
        run_time, tz_name,
        " (weekdays only)" if weekdays else "",
    )

    while True:
        next_dt = _next_run_dt(run_time, tz_name, weekdays)
        sleep_s = (next_dt - datetime.now(next_dt.tzinfo)).total_seconds()
        log.info("[daemon] Next run scheduled for %s (in %.0f s / %.1f h)",
                 next_dt.strftime('%Y-%m-%d %H:%M %Z'), sleep_s, sleep_s / 3600)
        time.sleep(max(sleep_s, 0))

        log.info("[daemon] Waking up — starting agent cycle")
        try:
            _run_once(args, headless=True)
        except SystemExit:
            pass   # _run_close_command calls sys.exit; ignore in daemon
        except Exception as exc:
            log.error("[daemon] Cycle failed: %s", exc, exc_info=True)

        # Brief pause so we don't re-trigger within the same minute
        time.sleep(90)


# ── Scanner factory ───────────────────────────────────────────────────────────

def _build_scanner(config: dict, scanner_type: str):
    """Return the configured scanner instance.

    scanner_type == 'ml'     (default)
        Uses LivePaperInferenceProvider directly — loads the champion model from
        the registry and scores short-option candidates with it.  Falls back to
        a configured ml_scanner.provider if one is set in config.
        Automatically injects pick_selection.mode=model_ranked unless the user
        has already set a mode in config.

    scanner_type == 'legacy'
        Uses the deterministic OptionScanner from src.scanner.  Useful for
        running side-by-side with the ML agent to compare candidate quality.
    """
    if scanner_type == 'legacy':
        from src.scanner import OptionScanner
        log.info("[agent] Scanner: legacy deterministic (OptionScanner)")
        return OptionScanner(config)

    # ML path — default pick_selection to model_ranked unless overridden
    ml_config = dict(config)
    ps = ml_config.get('pick_selection')
    if not isinstance(ps, dict) or 'mode' not in ps:
        ml_config['pick_selection'] = dict(ps or {})
        ml_config['pick_selection'].setdefault('mode', 'model_ranked')

    ml_cfg = ml_config.get('ml_scanner', {})
    explicit_provider = ml_cfg.get('provider')

    if explicit_provider:
        from src.model_scanner import ModelScanner
        log.info("[agent] Scanner: ML (ModelScanner, provider=%s)", explicit_provider)
        return ModelScanner(ml_config)

    from src.model_scanner import LivePaperInferenceProvider
    try:
        scanner = LivePaperInferenceProvider(ml_config)
        log.info(
            "[agent] Scanner: ML (LivePaperInferenceProvider, registry=%s, mode=model_ranked)",
            ml_cfg.get('registry_path', 'artifacts/model_registry.json'),
        )
        return scanner
    except Exception as exc:
        log.warning(
            "[agent] LivePaperInferenceProvider init failed (%s: %s) — "
            "no ML picks will be generated this run.",
            type(exc).__name__, exc,
        )
        from src.model_scanner import ModelScanner
        return ModelScanner({'ml_scanner': {'enabled': False}})


# ── Main ───────────────────────────────────────────────────────────────────────

def _run_once(args, headless: bool = False) -> None:
    """Single agent cycle: monitor → score candidates → approve → execute."""
    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── Resolve execution mode ────────────────────────────────────────────────
    # CLI --mode > config approve_mode > default 'approve'
    if args.mode is not None:
        mode = args.mode
    else:
        raw = config.get('alpaca', {}).get('approve_mode', True)
        if raw is True:
            mode = 'approve'
        elif raw is False or raw == 'auto':
            mode = 'auto'
        else:
            mode = str(raw)   # 'scan-only' etc. passed as string in config

    # ── Resolve dry-run ───────────────────────────────────────────────────────
    # --live wins over --dry-run; default is dry_run=True (safe)
    if args.live:
        dry_run = False
    else:
        dry_run = True   # --dry-run or neither flag → always safe default

    # ── Resolve top-n and capital cap ─────────────────────────────────────────
    # top_n is the max number of model-ranked candidates requested from the
    # ML scanner hook. Legacy per-strategy quotas are no longer used for
    # candidate generation.
    if args.top_n:
        top_n = args.top_n  # CLI value is treated as the raw total
    else:
        top_n = config.get('ml_scanner', {}).get(
            'top_n',
            config.get('top_n_picks', config.get('top_n_per_strategy', 10)),
        )
    capital_budget = args.max_capital or config.get('max_capital_per_period')

    # ── Bootstrap components ──────────────────────────────────────────────────
    db = TradeDatabase(args.db_path)

    # Self-heal any open positions whose premium was stored as a negative value
    # (Alpaca returns filled_avg_price < 0 for net-credit MLEG orders; the
    # abs() guard was added in agent.py after some positions were already
    # recorded).  This is idempotent and a no-op once all records are correct.
    _fixed = db.fix_negative_premiums()
    if _fixed:
        log.warning("[agent] Corrected %d open position(s) with negative entry "
                    "premium (Alpaca MLEG sign convention).", _fixed)

    # Resolve scanner type: CLI --scanner > config["scanner"] > default "ml"
    scanner_type: str = (
        getattr(args, 'scanner', None)
        or config.get('scanner', 'ml')
    )
    scanner  = _build_scanner(config, scanner_type)
    executor = AlpacaExecutor(args.config)

    # ── Fetch current VIX for account-level regime controls ───────────────────
    current_vix: Optional[float] = None
    vix_cfg = config.get('risk_parameters', {}).get('vix_filter', {})
    if vix_cfg.get('enabled', False):
        _vix = _fetch_vix(config, log)
        if _vix is not None:
            current_vix = _vix
            log.info("VIX=%0.1f captured for regime/risk controls.", current_vix)
        else:
            log.warning("Could not fetch VIX level from any source — VIX-based regime controls will not apply.")

    # ── Settle expired positions first (pure bookkeeping, no orders) ─────────
    monitor  = PositionMonitor(db, executor, config)
    notifier = EmailNotifier(config)
    log.info("Email notifier %s.", "enabled" if notifier.enabled else "disabled")

    # ── Short-circuit for manual close / list-open commands ──────────────────
    if args.close_ids or args.close_all or args.list_open:
        _run_close_command(args, db, executor, monitor, dry_run)
        # _run_close_command always calls sys.exit(); this line is not reached.
        return

    monitor.settle_expired()

    # A prior close can have been submitted/filled by the monitor daemon, the
    # dashboard, or directly in Alpaca.  Reconcile stale state before the risk
    # pass and before using DB rows for dedup and budget.
    _reconcile_positions_before_budget(db, executor, config)

    # ── Stop-loss monitor: check open positions before considering new ones ───
    closed_positions = monitor.run(dry_run=dry_run)
    for _cp in closed_positions:
        notifier.send_position_closed(_cp, _cp.get('reason_tag', 'STOP_LOSS'))

    # ── Snapshot open positions for dedup + capital accounting ───────────────
    open_positions = db.get_open_positions()
    pending_close_positions = db.get_pending_close_positions()
    capital_positions = open_positions + pending_close_positions

    # Keys used to deduplicate new picks: same symbol + strategy already held
    open_keys: set[tuple[str, str]] = {
        (p['symbol'], p['type']) for p in capital_positions
    }

    # Capital already committed to open positions — subtracted from budget
    deployed_capital = sum(_capital_for_position(p) for p in capital_positions)

    if capital_positions:
        log.info(
            f"{len(open_positions)} open position(s), {len(pending_close_positions)} pending close(s) found — "
            f"${deployed_capital:,.0f} capital already deployed."
        )

    regime = _evaluate_regime_filter(config, log, current_vix=current_vix)
    if regime.pause_new_trades:
        log.warning(
            "Regime filter is RED — skipping new trades this run. Existing "
            "positions were still monitored and reconciled."
        )
        return

    scan_top_n = top_n
    if regime.top_n_multiplier < 1.0:
        scan_top_n = max(1, int(top_n * regime.top_n_multiplier))
        log.info(
            "Regime filter reduced candidate top-N from %d to %d.",
            top_n, scan_top_n,
        )

    # ── Load ticker universe ──────────────────────────────────────────────────
    tickers = _load_tickers(args, config)
    if not tickers:
        log.error("No tickers loaded — exiting.")
        return

    # ── Candidate scoring ─────────────────────────────────────────────────────
    scanner_label = 'ML' if scanner_type == 'ml' else 'legacy'
    log.info(
        "Requesting up to %d %s-ranked candidates from %d tickers ...",
        scan_top_n, scanner_label, len(tickers),
    )
    picks = scanner.get_top_picks(tickers, n=scan_top_n)
    # Tag picks with their scanner source for audit/comparison
    default_source = 'ml_model' if scanner_type == 'ml' else 'legacy_scanner'
    for _p in picks:
        _p.setdefault('source', default_source)
    risk_rejected: list[dict] = []
    _annotate_mispricing_scores(picks)
    if scanner_type == 'ml':
        _record_model_predictions(db, picks)

    if not picks:
        if scanner_type == 'ml':
            log.info(
                "No ML candidates returned — check that the champion model registry "
                "exists at ml_scanner.registry_path and that Alpaca credentials are set."
            )
        else:
            log.info("No legacy scanner candidates returned.")
        _write_scan_audit([], [], db=db, scanner_type=scanner_type)
        return

    # ── Deduplicate against open positions ───────────────────────────────────
    if open_keys:
        kept_picks: list[dict] = []
        skipped = 0
        for p in picks:
            if (p['symbol'], p['strategy']) in open_keys:
                item = dict(p)
                item['filtered_stage'] = 'Open-position dedup'
                item['reject_reason'] = 'Same symbol and strategy already held as an open or pending-close position'
                item['mispricing_score'] = _mispricing_score_for_pick(item)
                risk_rejected.append(item)
                skipped += 1
            else:
                kept_picks.append(p)
        picks = kept_picks
        if skipped:
            log.info(
                f"Dedup: skipped {skipped} pick(s) already held as open positions."
            )

    if not picks:
        log.info("All picks are already held as open positions — nothing new to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return

    # ── Pre-flight: filter picks whose contracts are inactive on Alpaca ───────
    # Runs whenever Alpaca credentials are present (even in dry-run) so the
    # plan shows only contracts that can actually be traded.  Skips silently
    # when credentials are absent (yfinance-only mode).
    picks, _rejected = executor.preflight_check_picks(picks)
    for _pick in _rejected:
        _pick['filtered_stage'] = 'Pre-flight contract validation'
        bad = ', '.join(_pick.get('_inactive_contracts') or [])
        _pick['reject_reason'] = f"Inactive/non-tradeable contract leg(s): {bad or 'unknown'}"
        _pick['mispricing_score'] = _mispricing_score_for_pick(_pick)
        risk_rejected.append(_pick)
    if not picks:
        log.info("No picks survived pre-flight contract validation — nothing to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return

    before_gate = list(picks)
    picks = _filter_max_loss_multiple(picks, config)
    _capture_rejections(
        before_gate,
        picks,
        'Max-loss multiple',
        lambda p: (
            f"Max-loss multiple {p.get('max_loss_multiple', _max_loss_multiple_for_pick(p)):.2f}x "
            "exceeded configured limit"
        ),
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived max-loss multiple filtering — nothing to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return

    # ── Apply capital budget (net of already-deployed capital) ───────────────
    if capital_budget is not None:
        remaining_budget = capital_budget - deployed_capital
        if deployed_capital > 0:
            log.info(
                f"Capital budget ${capital_budget:,.0f} — "
                f"${deployed_capital:,.0f} already deployed → "
                f"${remaining_budget:,.0f} available for new picks."
            )
        if remaining_budget <= 0:
            log.info("No remaining capital budget for new picks.")
            for _pick in picks:
                _item = dict(_pick)
                _item['filtered_stage'] = 'Capital budget'
                _item['reject_reason'] = 'No remaining capital budget after open positions'
                _item['mispricing_score'] = _mispricing_score_for_pick(_item)
                risk_rejected.append(_item)
            _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
            return
        max_contracts = int(config.get('max_contracts_per_pick', 50))
        picks_sorted  = sorted(picks, key=lambda x: x.get('score', 0.0), reverse=True)
        # Keep only picks we can afford at least 1 contract of
        affordable = [p for p in picks_sorted if _capital_for_pick(p) <= remaining_budget]
        _capture_rejections(
            picks_sorted,
            affordable,
            'Capital budget',
            lambda p: f"Capital requirement ${_capital_for_pick(p):,.0f} exceeded remaining budget ${remaining_budget:,.0f}",
            risk_rejected,
        )
        if not affordable:
            log.info("No picks fit within the remaining capital budget.")
            _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
            return
        # Equal-split remaining budget across affordable picks, then size each
        per_pick_alloc = remaining_budget / len(affordable)
        new_deployed   = 0.0
        for p in affordable:
            cap = _capital_for_pick(p)
            qty = max(1, int(per_pick_alloc // cap))
            qty = min(qty, max_contracts)
            p['quantity'] = qty
            new_deployed += cap * qty
        log.info(
            f"Capital budget: {len(affordable)} picks, {sum(p['quantity'] for p in affordable)} "
            f"total contracts — ${new_deployed:,.0f} of ${remaining_budget:,.0f} available deployed "
            f"(max {max_contracts} contracts/pick)"
        )
        picks = affordable

    picks = _apply_regime_quantity_multiplier(picks, regime)
    if not picks:
        log.info("No picks survived regime quantity throttle — nothing to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return
    if regime.quantity_multiplier < 1.0:
        log.info(
            "Regime filter applied %.0f%% quantity throttle to %d pick(s).",
            regime.quantity_multiplier * 100,
            len(picks),
        )

    account_capital = config.get('account_capital') or capital_budget
    before_gate = list(picks)
    picks = _apply_directional_exposure_caps(
        picks, capital_positions, config, account_capital,
    )
    _capture_rejections(
        before_gate,
        picks,
        'Directional exposure cap',
        'Position would exceed configured put/call side exposure limits',
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived directional exposure caps — nothing to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return

    before_gate = list(picks)
    picks = _apply_portfolio_gamma_risk(
        picks, capital_positions, config, account_capital, monitor,
    )
    _capture_rejections(
        before_gate,
        picks,
        'Portfolio gamma risk',
        'Position would exceed portfolio stress/gamma concentration limits',
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived portfolio gamma-risk controls — nothing to trade.")
        _write_scan_audit([], risk_rejected, db=db, scanner_type=scanner_type)
        return

    _annotate_mispricing_scores(picks)
    _write_scan_audit(picks, risk_rejected, db=db, scanner_type=scanner_type)

    # ── Print open positions with current P&L ────────────────────────────────
    if open_positions:
        _print_open_positions(open_positions, monitor)

    # ── Print plan ────────────────────────────────────────────────────────────
    _print_plan(picks, capital_budget=capital_budget)

    # ── scan-only mode ────────────────────────────────────────────────────────
    if mode == 'scan-only':
        os.makedirs('data', exist_ok=True)
        out_path = 'data/pending_picks.json'
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(picks, fh, indent=2, default=str)
        log.info(f"Scan-only mode: {len(picks)} picks saved to {out_path}.")
        return

    # ── auto mode ─────────────────────────────────────────────────────────────
    if mode == 'auto':
        risk    = config.get('risk_parameters', {})
        thresh  = risk.get('auto_execute_prob', 0.90)
        approved = [p for p in picks if p.get('prob_win', 0) >= thresh]
        log.info(
            f"Auto mode: {len(approved)}/{len(picks)} picks meet prob >= {thresh:.0%}"
        )
        if not approved:
            log.info("No picks meet the auto-execute threshold — nothing submitted.")
            return
        exec_results = _execute_picks(approved, executor, db, dry_run)
        for _pick, _oid in exec_results:
            if _oid:
                notifier.send_trade_executed(_pick, _oid, 'AUTO')
        return

    # ── approve mode — replan loop ────────────────────────────────────────────
    # The loop runs once normally, but on a REPLAN signal it requests fresh
    # model candidates, re-prints the plan, and waits for a new approval.
    # max_replan_attempts (default 3) caps the number of fresh requests.
    max_replans = int(config.get('email', {}).get('max_replan_attempts', 3))
    replan_count = 0

    while True:
        # Stamp pre-computed capital onto each pick so the notifier uses the
        # same value as the terminal display (strike-based, × 100).
        for _p in picks:
            _p['capital'] = _capital_for_pick(_p) * _p.get('quantity', 1)

        # Collect the approval decision (list of approved picks, 'REPLAN', or None)
        approved_or_signal = None
        if notifier.enabled:
            _msg_id = notifier.send_trade_plan(picks, capital_budget,
                                               deployed_capital=deployed_capital)
            if _msg_id:
                _result = notifier.wait_for_approval(_msg_id, picks)
                if _result == 'REPLAN':
                    approved_or_signal = 'REPLAN'
                elif _result is not None:
                    approved_or_signal = _result
                elif headless:
                    log.warning(
                        "Email approval timed out in headless mode — skipping this cycle. "
                        "Increase email.approval_timeout_seconds or switch to --mode auto."
                    )
                    return
                else:
                    log.info("Email approval timed out — falling back to TTY.")
                    approved_or_signal = _approval_gate(picks)
            elif headless:
                log.warning("Trade plan email failed to send in headless mode — skipping cycle.")
                return
            else:
                log.warning("Trade plan email failed to send — falling back to TTY.")
                approved_or_signal = _approval_gate(picks)
        elif headless:
            log.warning(
                "Running headless (--daemon) without email configured — skipping cycle. "
                "Add email credentials to config.json or use --mode auto."
            )
            return
        else:
            approved_or_signal = _approval_gate(picks)

        # ── Handle REPLAN ─────────────────────────────────────────────────────
        if approved_or_signal == 'REPLAN':
            replan_count += 1
            if replan_count > max_replans:
                log.warning(
                    "[agent] Max replan attempts (%d) reached — aborting cycle.",
                    max_replans,
                )
                return
            log.info(
                "[agent] REPLAN requested (attempt %d/%d) — requesting fresh model candidates...",
                replan_count, max_replans,
            )
            print(f"\n  Requesting fresh model candidates (attempt {replan_count}/{max_replans}) ...\n")

            new_picks = scanner.get_top_picks(tickers, n=top_n)
            if not new_picks:
                log.info("[agent] Fresh model request returned no picks — aborting cycle.")
                return
            for _p in new_picks:
                _p.setdefault('source', default_source)
            _annotate_mispricing_scores(new_picks)
            if scanner_type == 'ml':
                _record_model_predictions(db, new_picks)
            if open_keys:
                new_picks = [p for p in new_picks
                             if (p['symbol'], p['strategy']) not in open_keys]
            if not new_picks:
                log.info("[agent] All fresh model candidates already held as open positions — aborting.")
                return
            new_picks, _ = executor.preflight_check_picks(new_picks)
            if not new_picks:
                log.info("[agent] No picks survived pre-flight after fresh model request — aborting.")
                return
            new_picks = _filter_max_loss_multiple(new_picks, config)
            if not new_picks:
                log.info("[agent] No picks survived max-loss multiple filtering after fresh model request — aborting.")
                return
            if capital_budget is not None:
                remaining_budget = capital_budget - deployed_capital
                if remaining_budget <= 0:
                    log.info("[agent] No remaining capital budget after fresh model request — aborting.")
                    return
                _sorted = sorted(new_picks, key=lambda x: x.get('score', 0.0), reverse=True)
                budgeted: list[dict] = []
                _new_dep = 0.0
                for _p in _sorted:
                    _cap = _capital_for_pick(_p)
                    if _new_dep + _cap <= remaining_budget:
                        budgeted.append(_p)
                        _new_dep += _cap
                new_picks = budgeted
            if not new_picks:
                log.info("[agent] No picks fit within capital budget after fresh model request — aborting.")
                return
            new_picks = _apply_directional_exposure_caps(
                new_picks, capital_positions, config, account_capital,
            )
            if not new_picks:
                log.info("[agent] No picks survived directional exposure caps after fresh model request — aborting.")
                return
            new_picks = _apply_portfolio_gamma_risk(
                new_picks, capital_positions, config, account_capital, monitor,
            )
            if not new_picks:
                log.info("[agent] No picks survived portfolio gamma-risk controls after fresh model request — aborting.")
                return

            picks = new_picks
            _print_plan(picks, capital_budget=capital_budget)
            continue   # ← loop back to approval with fresh picks

        # ── Normal execution ──────────────────────────────────────────────────
        approved = approved_or_signal or []
        if not approved:
            return

        # In headless mode the email reply IS the confirmation; skip the TTY prompt.
        if not headless and not _confirm_execution(approved, dry_run):
            print("  Execution cancelled.")
            return

        exec_results = _execute_picks(approved, executor, db, dry_run)
        for _pick, _oid in exec_results:
            if _oid:
                notifier.send_trade_executed(_pick, _oid, 'TRADE_PLAN')

        # Save the full plan (including rejected picks) for auditing
        os.makedirs('data', exist_ok=True)
        with open('data/pending_picks.json', 'w', encoding='utf-8') as fh:
            json.dump(picks, fh, indent=2, default=str)
        log.info("Full plan saved to data/pending_picks.json for audit trail.")
        break  # normal exit from replan loop


def run_agent(argv=None) -> None:
    try:
        from dotenv import load_dotenv as _load_dotenv
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        _load_dotenv(_env_path, override=False)
    except ImportError:
        pass
    args = _parse_args()
    if args.log_file:
        get_logger('optionwheel', log_file=args.log_file)
    if args.daemon:
        run_daemon(args)
    else:
        _run_once(args, headless=False)


if __name__ == "__main__":
    run_agent()
