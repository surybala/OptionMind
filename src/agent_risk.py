"""
agent_risk.py
=============

Risk calculations, sizing, and deterministic filters applied to model
candidates before they reach the approval/execution layer.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

from typing import Optional

from src.portfolio_risk import PortfolioRiskService
from src.position_monitor import PositionMonitor
from src.utils import get_logger

log = get_logger()


# ── Per-pick capital & loss helpers ──────────────────────────────────────────

def capital_for_pick(pick: dict) -> float:
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


def strategy_sides(strategy: str) -> set[str]:
    """Return directional risk side(s) for a strategy."""
    strat = (strategy or '').upper()
    if strat in ('CSP', 'PCS'):
        return {'put'}
    if strat in ('CC', 'CCS'):
        return {'call'}
    if strat in ('IC', 'IFLY', 'STRANGLE'):
        return {'put', 'call'}
    return set()


def pick_width(pick: dict) -> float:
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


def max_loss_per_contract(pick: dict) -> float:
    """Return estimated max loss per contract in dollars."""
    premium = max(0.0, float(pick.get('premium') or 0))
    width = pick_width(pick)
    if width <= 0:
        return 0.0
    return max(0.0, round((width - premium) * 100, 2))


def max_loss_multiple(pick: dict) -> float:
    credit = max(0.0, float(pick.get('premium') or 0) * 100)
    if credit <= 0:
        return float('inf')
    return round(max_loss_per_contract(pick) / credit, 4)


# ── Position-level helpers ───────────────────────────────────────────────────

def max_loss_for_position(pos: dict) -> float:
    """Estimate max remaining strategy loss for an open DB position."""
    import json as _json
    strat = (pos.get('type') or '').upper()
    legs = pos.get('legs') or {}
    if isinstance(legs, str):
        try:
            legs = _json.loads(legs) or {}
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


def directional_exposure(open_positions: list[dict]) -> dict[str, float]:
    exposure = {'put': 0.0, 'call': 0.0}
    for pos in open_positions:
        sides = strategy_sides(pos.get('type', ''))
        if not sides:
            continue
        loss = max_loss_for_position(pos)
        share = loss / len(sides)
        for side in sides:
            exposure[side] += share
    return {k: round(v, 2) for k, v in exposure.items()}


# ── Risk filters ─────────────────────────────────────────────────────────────

def filter_max_loss_multiple(picks: list[dict], config: dict) -> list[dict]:
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
        multiple = max_loss_multiple(pick)
        pick['max_loss_multiple'] = multiple
        pick['max_loss_per_contract'] = max_loss_per_contract(pick)
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


def apply_directional_exposure_caps(
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
    used = directional_exposure(open_positions)
    capped: list[dict] = []

    for pick in sorted(picks, key=lambda x: x.get('score', 0.0), reverse=True):
        sides = strategy_sides(pick.get('strategy', ''))
        if not sides:
            capped.append(pick)
            continue
        per_contract_loss = max_loss_per_contract(pick)
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


def apply_portfolio_gamma_risk(
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


def apply_regime_quantity_multiplier(
    picks: list[dict],
    regime,
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
