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
