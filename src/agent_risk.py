"""
agent_risk.py
=============

Risk calculations, sizing, and deterministic filters applied to model
candidates before they reach the approval/execution layer.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

import json
from typing import Optional

from src.capital import capital_for_position
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


def pick_prob_expiry(pick: dict) -> float | None:
    """Return the expiry/profit probability used for max-loss tiering."""
    for key in ('prob_expiry', 'probability_of_profit', 'prob_win'):
        raw = pick.get(key)
        if raw is None:
            continue
        try:
            prob = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= prob <= 1.0:
            return prob
    return None


def max_loss_multiple_limit(config: dict, pick: dict) -> float:
    """
    Resolve the allowed max-loss multiple for a pick.

    The base cap is strategy-specific, but very high ``prob_expiry`` trades can
    opt into a looser limit via configured probability tiers so the bot can
    still trade when all available credits are thin.
    """
    cfg = config.get('risk_parameters', {}).get('max_loss_multiple', {})
    default_limit = float(cfg.get('default', cfg.get('limit', 6.0)))
    by_strategy = cfg.get('by_strategy', {})
    strat = (pick.get('strategy') or '').upper()
    try:
        limit = float(by_strategy.get(strat, default_limit))
    except (TypeError, ValueError):
        limit = default_limit

    prob_expiry = pick_prob_expiry(pick)
    tiers = cfg.get('prob_expiry_tiers', [])
    if prob_expiry is None or not isinstance(tiers, list):
        return limit

    for tier in tiers:
        if not isinstance(tier, dict):
            continue
        try:
            min_prob = float(
                tier.get('min_prob_expiry', tier.get('min_prob', tier.get('prob_expiry', -1.0)))
            )
        except (TypeError, ValueError):
            continue
        if prob_expiry < min_prob:
            continue

        candidate = tier.get('by_strategy', {}).get(strat, tier.get('limit'))
        if candidate is None:
            continue
        try:
            limit = max(limit, float(candidate))
        except (TypeError, ValueError):
            continue
    return limit


def overlay_exposure_per_contract(pick: dict) -> float:
    """Overlay budgets use max loss when available, not gross width."""
    max_loss = max_loss_per_contract(pick)
    if max_loss > 0:
        return max_loss
    return capital_for_pick(pick)


# ── Risk filters ─────────────────────────────────────────────────────────────

def filter_max_loss_multiple(picks: list[dict], config: dict) -> list[dict]:
    cfg = config.get('risk_parameters', {}).get('max_loss_multiple', {})
    if not cfg.get('enabled', True):
        return picks
    kept: list[dict] = []
    rejected = 0
    for pick in picks:
        strat = (pick.get('strategy') or '').upper()
        limit = max_loss_multiple_limit(config, pick)
        multiple = max_loss_multiple(pick)
        pick['max_loss_multiple'] = multiple
        pick['max_loss_per_contract'] = max_loss_per_contract(pick)
        pick['max_loss_multiple_limit'] = limit
        if multiple <= limit:
            kept.append(pick)
        else:
            rejected += 1
            prob_expiry = pick_prob_expiry(pick)
            prob_note = f", prob_expiry={prob_expiry:.1%}" if prob_expiry is not None else ""
            log.info(
                "Max-loss multiple filter: rejected %s %s %.2fx > %.2fx%s "
                "(credit=$%.2f, max_loss=$%.2f/contract).",
                strat, pick.get('symbol'), multiple, limit,
                prob_note,
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


def apply_ml_position_sizing(
    picks: list[dict],
    config: dict,
    *,
    max_contracts: Optional[int] = None,
) -> list[dict]:
    """Request more size for stronger ML-ranked picks before soft overlays."""
    if not picks:
        return []

    contract_cap = max(1, int(max_contracts or config.get('max_contracts_per_pick', 1) or 1))
    cfg = config.get('risk_parameters', {}).get('ml_position_sizing', {})
    if not cfg.get('enabled', True) or contract_cap <= 1:
        adjusted: list[dict] = []
        for raw_pick in picks:
            pick = dict(raw_pick)
            pick['quantity'] = max(1, min(int(pick.get('quantity') or 1), contract_cap))
            adjusted.append(pick)
        return adjusted

    tiers = _ml_position_sizing_tiers(cfg, contract_cap)
    ranked = sorted(enumerate(picks), key=lambda item: item[1].get('score', 0.0), reverse=True)
    adjusted: list[dict | None] = [None] * len(picks)
    for rank, (idx, raw_pick) in enumerate(ranked, start=1):
        pick = dict(raw_pick)
        base_qty = max(1, min(int(pick.get('quantity') or 1), contract_cap))
        suggested_qty = _ml_rank_quantity(rank, tiers, contract_cap)
        requested_qty = max(base_qty, suggested_qty)
        pick['quantity'] = requested_qty
        pick['requested_quantity'] = requested_qty
        pick['requested_quantity_basis'] = 'ml_rank_tiers'
        pick['ml_sizing_rank'] = rank
        adjusted[idx] = pick
    return [pick for pick in adjusted if pick is not None]


def apply_ml_quantity_overlays(
    picks: list[dict],
    capital_positions: list[dict],
    config: dict,
    *,
    account_capital: Optional[float] = None,
    available_capital: Optional[float] = None,
    regime=None,
    max_contracts: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Apply light ML-aligned overlays that resize quantity before rejecting."""
    if not picks:
        return [], []

    account_size = float(
        account_capital
        or config.get('account_capital')
        or config.get('max_capital_per_period')
        or 0.0
    )
    contract_cap = int(max_contracts or config.get('max_contracts_per_pick', 1) or 1)
    side_limits = _directional_limits(config, account_size)
    cluster_limits = _cluster_limits(config, account_size)
    used_sides = _used_side_exposure(capital_positions)
    used_clusters = _used_cluster_exposure(capital_positions, config)
    remaining_capital = float(available_capital) if available_capital is not None else None

    accepted: list[dict] = []
    rejected: list[dict] = []
    ranked = sorted(picks, key=lambda item: item.get('score', 0.0), reverse=True)
    for raw_pick in ranked:
        pick = dict(raw_pick)
        per_contract_risk = capital_for_pick(pick)
        per_contract_overlay_risk = overlay_exposure_per_contract(pick)
        if per_contract_risk <= 0:
            item = dict(pick)
            item['filtered_stage'] = 'Capital budget'
            item['reject_reason'] = 'Capital requirement could not be estimated for this pick'
            rejected.append(item)
            continue

        requested_qty = max(1, min(int(pick.get('quantity') or 1), contract_cap))
        capital_qty = requested_qty
        if remaining_capital is not None:
            capital_qty = min(capital_qty, int(remaining_capital // per_contract_risk))

        side_qty, side_reason = _side_quantity_cap(
            pick,
            requested_qty=requested_qty,
            used_sides=used_sides,
            side_limits=side_limits,
            per_contract_risk=per_contract_overlay_risk,
        )
        cluster_qty, cluster_reason, cluster_keys = _cluster_quantity_cap(
            pick,
            requested_qty=requested_qty,
            used_clusters=used_clusters,
            cluster_limits=cluster_limits,
            per_contract_risk=per_contract_overlay_risk,
        )
        regime_qty = requested_qty
        if regime is not None and getattr(regime, 'quantity_multiplier', 1.0) < 1.0:
            regime_qty = max(1, int(requested_qty * float(regime.quantity_multiplier)))

        final_qty = min(capital_qty, side_qty, cluster_qty, regime_qty)
        if final_qty <= 0:
            item = dict(pick)
            if capital_qty <= 0:
                item['filtered_stage'] = 'Capital budget'
                left = remaining_capital or 0.0
                item['reject_reason'] = (
                    f"Insufficient remaining budget after higher-scored picks "
                    f"(${left:,.0f} left vs ${per_contract_risk:,.0f} required)"
                )
            elif side_qty <= 0:
                item['filtered_stage'] = 'Directional exposure'
                item['reject_reason'] = side_reason or 'Directional side risk budget exhausted'
            elif cluster_qty <= 0:
                item['filtered_stage'] = 'Correlated cluster'
                item['reject_reason'] = cluster_reason or 'Correlated cluster risk budget exhausted'
            else:
                item['filtered_stage'] = 'Regime quantity throttle'
                item['reject_reason'] = 'Regime throttle reduced requested quantity to zero'
            rejected.append(item)
            continue

        pick['requested_quantity'] = requested_qty
        pick['quantity'] = final_qty
        if regime is not None and getattr(regime, 'quantity_multiplier', 1.0) < 1.0:
            pick['regime'] = regime.label
            pick['regime_quantity_multiplier'] = regime.quantity_multiplier
            pick['regime_reduced'] = final_qty < requested_qty
        if side_limits is not None:
            pick['directional_reduced'] = final_qty < requested_qty and final_qty == side_qty
        if cluster_limits is not None and cluster_keys:
            pick['correlated_clusters'] = cluster_keys
            pick['cluster_reduced'] = final_qty < requested_qty and final_qty == cluster_qty

        if remaining_capital is not None:
            remaining_capital -= per_contract_risk * final_qty
        _consume_side_exposure(used_sides, pick, final_qty, per_contract_overlay_risk)
        _consume_cluster_exposure(used_clusters, cluster_keys, per_contract_overlay_risk * final_qty)
        accepted.append(pick)

    return accepted, rejected


def _ml_position_sizing_tiers(cfg: dict, contract_cap: int) -> list[tuple[int, int]]:
    raw_tiers = cfg.get('rank_tiers') or []
    tiers: list[tuple[int, int]] = []
    for item in raw_tiers:
        try:
            max_rank = max(1, int(item.get('max_rank', 0)))
            quantity = max(1, min(int(item.get('quantity', 1)), contract_cap))
        except (AttributeError, TypeError, ValueError):
            continue
        tiers.append((max_rank, quantity))
    if tiers:
        tiers.sort(key=lambda item: item[0])
        return tiers

    if contract_cap <= 1:
        return [(1, 1)]
    return [
        (1, contract_cap),
        (3, max(1, contract_cap - 1)),
        (6, max(1, contract_cap - 2)),
    ]


def _ml_rank_quantity(rank: int, tiers: list[tuple[int, int]], contract_cap: int) -> int:
    for max_rank, quantity in tiers:
        if rank <= max_rank:
            return max(1, min(quantity, contract_cap))
    return 1


def _strategy_sides(strategy: str) -> set[str]:
    normalized = str(strategy or '').upper()
    if normalized in {'PCS', 'CSP', 'STRANGLE'}:
        return {'put'}
    if normalized in {'CCS', 'CC'}:
        return {'call'}
    if normalized in {'IC', 'IFLY'}:
        return {'put', 'call'}
    return set()


def _directional_limits(config: dict, account_capital: float) -> dict[str, float] | None:
    cfg = config.get('risk_parameters', {}).get('directional_exposure_caps', {})
    if not cfg.get('enabled', False):
        return None
    min_side_cap = float(cfg.get('min_side_cap_dollars', 0.0) or 0.0)
    return {
        'put': max(
            float(cfg.get('put', cfg.get('max_put_pct', 0.0))) * account_capital,
            float(cfg.get('min_put_cap_dollars', min_side_cap) or 0.0),
        ),
        'call': max(
            float(cfg.get('call', cfg.get('max_call_pct', 0.0))) * account_capital,
            float(cfg.get('min_call_cap_dollars', min_side_cap) or 0.0),
        ),
    }


def _cluster_limits(config: dict, account_capital: float) -> tuple[dict[str, float], dict[str, list[str]]] | None:
    cfg = config.get('risk_parameters', {}).get('correlated_cluster_caps', {})
    if not cfg.get('enabled', False):
        return None
    cap = max(
        float(cfg.get('max_cluster_pct', 0.0)) * account_capital,
        float(cfg.get('min_cluster_cap_dollars', 0.0) or 0.0),
    )
    raw_clusters = cfg.get('clusters', {}) or {}
    clusters = {
        str(name).upper(): [str(sym).upper() for sym in symbols or []]
        for name, symbols in raw_clusters.items()
    }
    limits = {name: cap for name in clusters}
    return limits, clusters


def _used_side_exposure(capital_positions: list[dict]) -> dict[str, float]:
    used = {'put': 0.0, 'call': 0.0}
    for position in capital_positions:
        sides = _strategy_sides(position.get('type') or position.get('strategy') or '')
        if not sides:
            continue
        total_risk = _position_overlay_exposure(position)
        if total_risk <= 0:
            continue
        per_side = total_risk / len(sides)
        for side in sides:
            used[side] += per_side
    return used


def _used_cluster_exposure(capital_positions: list[dict], config: dict) -> dict[str, float]:
    used: dict[str, float] = {}
    cluster_meta = _cluster_limits(config, 1.0)
    if cluster_meta is None:
        return used
    _, clusters = cluster_meta
    for position in capital_positions:
        symbol = str(position.get('symbol') or position.get('underlying') or '').upper()
        if not symbol:
            continue
        total_risk = _position_overlay_exposure(position)
        if total_risk <= 0:
            continue
        for name, members in clusters.items():
            if symbol in members:
                used[name] = used.get(name, 0.0) + total_risk
    return used


def _side_quantity_cap(
    pick: dict,
    *,
    requested_qty: int,
    used_sides: dict[str, float],
    side_limits: dict[str, float] | None,
    per_contract_risk: float,
) -> tuple[int, str | None]:
    sides = _strategy_sides(pick.get('strategy') or '')
    if not side_limits or not sides or per_contract_risk <= 0:
        return requested_qty, None
    per_side_risk = per_contract_risk / len(sides)
    qty_cap = requested_qty
    reasons: list[str] = []
    for side in sides:
        remaining = side_limits[side] - used_sides.get(side, 0.0)
        side_cap = int(remaining // per_side_risk)
        qty_cap = min(qty_cap, side_cap)
        if side_cap < requested_qty:
            reasons.append(
                f"{side}-side risk budget ${max(remaining, 0.0):,.0f} remaining vs ${per_side_risk:,.0f} per contract"
            )
    return qty_cap, '; '.join(reasons) if reasons else None


def _cluster_quantity_cap(
    pick: dict,
    *,
    requested_qty: int,
    used_clusters: dict[str, float],
    cluster_limits: tuple[dict[str, float], dict[str, list[str]]] | None,
    per_contract_risk: float,
) -> tuple[int, str | None, list[str]]:
    if cluster_limits is None or per_contract_risk <= 0:
        return requested_qty, None, []
    limits, clusters = cluster_limits
    symbol = str(pick.get('symbol') or pick.get('underlying') or '').upper()
    cluster_keys = [name for name, members in clusters.items() if symbol in members]
    if not cluster_keys:
        return requested_qty, None, []
    qty_cap = requested_qty
    reasons: list[str] = []
    for name in cluster_keys:
        remaining = limits[name] - used_clusters.get(name, 0.0)
        cluster_cap = int(remaining // per_contract_risk)
        qty_cap = min(qty_cap, cluster_cap)
        if cluster_cap < requested_qty:
            reasons.append(
                f"{name} cluster risk budget ${max(remaining, 0.0):,.0f} remaining vs ${per_contract_risk:,.0f} per contract"
            )
    return qty_cap, '; '.join(reasons) if reasons else None, cluster_keys


def _position_overlay_exposure(position: dict) -> float:
    max_loss = position.get('max_loss_dollars')
    try:
        if max_loss is not None:
            return max(0.0, float(max_loss))
    except (TypeError, ValueError):
        pass

    strat = str(position.get('type') or position.get('strategy') or '').upper()
    if strat not in {'PCS', 'CCS', 'IC', 'IFLY'}:
        return capital_for_position(position)

    contracts = max(1, int(position.get('contracts') or position.get('quantity') or 1))
    premium = max(0.0, float(position.get('premium') or 0.0))
    width = 0.0
    legs = position.get('legs') or {}
    if isinstance(legs, str):
        try:
            legs = json.loads(legs) or {}
        except Exception:
            legs = {}
    if not isinstance(legs, dict):
        legs = {}

    if strat in {'PCS', 'CCS'}:
        ss = legs.get('short_strike') or legs.get('short_put') or legs.get('short_call') or position.get('strike') or 0.0
        ls = legs.get('long_strike') or legs.get('long_put') or legs.get('long_call') or 0.0
        width = abs(float(ss or 0.0) - float(ls or 0.0))
    else:
        sp = float(legs.get('short_put') or 0.0)
        lp = float(legs.get('long_put') or 0.0)
        sc = float(legs.get('short_call') or 0.0)
        lc = float(legs.get('long_call') or 0.0)
        put_wing = float(position.get('put_wing') or abs(sp - lp))
        call_wing = float(position.get('call_wing') or abs(lc - sc))
        width = max(put_wing, call_wing)

    if width <= 0:
        return capital_for_position(position)
    return max(0.0, round((width - premium) * 100 * contracts, 2))


def _consume_side_exposure(
    used_sides: dict[str, float],
    pick: dict,
    quantity: int,
    per_contract_risk: float,
) -> None:
    sides = _strategy_sides(pick.get('strategy') or '')
    if not sides or per_contract_risk <= 0:
        return
    per_side_risk = per_contract_risk * quantity / len(sides)
    for side in sides:
        used_sides[side] = used_sides.get(side, 0.0) + per_side_risk


def _consume_cluster_exposure(
    used_clusters: dict[str, float],
    cluster_keys: list[str],
    risk_dollars: float,
) -> None:
    if risk_dollars <= 0:
        return
    for key in cluster_keys:
        used_clusters[key] = used_clusters.get(key, 0.0) + risk_dollars
