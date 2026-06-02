"""
agent_audit.py
==============

Candidate audit trail: scoring annotations, rejection tracking, scan-audit
persistence, and model prediction/decision ledger writes.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

from src.database import TradeDatabase
from src.agent_display import legs_str
from src.utils import get_logger

log = get_logger()

SCAN_AUDIT_PATH = os.path.join('data', 'model_candidates.json')


# ── Pick identity & scoring ─────────────────────────────────────────────────

def pick_key(pick: dict) -> tuple:
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


def mispricing_score(pick: dict) -> float:
    """
    Practical model score adapter for existing risk/audit displays.

    New candidates should provide ``model_score``. Legacy ``score`` remains
    accepted only so old fixtures and execution code can share the same shape.
    """
    try:
        return round(float(pick.get('mispricing_score', pick.get('score', 0.0)) or 0.0), 4)
    except (TypeError, ValueError):
        return 0.0


def annotate_mispricing_scores(picks: list[dict]) -> list[dict]:
    for pick in picks:
        pick['mispricing_score'] = mispricing_score(pick)
        pick.setdefault(
            'mispricing_score_basis',
            'ML model score: expected utility / P&L-centered inference',
        )
    return picks


# ── Rejection tracking ───────────────────────────────────────────────────────

def capture_rejections(
    before: list[dict],
    after: list[dict],
    gate: str,
    reason,
    rejected: list[dict],
) -> None:
    kept = {pick_key(p) for p in after}
    seen = {pick_key(p) for p in rejected}
    for pick in before:
        key = pick_key(pick)
        if key in kept or key in seen:
            continue
        item = dict(pick)
        item['filtered_stage'] = gate
        item['reject_reason'] = reason(pick) if callable(reason) else str(reason)
        item['mispricing_score'] = mispricing_score(item)
        rejected.append(item)


# ── Scan audit persistence ───────────────────────────────────────────────────

def _pick_audit_row(pick: dict, status: str) -> dict:
    quantity = int(pick.get('quantity') or 1)
    premium = float(pick.get('premium') or 0.0)
    return {
        'status': status,
        'symbol': pick.get('symbol'),
        'strategy': pick.get('strategy'),
        'expiry': pick.get('expiry'),
        'legs': legs_str(pick),
        'quantity': quantity,
        'premium': round(premium, 4),
        'total_credit': round(premium * 100 * quantity, 2),
        'prob_win': pick.get('prob_win'),
        'roi': pick.get('roi'),
        'score': pick.get('score'),
        'mispricing_score': mispricing_score(pick),
        'mispricing_score_basis': pick.get('mispricing_score_basis'),
        'ranking_reason': pick.get('ranking_reason'),
        'ranking_context': pick.get('ranking_context'),
        'large_loss_prob': pick.get('large_loss_prob'),
        'stop_loss_prob': pick.get('stop_loss_prob'),
        'current_price': pick.get('current_price'),
        'max_loss_multiple': pick.get('max_loss_multiple'),
        'max_loss_per_contract': pick.get('max_loss_per_contract'),
        'filtered_stage': pick.get('filtered_stage'),
        'reject_reason': pick.get('reject_reason'),
    }


def write_scan_audit(
    selected: list[dict],
    rejected: list[dict],
    *,
    db: TradeDatabase | None = None,
    path: str = SCAN_AUDIT_PATH,
    max_rejected: int = 25,
) -> None:
    """Persist latest candidate plan/rejections for the dashboard."""
    if db is not None:
        record_model_decisions(db, selected, rejected)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    selected_rows = [_pick_audit_row(p, 'SELECTED') for p in selected]
    selected_floor = min(
        (row['mispricing_score'] for row in selected_rows),
        default=0.0,
    )
    interesting_rejected = [
        p for p in rejected
        if mispricing_score(p) >= selected_floor
    ]
    if not interesting_rejected:
        interesting_rejected = list(rejected)
    interesting_rejected = sorted(
        interesting_rejected,
        key=lambda p: mispricing_score(p),
        reverse=True,
    )[:max_rejected]
    rejected_rows = [_pick_audit_row(p, 'REJECTED') for p in interesting_rejected]
    payload = {
        'generated_at': datetime.now().isoformat(),
        'scanner': 'ml',
        'score_basis': (
            'Higher means the ML inference layer ranked the candidate as more '
            'attractive; deterministic risk gates still apply.'
        ),
        'selected': selected_rows,
        'rejected': rejected_rows,
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=str)


# ── Model prediction / decision ledger ───────────────────────────────────────

def record_model_predictions(db: TradeDatabase, picks: list[dict]) -> None:
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


def record_model_decisions(
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
