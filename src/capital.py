"""Capital-at-risk helpers shared by agent, dashboard, and monitor."""
from __future__ import annotations

import json
from typing import Any


def _legs_dict(pos: dict[str, Any]) -> dict[str, Any]:
    legs = pos.get('legs') or {}
    if isinstance(legs, str):
        try:
            return json.loads(legs) or {}
        except Exception:
            return {}
    return legs if isinstance(legs, dict) else {}


def capital_for_position(pos: dict[str, Any]) -> float:
    """
    Estimate deployed buying power for a DB position row.

    The result is total dollars for the row, including ``contracts``. Dry-run
    rows return zero because they do not tie up broker capital.
    """
    if str(pos.get('status', '')).upper() == 'DRY_RUN':
        return 0.0

    max_loss = pos.get('max_loss_dollars')
    if max_loss is not None:
        try:
            return max(0.0, float(max_loss))
        except (TypeError, ValueError):
            pass

    strat = pos.get('type') or pos.get('strategy') or ''
    legs = _legs_dict(pos)
    strike = float(pos.get('strike') or 0)
    contracts = int(pos.get('contracts') or pos.get('quantity') or 1)

    per_contract = 0.0
    if strat == 'CSP':
        ss = legs.get('short_strike') or strike
        per_contract = float(ss or 0) * 100
    elif strat in ('PCS', 'CCS'):
        ss = legs.get('short_strike') or legs.get('short_put') or legs.get('short_call') or strike
        ls = legs.get('long_strike') or legs.get('long_put') or legs.get('long_call') or 0
        per_contract = abs(float(ss or 0) - float(ls or 0)) * 100
    elif strat in ('IC', 'IFLY'):
        sp = float(legs.get('short_put') or 0)
        lp = float(legs.get('long_put') or 0)
        sc = float(legs.get('short_call') or 0)
        lc = float(legs.get('long_call') or 0)
        per_contract = max(abs(sp - lp), abs(sc - lc)) * 100
    elif strat == 'CC':
        ss = legs.get('short_strike') or legs.get('short_call') or strike
        per_contract = float(ss or 0) * 100
    elif strat == 'STRANGLE':
        sp = legs.get('short_put') or strike
        per_contract = float(sp or 0) * 100
    elif pos.get('premium') is not None:
        per_contract = float(pos.get('premium') or 0) * 100

    return round(per_contract * contracts, 2)


def capital_by_strategy(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for pos in positions:
        strat = pos.get('type') or pos.get('strategy') or 'UNKNOWN'
        bucket = totals.setdefault(strat, {'strategy': strat, 'positions': 0, 'capital_deployed': 0.0})
        bucket['positions'] += 1
        bucket['capital_deployed'] += capital_for_position(pos)
    rows = list(totals.values())
    for row in rows:
        row['capital_deployed'] = round(row['capital_deployed'], 2)
    return sorted(rows, key=lambda r: r['capital_deployed'], reverse=True)
