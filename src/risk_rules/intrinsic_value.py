"""
risk_rules.intrinsic_value
==========================

``compute_cost_to_close(spot, strat, legs)`` — pure function.

Returns the intrinsic (expiry) cost-to-close for every supported strategy.
``PositionMonitor._compute_expiry_pnl`` delegates here so the payout maths
live in exactly one place.

``legs`` key names
------------------
PCS / CCS          : ``short_strike``, ``long_strike``
IC / IFLY          : ``short_put``, ``long_put``, ``short_call``, ``long_call``
CSP / CC           : ``short_strike``  (or fall back to ``pos['strike']`` at call site)
STRANGLE           : ``short_put``, ``short_call``

Position dictionaries use the same key names, so they can be passed directly
without any mapping.
"""
from __future__ import annotations

from typing import Optional


def compute_cost_to_close(
    spot: float,
    strat: str,
    legs: dict,
) -> Optional[float]:
    """
    Intrinsic cost-to-close per share at expiry.

    Parameters
    ----------
    spot  : underlying closing price at expiry
    strat : strategy code (``'CSP'``, ``'PCS'``, ``'CCS'``, ``'IC'``,
            ``'IFLY'``, ``'CC'``, ``'STRANGLE'``)
    legs  : dict with strike keys (see module docstring)

    Returns
    -------
    float
        Cost per share to close the position.  Subtract from entry premium to
        get realised P&L per share.
    None
        If required leg strikes are missing or zero (caller should treat the
        position as un-priceable and skip/return ``None``).
    """
    if strat == 'PCS':
        ss = float(legs.get('short_strike') or 0)
        ls = float(legs.get('long_strike') or 0)
        if ss <= 0:
            return None
        return max(0.0, ss - spot) - max(0.0, ls - spot)

    elif strat == 'CCS':
        ss = float(legs.get('short_strike') or 0)
        ls = float(legs.get('long_strike') or 0)
        if ss <= 0:
            return None
        return max(0.0, spot - ss) - max(0.0, spot - ls)

    elif strat in ('IC', 'IFLY'):
        sp = float(legs.get('short_put',  0) or 0)
        lp = float(legs.get('long_put',   0) or 0)
        # For IFLY, short_call == short_put (ATM center); allow that key to be
        # absent and fall back to short_put.
        sc = float(legs.get('short_call', 0) or 0) or sp
        lc = float(legs.get('long_call',  0) or 0)
        if sp <= 0 or sc <= 0:
            return None
        return (
            max(0.0, sp - spot) - max(0.0, lp - spot)
            + max(0.0, spot - sc) - max(0.0, spot - lc)
        )

    elif strat == 'CSP':
        ss = float(legs.get('short_strike') or 0)
        if ss <= 0:
            return None
        return max(0.0, ss - spot)

    elif strat == 'CC':
        ss = float(legs.get('short_strike') or 0)
        if ss <= 0:
            return None
        return max(0.0, spot - ss)

    elif strat == 'STRANGLE':
        sp = float(legs.get('short_put',  0) or 0)
        sc = float(legs.get('short_call', 0) or 0)
        if sp <= 0 or sc <= 0:
            return None
        return max(0.0, sp - spot) + max(0.0, spot - sc)

    return None
