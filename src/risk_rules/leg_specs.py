"""
risk_rules.leg_specs
====================

Pure functions for parsing leg data from a position dict.

``parse_legs``             — deserialise ``pos['legs']`` to a plain dict.
``get_position_leg_specs`` — return ``[(strike, opt_type, side), …]`` for
                             every leg in the position, used to build OSI
                             symbols and look up snapshot / chain data.
"""
from __future__ import annotations

import json
from typing import Optional


def parse_legs(pos: dict) -> dict:
    """
    Return the legs sub-dict for *pos*, deserialising from JSON if needed.

    Returns an empty dict if ``pos`` has no ``'legs'`` key or it is falsy.
    """
    raw = pos.get('legs')
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def get_position_leg_specs(pos: dict) -> list[tuple[float, str, str]]:
    """
    Return ``[(strike, opt_type, position_side), …]`` for every leg in *pos*.

    ``position_side`` is ``'short'`` or ``'long'``.

    Used to build OSI symbols for Alpaca snapshot fetches and to look up
    per-leg data in the put/call chain maps.
    """
    strat = pos['type']
    legs  = parse_legs(pos)
    specs: list[tuple[float, str, str]] = []

    def _add(strike, opt_type: str, side: str) -> None:
        if strike is not None:
            specs.append((float(strike), opt_type, side))

    if strat == 'PCS':
        _add(legs.get('short_strike') or legs.get('short_put') or pos.get('strike'), 'put', 'short')
        _add(legs.get('long_strike')  or legs.get('long_put'),                        'put', 'long')
    elif strat == 'CCS':
        _add(legs.get('short_strike') or legs.get('short_call') or pos.get('strike'), 'call', 'short')
        _add(legs.get('long_strike')  or legs.get('long_call'),                        'call', 'long')
    elif strat in ('IC', 'IFLY'):
        _add(legs.get('short_put'),  'put',  'short')
        _add(legs.get('long_put'),   'put',  'long')
        _add(legs.get('short_call'), 'call', 'short')
        _add(legs.get('long_call'),  'call', 'long')
    elif strat == 'CSP':
        _add(legs.get('short_strike') or pos.get('strike'), 'put', 'short')
    elif strat == 'CC':
        _add(legs.get('short_strike') or pos.get('strike'), 'call', 'short')
    elif strat == 'STRANGLE':
        _add(legs.get('short_put')  or pos.get('strike'), 'put',  'short')
        _add(legs.get('short_call'),                       'call', 'short')

    return specs
