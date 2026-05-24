"""
market_data.base
================

Shared data structures for the HFT / non-HFT data adapter layer.
"""
from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class PositionChainResult:
    """
    Unified return value from :meth:`DataAdapter.get_position_chain`.

    Both the HFT path (Alpaca snapshots + broker greeks) and the non-HFT
    path (yfinance chain + IV-based greeks) return this same structure.

    Attributes
    ----------
    spot :
        Current underlying price; ``None`` if unavailable.
    put_map :
        ``{strike_float: row_dict}`` for put legs; ``None`` on fetch failure.
    call_map :
        ``{strike_float: row_dict}`` for call legs; ``None`` on fetch failure.
    has_broker_greeks :
        ``True`` when this result came from Alpaca snapshots (HFT mode) so
        callers can use the broker-supplied greeks directly rather than
        computing Black-Scholes.
    snapshots :
        ``{osi_str: snapshot_row_dict}`` — only set when ``has_broker_greeks``
        is ``True``.
    osi_map :
        ``{(strike, opt_type): osi_str}`` — only set when ``has_broker_greeks``
        is ``True``.
    leg_specs :
        ``[(strike, opt_type, position_side), …]`` — only set when
        ``has_broker_greeks`` is ``True``.
    """

    spot: Optional[float]
    put_map: Optional[dict]
    call_map: Optional[dict]
    has_broker_greeks: bool
    snapshots: Optional[dict] = None
    osi_map: Optional[dict] = None
    leg_specs: Optional[list] = None
