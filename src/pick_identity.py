"""Helpers for identifying duplicate or laddered option positions."""
from __future__ import annotations

from typing import Any

from src.risk_rules.leg_specs import parse_legs


def symbol_strategy_key(item: dict[str, Any]) -> tuple[str, str]:
    """Return a normalized `(symbol, strategy)` key for a pick or position."""
    symbol = str(item.get("symbol") or item.get("underlying") or "").upper()
    strategy = str(item.get("strategy") or item.get("type") or "").upper()
    return symbol, strategy


def pick_contract_signature(pick: dict[str, Any]) -> tuple[str, str, str, tuple[tuple[str, float], ...]]:
    """Return a normalized signature for a candidate pick."""
    strategy = str(pick.get("strategy") or "").upper()
    return (
        str(pick.get("symbol") or "").upper(),
        strategy,
        str(pick.get("expiry") or ""),
        _legs_signature(strategy, pick, fallback_strike=pick.get("strike")),
    )


def position_contract_signature(position: dict[str, Any]) -> tuple[str, str, str, tuple[tuple[str, float], ...]]:
    """Return a normalized signature for an open or pending-close position."""
    strategy = str(position.get("type") or position.get("strategy") or "").upper()
    legs = parse_legs(position)
    leg_data = dict(position)
    leg_data.update(legs)
    return (
        str(position.get("symbol") or "").upper(),
        strategy,
        str(position.get("expiry") or ""),
        _legs_signature(strategy, leg_data, fallback_strike=position.get("strike")),
    )

def _legs_signature(
    strategy: str,
    data: dict[str, Any],
    *,
    fallback_strike: Any = None,
) -> tuple[tuple[str, float], ...]:
    signature: list[tuple[str, float]] = []

    def _append(label: str, *values: Any) -> None:
        for value in values:
            normalized = _strike_value(value)
            if normalized is not None:
                signature.append((label, normalized))
                return

    if strategy in ("PCS", "CCS"):
        _append("short", data.get("short_strike"), data.get("short_put"), data.get("short_call"), fallback_strike)
        _append("long", data.get("long_strike"), data.get("long_put"), data.get("long_call"))
    elif strategy in ("IC", "IFLY"):
        _append("short_put", data.get("short_put"))
        _append("long_put", data.get("long_put"))
        _append("short_call", data.get("short_call"))
        _append("long_call", data.get("long_call"))
    elif strategy == "CSP":
        _append("short", data.get("short_strike"), data.get("short_put"), fallback_strike)
    elif strategy == "CC":
        _append("short", data.get("short_strike"), data.get("short_call"), fallback_strike)
    elif strategy == "STRANGLE":
        _append("short_put", data.get("short_put"), fallback_strike)
        _append("short_call", data.get("short_call"))

    return tuple(signature)


def _strike_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None
