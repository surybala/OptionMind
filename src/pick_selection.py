"""Shared pick-selection controls for scanner and model evaluation.

Two selection modes are supported via ``config["pick_selection"]["mode"]``:

equal_diversity (default)
    Each active strategy receives a guaranteed floor of ``floor(n / k)`` slots
    where *k* is the number of distinct strategies present.  Remainder slots are
    filled greedily by score.  IC is further capped by
    ``strategies.iron_condor.ic_allocation_pct``.

model_ranked
    Pure top-N by model score with no per-strategy floors or regime-driven
    side overrides. Only hard caps apply: IC allocation cap, per-ticker cap,
    and optional per-strategy caps from
    ``config["pick_selection"]["strategy_caps"]``. Designed for the ML
    scanner where the ranker and loss models already encode directional trade
    quality.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def select_top_picks_with_scanner_controls(
    candidates: list[dict[str, Any]],
    *,
    n: int,
    config: dict[str, Any] | None = None,
    regime_label: str | None = None,
) -> list[dict[str, Any]]:
    """Apply scanner diversity and per-symbol controls to ranked candidates."""
    if not candidates or n <= 0:
        return []

    cfg = config or {}
    selection_cfg = (
        cfg.get("pick_selection", {})
        if isinstance(cfg.get("pick_selection"), dict)
        else {}
    )
    mode = str(selection_cfg.get("mode", "equal_diversity"))

    ranked = sorted(
        candidates,
        key=lambda x: x.get("score", x.get("model_score", 0.0)),
        reverse=True,
    )
    strategies_cfg = (
        cfg.get("strategies", {}) if isinstance(cfg.get("strategies"), dict) else {}
    )
    ic_cfg = (
        strategies_cfg.get("iron_condor", {})
        if isinstance(strategies_cfg.get("iron_condor"), dict)
        else {}
    )
    ic_pct = float(ic_cfg.get("ic_allocation_pct", 1.0))
    max_ic_slots = max(1, int(ic_pct * n))
    max_per_ticker = cfg.get("max_picks_per_ticker")
    if mode == "model_ranked":
        raw_caps = (
            selection_cfg.get("strategy_caps", {})
            if isinstance(selection_cfg.get("strategy_caps"), dict)
            else {}
        )
        strategy_caps: dict[str, int] = {
            strat: max(1, int(float(frac) * n))
            for strat, frac in raw_caps.items()
        }
        return _model_ranked_selection(
            ranked,
            n,
            max_ic_slots=max_ic_slots,
            ic_pct=ic_pct,
            max_per_ticker=max_per_ticker,
            strategy_caps=strategy_caps,
        )

    return _equal_diversity_selection(
        ranked,
        n,
        max_ic_slots=max_ic_slots,
        ic_pct=ic_pct,
        max_per_ticker=max_per_ticker,
    )


def _model_ranked_selection(
    ranked: list[dict[str, Any]],
    n: int,
    *,
    max_ic_slots: int,
    ic_pct: float,
    max_per_ticker: int | None,
    strategy_caps: dict[str, int],
) -> list[dict[str, Any]]:
    """Greedy top-N by score with hard caps only."""
    selected: list[dict[str, Any]] = []
    strategy_ct: dict[str, int] = defaultdict(int)
    ticker_ct: dict[str, int] = defaultdict(int)

    for pick in ranked:
        if len(selected) >= n:
            break
        if not _model_ranked_can_select(
            pick,
            strategy_ct=strategy_ct,
            ticker_ct=ticker_ct,
            max_per_ticker=max_per_ticker,
            max_ic_slots=max_ic_slots,
            ic_pct=ic_pct,
            strategy_caps=strategy_caps,
        ):
            continue
        selected.append(pick)
        strat = str(pick.get("strategy") or "")
        sym = str(pick.get("symbol") or "")
        strategy_ct[strat] += 1
        ticker_ct[sym] += 1

    selected.sort(
        key=lambda x: x.get("score", x.get("model_score", 0.0)), reverse=True
    )
    return selected


def _model_ranked_can_select(
    pick: dict[str, Any],
    *,
    strategy_ct: dict[str, int],
    ticker_ct: dict[str, int],
    max_per_ticker: int | None,
    max_ic_slots: int,
    ic_pct: float,
    strategy_caps: dict[str, int],
) -> bool:
    strat = str(pick.get("strategy") or "")
    sym = str(pick.get("symbol") or "")
    if max_per_ticker is not None and ticker_ct[sym] >= int(max_per_ticker):
        return False
    if strat == "IC" and ic_pct < 1.0 and strategy_ct["IC"] >= max_ic_slots:
        return False
    cap = strategy_caps.get(strat)
    if cap is not None and strategy_ct[strat] >= cap:
        return False
    return True


def _equal_diversity_selection(
    ranked: list[dict[str, Any]],
    n: int,
    *,
    max_ic_slots: int,
    ic_pct: float,
    max_per_ticker: int | None,
) -> list[dict[str, Any]]:
    """Equal-diversity selection: guaranteed per-strategy floor + IC cap."""
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_ticker_ct: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pick in ranked:
        strat = str(pick.get("strategy") or "")
        sym = str(pick.get("symbol") or "")
        pool_quota = max_ic_slots if strat == "IC" and ic_pct < 1.0 else n
        if len(pools[strat]) >= pool_quota:
            continue
        if max_per_ticker is not None and pool_ticker_ct[strat][sym] >= int(max_per_ticker):
            continue
        pools[strat].append(pick)
        pool_ticker_ct[strat][sym] += 1

    if not pools:
        return []

    per_strat_q = max(1, n // len(pools))
    selected: list[dict[str, Any]] = []
    pool_ptr: dict[str, int] = {}
    for strat, group in sorted(pools.items()):
        q = min(per_strat_q, max_ic_slots) if strat == "IC" else per_strat_q
        selected.extend(group[:q])
        pool_ptr[strat] = q

    remaining = n - len(selected)
    if remaining > 0:
        extras: list[dict[str, Any]] = []
        for strat, group in sorted(pools.items()):
            extras.extend(group[pool_ptr[strat]:])
        extras.sort(
            key=lambda x: x.get("score", x.get("model_score", 0.0)), reverse=True
        )

        ic_in_selected = sum(1 for p in selected if p.get("strategy") == "IC")
        global_ticker_ct: dict[str, int] = defaultdict(int)
        for pick in selected:
            global_ticker_ct[str(pick.get("symbol") or "")] += 1

        for pick in extras:
            if remaining <= 0:
                break
            strat = str(pick.get("strategy") or "")
            sym = str(pick.get("symbol") or "")
            if max_per_ticker is not None and global_ticker_ct[sym] >= int(max_per_ticker):
                continue
            if strat == "IC" and ic_pct < 1.0 and ic_in_selected >= max_ic_slots:
                continue
            selected.append(pick)
            global_ticker_ct[sym] += 1
            if strat == "IC":
                ic_in_selected += 1
            remaining -= 1

    selected.sort(
        key=lambda x: x.get("score", x.get("model_score", 0.0)), reverse=True
    )
    return selected[:n]
