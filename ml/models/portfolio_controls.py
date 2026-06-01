"""Portfolio-level risk control helpers shared across the ML training pipeline.

These functions mirror the live risk gates in agent.py / src/portfolio_risk.py
so that model training and evaluation always operate on the same tradeable
universe as live execution.

apply_portfolio_risk_controls() is the entry point. It processes a scored
DataFrame one entry date at a time, runs each date's top-ranked candidates
through PortfolioRiskService.filter_picks(), and marks rejected rows -inf
so they are excluded from selection metrics.

Why here and not in evaluate_risk_adjusted_ranking.py?
evaluate_exit_criteria.py and train_xgboost.py both sit earlier in the import
chain than evaluate_risk_adjusted_ranking.py. Putting this code in a leaf
module with no ml.models dependencies breaks the cycle while keeping a
single canonical implementation.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agent_risk import apply_directional_exposure_caps
from src.pick_selection import select_top_picks_with_scanner_controls
from src.portfolio_risk import PortfolioRiskService


def apply_portfolio_risk_controls(
    df: pd.DataFrame,
    score_column: str,
    *,
    account_capital: float = 50_000.0,
    max_candidates_per_entry_date: int | None = 50,
    scanner_controls: bool = True,
    scanner_config: dict[str, Any] | None = None,
    scanner_top_n_per_entry_date: int | None = None,
    regime_label_column: str | None = None,
    return_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, Any]]:
    """Return scores filtered by live scanner controls plus rejection diagnostics.

    Rows that survive pick selection, directional exposure caps, and portfolio
    gamma risk keep their original score; rejected rows are set to ``-inf`` so
    downstream selectors ignore them.

    Historical rows are evaluated one entry date at a time.  Expiry is
    normalised to today + row.dte because PortfolioRiskService computes DTE
    from calendar dates rather than storing it directly.

    Parameters
    ----------
    df:
        Scored candidate DataFrame.  Must contain ``strategy``, ``dte``,
        ``underlying_close``, ``short_strike``, ``long_strike``,
        ``implied_volatility``, ``entry_credit``, and either
        ``entry_timestamp`` or ``entry_date``.
    score_column:
        Column used to rank candidates within each entry date.
    account_capital:
        Simulated account size passed to PortfolioRiskService stress limits.
    max_candidates_per_entry_date:
        Upper bound on candidates forwarded to the risk service per date
        when scanner_controls is False.
    scanner_controls:
        Whether to apply pick-selection scanner rules before the gamma gate.
    scanner_config:
        Parsed config.json dict.  Pass None to use risk-service defaults.
    scanner_top_n_per_entry_date:
        Hard override for the per-date candidate cap (ignores config).
    regime_label_column:
        Optional column name carrying a per-row regime label (GREEN/YELLOW/ORANGE).
        When provided, the first non-null label within each entry-date group is
        forwarded to pick selection so regime-aware allocator caps/floors apply.
    return_diagnostics:
        When True, also return per-row gate stages/reasons plus aggregate counts
        describing which risk gate rejected each candidate.
    """
    out = pd.Series(-np.inf, index=df.index, dtype=float)
    diagnostics = {
        "gate_stage": pd.Series(pd.NA, index=df.index, dtype="string"),
        "gate_reason": pd.Series(pd.NA, index=df.index, dtype="string"),
        "gate_violation_codes": pd.Series(pd.NA, index=df.index, dtype="string"),
        "directional_reduced": pd.Series(False, index=df.index, dtype=bool),
        "portfolio_gamma_reduced": pd.Series(False, index=df.index, dtype=bool),
    }
    if df.empty:
        return _controls_result(out, diagnostics, return_diagnostics)

    config = scanner_config or {"risk_parameters": {"portfolio_gamma_risk": {}}}
    svc = PortfolioRiskService(config)
    working = df.copy()
    working["_portfolio_entry_date"] = _entry_dates(working)
    scores = pd.to_numeric(working[score_column], errors="coerce")

    for _, group in working[np.isfinite(scores)].groupby("_portfolio_entry_date", sort=True):
        ranked = group.assign(_portfolio_score=scores.loc[group.index]).sort_values(
            "_portfolio_score", ascending=False
        )
        candidate_limit = (
            scanner_top_n_per_entry_date
            or _scanner_top_n(scanner_config)
            or max_candidates_per_entry_date
        )
        diagnostics["gate_stage"].loc[ranked.index] = "candidate_build"

        built_picks: list[dict[str, Any]] = []
        built_index: set[Any] = set()
        for idx, row in ranked.iterrows():
            pick = _pick_from_scored_row(idx, row)
            if pick is None:
                diagnostics["gate_reason"].loc[idx] = "missing_required_trade_fields"
                continue
            built_picks.append(pick)
            built_index.add(idx)

        if scanner_controls:
            regime_label = _group_regime_label(ranked, regime_label_column)
            picks = select_top_picks_with_scanner_controls(
                built_picks,
                n=int(candidate_limit or len(built_picks)),
                config=scanner_config or {},
                regime_label=regime_label,
            )
        else:
            if max_candidates_per_entry_date is not None and max_candidates_per_entry_date > 0:
                ranked = ranked.head(max_candidates_per_entry_date)
                built_picks = [pick for pick in built_picks if pick["_row_index"] in set(ranked.index)]
                built_index = {pick["_row_index"] for pick in built_picks}
            picks = built_picks

        selected_index = {pick["_row_index"] for pick in picks}
        rejected_by_selection = built_index - selected_index
        if rejected_by_selection:
            diagnostics["gate_stage"].loc[list(rejected_by_selection)] = "pick_selection"
            diagnostics["gate_reason"].loc[list(rejected_by_selection)] = "scanner_controls"

        directional_requested_qty = {
            pick["_row_index"]: int(pick.get("quantity") or 1)
            for pick in picks
        }
        directional_picks = apply_directional_exposure_caps(
            picks,
            [],
            scanner_config or {},
            account_capital,
        )
        directional_index = {pick["_row_index"] for pick in directional_picks}
        rejected_by_directional = selected_index - directional_index
        if rejected_by_directional:
            diagnostics["gate_stage"].loc[list(rejected_by_directional)] = "directional_exposure"
            diagnostics["gate_reason"].loc[list(rejected_by_directional)] = "side_exposure_cap"
        for pick in directional_picks:
            row_index = pick["_row_index"]
            diagnostics["directional_reduced"].loc[row_index] = (
                int(pick.get("quantity") or 1) < directional_requested_qty.get(row_index, 1)
            )

        gamma_requested_qty = {
            pick["_row_index"]: int(pick.get("quantity") or 1)
            for pick in directional_picks
        }
        gamma_rejections: list[dict[str, Any]] = []
        accepted = svc.filter_picks(
            directional_picks,
            [],
            account_capital=account_capital,
            rejection_sink=gamma_rejections,
        )
        for rejected in gamma_rejections:
            row_index = rejected.get("_row_index")
            if row_index is None:
                continue
            diagnostics["gate_stage"].loc[row_index] = "portfolio_gamma"
            diagnostics["gate_reason"].loc[row_index] = str(rejected.get("reject_reason") or "portfolio_gamma")
            diagnostics["gate_violation_codes"].loc[row_index] = ",".join(
                str(code) for code in rejected.get("portfolio_violation_codes", []) if code
            ) or pd.NA
        for pick in accepted:
            row_index = pick["_row_index"]
            out.loc[row_index] = float(pick["_portfolio_score"])
            diagnostics["gate_stage"].loc[row_index] = "selected"
            diagnostics["portfolio_gamma_reduced"].loc[row_index] = (
                int(pick.get("quantity") or 1) < gamma_requested_qty.get(row_index, 1)
            )

    return _controls_result(out, diagnostics, return_diagnostics)


def load_scanner_config(path: str | None = "config.json") -> dict[str, Any]:
    """Load and return config.json, or an empty dict if unavailable."""
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entry_dates(df: pd.DataFrame) -> pd.Series:
    if "entry_timestamp" in df:
        return pd.to_datetime(df["entry_timestamp"], errors="coerce").dt.date.astype(str)
    if "entry_date" in df:
        return df["entry_date"].astype(str)
    return pd.Series(["all"] * len(df), index=df.index)


def _pick_from_scored_row(index: Any, row: pd.Series) -> dict[str, Any] | None:
    strategy = str(row.get("strategy") or "").upper()
    if strategy not in {"PCS", "CCS"}:
        return None
    dte = _positive_int(row.get("dte"))
    spot = _positive_float(row.get("underlying_close"))
    short_strike = _positive_float(row.get("short_strike") or row.get("strike"))
    long_strike = _positive_float(row.get("long_strike"))
    if dte is None or spot is None or short_strike is None or long_strike is None:
        return None
    short_iv = _positive_float(row.get("implied_volatility")) or 0.25
    iv_skew = _float_value(row.get("iv_skew_wing")) or 0.0
    long_iv = max(0.01, short_iv + iv_skew)
    score = _float_value(row.get("_portfolio_score"))
    if score is None:
        return None
    pick: dict[str, Any] = {
        "_row_index": index,
        "_portfolio_score": score,
        "strategy": strategy,
        "symbol": str(row.get("underlying") or row.get("symbol") or "?"),
        "expiry": (date.today() + timedelta(days=dte)).isoformat(),
        "current_price": spot,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "premium": _positive_float(row.get("entry_credit")) or _positive_float(row.get("option_entry_price")) or 0.01,
        "quantity": 1,
        "score": score,
        "short_iv": short_iv,
        "long_iv": long_iv,
    }
    if strategy == "PCS":
        pick["short_put"] = short_strike
        pick["long_put"] = long_strike
    else:
        pick["short_call"] = short_strike
        pick["long_call"] = long_strike
    return pick


def _group_regime_label(group: pd.DataFrame, column: str | None) -> str | None:
    if not column or column not in group.columns:
        return None
    series = group[column].dropna()
    if series.empty:
        return None
    label = str(series.iloc[0]).strip().upper()
    return label or None


def _scanner_top_n(config: dict[str, Any] | None) -> int | None:
    cfg = config or {}
    value = (
        cfg.get("ml_scanner", {}).get("top_n")
        if isinstance(cfg.get("ml_scanner"), dict)
        else None
    )
    if value is None:
        value = cfg.get("top_n_picks", cfg.get("top_n_per_strategy"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _controls_result(
    scores: pd.Series,
    diagnostics: dict[str, pd.Series],
    return_diagnostics: bool,
) -> pd.Series | tuple[pd.Series, dict[str, Any]]:
    if not return_diagnostics:
        return scores
    gate_stage = diagnostics["gate_stage"].dropna()
    gate_codes = diagnostics["gate_violation_codes"].dropna()
    payload: dict[str, Any] = dict(diagnostics)
    payload["gate_stage_counts"] = dict(Counter(str(value) for value in gate_stage.tolist()))
    payload["portfolio_gamma_violation_counts"] = dict(
        Counter(
            code
            for codes in gate_codes.tolist()
            for code in str(codes).split(",")
            if code
        )
    )
    payload["quantity_reduction_counts"] = {
        "directional_exposure": int(diagnostics["directional_reduced"].sum()),
        "portfolio_gamma": int(diagnostics["portfolio_gamma_reduced"].sum()),
        "any": int((diagnostics["directional_reduced"] | diagnostics["portfolio_gamma_reduced"]).sum()),
    }
    return scores, payload


def _float_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _positive_float(value: Any) -> float | None:
    numeric = _float_value(value)
    return numeric if numeric is not None and numeric > 0 else None


def _positive_int(value: Any) -> int | None:
    numeric = _positive_float(value)
    return max(1, int(round(numeric))) if numeric is not None else None
