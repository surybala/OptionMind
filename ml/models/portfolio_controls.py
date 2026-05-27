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
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
) -> pd.Series:
    """Return a score Series filtered by the live portfolio risk service.

    Rows that pass PortfolioRiskService.filter_picks() keep their original
    score; rejected rows are set to -inf so downstream selectors ignore them.

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
    """
    out = pd.Series(-np.inf, index=df.index, dtype=float)
    if df.empty:
        return out

    svc = PortfolioRiskService({"risk_parameters": {"portfolio_gamma_risk": {}}})
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
        if scanner_controls:
            picks = [_pick_from_scored_row(idx, row) for idx, row in ranked.iterrows()]
            picks = [p for p in picks if p is not None]
            picks = select_top_picks_with_scanner_controls(
                picks,
                n=int(candidate_limit or len(picks)),
                config=scanner_config or {},
            )
        else:
            if max_candidates_per_entry_date is not None and max_candidates_per_entry_date > 0:
                ranked = ranked.head(max_candidates_per_entry_date)
            picks = [_pick_from_scored_row(idx, row) for idx, row in ranked.iterrows()]
            picks = [p for p in picks if p is not None]

        accepted = svc.filter_picks(picks, [], account_capital=account_capital)
        for pick in accepted:
            out.loc[pick["_row_index"]] = float(pick["_portfolio_score"])

    return out


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
