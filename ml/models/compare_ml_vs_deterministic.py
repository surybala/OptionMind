"""Compare the current ML backtest against the deterministic optionwheel agent."""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.backtest_agent_like import (
    _allocator_regime_labels,
    _gate_stage_counts,
    _portfolio_gamma_violation_counts,
    _quantity_reduction_counts,
    _resolve_optional_artifact_path,
    _resolve_ranker_artifact,
    _resolve_strategy_ranker_paths,
    _score_rankers,
    _selection_stats,
)
from ml.models.evaluate_risk_adjusted_ranking import (
    _score_classifier,
    apply_large_loss_gate,
)
from ml.models.portfolio_controls import apply_portfolio_risk_controls
from src.agent_risk import apply_ml_position_sizing, apply_ml_quantity_overlays
from src.pick_selection import select_top_picks_with_scanner_controls
from src.utils import load_config

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class RegimeProfile:
    label: str
    quantity_multiplier: float
    top_n_multiplier: float
    pause_new_trades: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare OptionMind ML vs deterministic optionwheel-style backtests."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Candidate dataset directory, parquet, or JSONL.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2022,
        help="First calendar year to include.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last calendar year to include.",
    )
    parser.add_argument(
        "--ml-config",
        default="config.json",
        help="OptionMind runtime config path.",
    )
    parser.add_argument(
        "--ml-registry",
        default="artifacts/model_registry.json",
        help="OptionMind model registry path.",
    )
    parser.add_argument(
        "--det-config",
        default="../optionwheel/config.json",
        help="Deterministic optionwheel config path.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional JSON report output path.",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Optional Markdown summary output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_comparison(
        dataset_path=Path(args.input),
        start_year=args.start_year,
        end_year=args.end_year,
        ml_config_path=Path(args.ml_config),
        ml_registry_path=Path(args.ml_registry),
        deterministic_config_path=Path(args.det_config),
    )
    payload = json.loads(json.dumps(report, default=str))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_comparison(
    *,
    dataset_path: Path,
    start_year: int,
    end_year: int,
    ml_config_path: Path,
    ml_registry_path: Path,
    deterministic_config_path: Path,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required for ML comparison backtesting.")
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    df = load_dataset(dataset_path)
    base = _filter_year_range(df, start_year, end_year)
    print(
        f"[compare] loaded {len(base):,} rows for {start_year}-{end_year} from {dataset_path}",
        file=sys.stderr,
        flush=True,
    )
    ml_report = _run_ml_side(
        base_df=base,
        dataset_path=dataset_path,
        start_year=start_year,
        end_year=end_year,
        config_path=ml_config_path,
        registry_path=ml_registry_path,
    )
    deterministic_report = _run_deterministic_side(
        base_df=base,
        dataset_path=dataset_path,
        start_year=start_year,
        end_year=end_year,
        config_path=deterministic_config_path,
    )
    quarter_labels = _quarter_labels(start_year, end_year)
    comparison = _build_comparison(
        ml_report["selected_frame"],
        deterministic_report["selected_frame"],
        quarter_labels,
    )
    return {
        "dataset_path": str(dataset_path),
        "window": {
            "start_year": start_year,
            "end_year": end_year,
            "quarter_labels": quarter_labels,
        },
        "assumptions": {
            "shared_backtest_semantics": [
                "Per-entry-date replay only; no rolling open-book carry across dates.",
                "Expected PnL is evaluated per selected candidate row; quantities affect gate diagnostics but not PnL aggregation.",
                "Deterministic replay uses dataset-backed proxies where the historical dataset lacks live scanner fields.",
            ],
            "deterministic_proxies": {
                "prob_win_proxy": "1 - abs(option_delta)",
                "liquidity_filter": "Historical dataset has no open-interest field, so deterministic replay skips the live min_open_interest gate.",
                "regime_proxy": "Dataset market_trend_regime / market_volatility_regime / vix_regime map into GREEN/YELLOW/ORANGE labels.",
            },
        },
        "ml": _strip_frame(ml_report),
        "deterministic": _strip_frame(deterministic_report),
        "comparison": comparison,
    }


def _run_ml_side(
    *,
    base_df: pd.DataFrame,
    dataset_path: Path,
    start_year: int,
    end_year: int,
    config_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    repo_root = config_path.parent.resolve()
    config = load_config(str(config_path))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ranker_artifact_path = _resolve_ranker_artifact(repo_root, registry)
    strategy_rankers = _resolve_strategy_ranker_paths(
        repo_root,
        config,
        default_ranker=ranker_artifact_path,
        pcs_ranker_artifact_path=None,
        ccs_ranker_artifact_path=None,
    )
    large_loss_artifact_path = _resolve_optional_artifact_path(
        repo_root,
        None,
        str(config.get("ml_scanner", {}).get("large_loss_classifier_path") or ""),
    )
    stop_loss_artifact_path = _resolve_optional_artifact_path(
        repo_root,
        None,
        str(config.get("ml_scanner", {}).get("stop_loss_classifier_path") or ""),
    )
    if large_loss_artifact_path is None:
        raise ValueError("Unable to resolve ML large-loss classifier artifact path.")

    working = base_df.copy()
    min_dte = int(config.get("ml_scanner", {}).get("min_dte", 7) or 7)
    max_dte = int(
        config.get("ml_scanner", {}).get(
            "max_dte",
            config.get("expiry_days_max", 45) or 45,
        )
        or 45
    )
    if "dte" in working.columns:
        dte = pd.to_numeric(working["dte"], errors="coerce")
        working = working.loc[dte.between(min_dte, max_dte, inclusive="both")].copy()
    print(
        f"[compare] ML side: scoring {len(working):,} rows with DTE {min_dte}-{max_dte}",
        file=sys.stderr,
        flush=True,
    )

    scored = _score_rankers(working, strategy_rankers)
    scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact_path)
    if stop_loss_artifact_path is not None and stop_loss_artifact_path.exists():
        scored["stop_loss_probability"] = _score_classifier(scored, stop_loss_artifact_path)
    else:
        scored["stop_loss_probability"] = pd.Series(pd.NA, index=scored.index, dtype="Float64")

    scored["gated_score"] = pd.to_numeric(scored["prediction"], errors="coerce").fillna(-np.inf)
    scored["gate_stage"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["gate_reason"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["gate_violation_codes"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["directional_reduced"] = False
    scored["portfolio_gamma_reduced"] = False

    large_loss_threshold = float(
        config.get("ml_scanner", {}).get("large_loss_veto_threshold", 0.60)
    )
    stop_loss_threshold = float(
        config.get("ml_scanner", {}).get("stop_loss_veto_threshold", 0.30)
    )
    scored["gated_score"] = apply_large_loss_gate(
        scored["gated_score"],
        scored["large_loss_probability"],
        max_large_loss_probability=large_loss_threshold,
    )
    large_loss_veto = ~np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce"))
    scored.loc[large_loss_veto, "gate_stage"] = "large_loss_veto"
    scored.loc[large_loss_veto, "gate_reason"] = f"p_large_loss>{large_loss_threshold:.2f}"

    scored["gated_score"] = apply_large_loss_gate(
        scored["gated_score"],
        scored["stop_loss_probability"],
        max_large_loss_probability=stop_loss_threshold,
    )
    stop_loss_veto = (
        ~np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce"))
        & ~large_loss_veto
    )
    scored.loc[stop_loss_veto, "gate_stage"] = "stop_loss_veto"
    scored.loc[stop_loss_veto, "gate_reason"] = f"p_stop_loss>{stop_loss_threshold:.2f}"

    scored["allocator_regime_label"] = _allocator_regime_labels(scored)
    portfolio_score, diagnostics = apply_portfolio_risk_controls(
        scored,
        "gated_score",
        account_capital=float(config.get("max_capital_per_period", 50_000.0)),
        scanner_controls=True,
        scanner_config=config,
        regime_label_column="allocator_regime_label",
        return_diagnostics=True,
    )
    scored["portfolio_score"] = portfolio_score
    for column in diagnostics:
        scored[column] = diagnostics[column]

    scored.loc[large_loss_veto, "gate_stage"] = "large_loss_veto"
    scored.loc[large_loss_veto, "gate_reason"] = f"p_large_loss>{large_loss_threshold:.2f}"
    scored.loc[stop_loss_veto, "gate_stage"] = "stop_loss_veto"
    scored.loc[stop_loss_veto, "gate_reason"] = f"p_stop_loss>{stop_loss_threshold:.2f}"
    scored["quarter"] = pd.PeriodIndex(pd.to_datetime(scored["entry_timestamp"]), freq="Q").astype(str)
    selected = scored[np.isfinite(pd.to_numeric(scored["portfolio_score"], errors="coerce"))].copy()
    selected = selected.sort_values(["entry_timestamp", "portfolio_score"], ascending=[True, False])
    print(
        f"[compare] ML side: selected {len(selected):,} rows",
        file=sys.stderr,
        flush=True,
    )

    return {
        "name": "ml",
        "config_path": str(config_path),
        "registry_path": str(registry_path),
        "runtime": {
            "min_dte": min_dte,
            "max_dte": max_dte,
            "scanner_top_n": int(config.get("ml_scanner", {}).get("top_n", 10) or 10),
            "pick_selection_mode": str((config.get("pick_selection") or {}).get("mode") or ""),
            "large_loss_veto_threshold": large_loss_threshold,
            "stop_loss_veto_threshold": stop_loss_threshold,
            "max_capital_per_period": float(config.get("max_capital_per_period", 50_000.0)),
        },
        "reports": _build_side_reports(
            scored=scored,
            candidate_mask=np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce")),
            selected=selected,
            gate_label="post_primary_gate",
            start_year=start_year,
            end_year=end_year,
        ),
        "selected_frame": selected,
        "gate_diagnostics": {
            "gate_stage_counts": _gate_stage_counts(scored),
            "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(scored),
            "quantity_reduction_counts": _quantity_reduction_counts(scored),
        },
    }


def _run_deterministic_side(
    *,
    base_df: pd.DataFrame,
    dataset_path: Path,
    start_year: int,
    end_year: int,
    config_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    working = base_df.copy()
    max_dte = int(config.get("expiry_days_max", 14) or 14)
    if "dte" in working.columns:
        dte = pd.to_numeric(working["dte"], errors="coerce")
        working = working.loc[dte.between(1, max_dte, inclusive="both")].copy()

    working["entry_timestamp"] = pd.to_datetime(working["entry_timestamp"], errors="coerce", utc=True)
    working = working.dropna(subset=["entry_timestamp"]).copy()
    print(
        f"[compare] Deterministic side: replaying {len(working):,} rows with max DTE {max_dte}",
        file=sys.stderr,
        flush=True,
    )
    working["_entry_date"] = working["entry_timestamp"].dt.date
    working["quarter"] = pd.PeriodIndex(working["entry_timestamp"], freq="Q").astype(str)
    working["allocator_regime_label"] = _allocator_regime_labels(working)
    working["prob_win_proxy"] = 1.0 - pd.to_numeric(working.get("option_delta"), errors="coerce").abs()
    working["score"] = _deterministic_scores(working)
    working["primary_pass"] = False
    working["candidate_pass"] = False
    working["portfolio_score"] = -np.inf
    working["gate_stage"] = pd.Series(pd.NA, index=working.index, dtype="string")
    working["gate_reason"] = pd.Series(pd.NA, index=working.index, dtype="string")
    working["gate_violation_codes"] = pd.Series(pd.NA, index=working.index, dtype="string")
    working["directional_reduced"] = False
    working["portfolio_gamma_reduced"] = False

    top_n = _deterministic_top_n(config)
    capital_budget = float(config.get("max_capital_per_period", 50_000.0))
    account_capital = float(config.get("account_capital") or capital_budget)
    entry_dates = working["_entry_date"].dropna().sort_values().unique().tolist()
    total_dates = len(entry_dates)
    for index, (_, group) in enumerate(working.groupby("_entry_date", sort=True), start=1):
        if index == 1 or index % 25 == 0 or index == total_dates:
            print(
                f"[compare] Deterministic side: processed {index}/{total_dates} entry dates",
                file=sys.stderr,
                flush=True,
            )
        primary = group.loc[_primary_mask(group, config)].copy()
        rejected_primary = group.index.difference(primary.index)
        if len(rejected_primary):
            working.loc[rejected_primary, "gate_stage"] = "deterministic_primary_filter"
            working.loc[rejected_primary, "gate_reason"] = "credit_delta_probability_filter"
        if primary.empty:
            continue

        working.loc[primary.index, "primary_pass"] = True
        regime_label = _group_regime_label(primary)
        regime = _regime_profile(config, regime_label)

        picks = [_pick_from_row(idx, row) for idx, row in primary.iterrows()]
        picks = [pick for pick in picks if pick is not None]
        if not picks:
            continue

        selected = select_top_picks_with_scanner_controls(
            picks,
            n=top_n,
            config=config,
            regime_label=regime.label,
        )
        selected_idx = {pick["_row_index"] for pick in selected}
        rejected = primary.index.difference(list(selected_idx))
        if len(rejected):
            working.loc[rejected, "gate_stage"] = "pick_selection"
            working.loc[rejected, "gate_reason"] = "selection_controls"
        if not selected:
            continue

        affordable_sorted = sorted(selected, key=lambda item: item.get("score", 0.0), reverse=True)
        affordable = [
            pick for pick in affordable_sorted
            if _capital_for_pick(pick) <= capital_budget
        ]
        rejected = {pick["_row_index"] for pick in affordable_sorted} - {pick["_row_index"] for pick in affordable}
        if rejected:
            working.loc[list(rejected), "gate_stage"] = "capital_budget"
            working.loc[list(rejected), "gate_reason"] = "single_contract_over_budget"
        if not affordable:
            continue

        affordable = apply_ml_position_sizing(
            affordable,
            config,
            max_contracts=int(config.get("max_contracts_per_pick", 50) or 50),
        )
        post_directional, overlay_rejections = apply_ml_quantity_overlays(
            affordable,
            [],
            config=config,
            account_capital=account_capital,
            available_capital=capital_budget,
            regime=regime,
            max_contracts=int(config.get("max_contracts_per_pick", 50) or 50),
        )
        for rejected_pick in overlay_rejections:
            row_index = rejected_pick.get("_row_index")
            if row_index is None:
                continue
            stage = str(rejected_pick.get("filtered_stage") or "")
            stage_key = {
                "Capital budget": "capital_budget",
                "Directional exposure": "directional_exposure",
                "Correlated cluster": "correlated_cluster",
                "Regime quantity throttle": "regime_quantity",
            }.get(stage, "quantity_overlay")
            working.loc[row_index, "gate_stage"] = stage_key
            working.loc[row_index, "gate_reason"] = str(
                rejected_pick.get("reject_reason") or stage_key
            )
        directional_idx = {pick["_row_index"] for pick in post_directional}
        if not post_directional:
            continue

        for pick in post_directional:
            row_index = pick["_row_index"]
            working.loc[row_index, "candidate_pass"] = True
            working.loc[row_index, "portfolio_score"] = float(pick.get("score") or 0.0)
            working.loc[row_index, "gate_stage"] = "selected"

    selected = working[np.isfinite(pd.to_numeric(working["portfolio_score"], errors="coerce"))].copy()
    selected = selected.sort_values(["entry_timestamp", "portfolio_score"], ascending=[True, False])
    print(
        f"[compare] Deterministic side: selected {len(selected):,} rows",
        file=sys.stderr,
        flush=True,
    )

    return {
        "name": "deterministic",
        "config_path": str(config_path),
        "runtime": {
            "max_dte": max_dte,
            "scanner_top_n": top_n,
            "pick_selection_mode": "equal_diversity",
            "max_capital_per_period": capital_budget,
            "auto_execute_prob": float(config.get("risk_parameters", {}).get("auto_execute_prob", 0.80)),
        },
        "reports": _build_side_reports(
            scored=working,
            candidate_mask=pd.Series(working["primary_pass"], index=working.index).astype(bool),
            selected=selected,
            gate_label="post_primary_gate",
            start_year=start_year,
            end_year=end_year,
        ),
        "selected_frame": selected,
        "gate_diagnostics": {
            "gate_stage_counts": _gate_stage_counts(working),
            "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(working),
            "quantity_reduction_counts": _quantity_reduction_counts(working),
        },
    }


def _build_side_reports(
    *,
    scored: pd.DataFrame,
    candidate_mask: pd.Series,
    selected: pd.DataFrame,
    gate_label: str,
    start_year: int,
    end_year: int,
) -> dict[str, Any]:
    quarter_labels = _quarter_labels(start_year, end_year)
    candidate = scored.loc[candidate_mask].copy()
    overall = _summary_report(scored, candidate, selected, f"{start_year}-{end_year}", gate_label)
    quarters = {
        quarter: _summary_report(
            scored.loc[scored["quarter"] == quarter].copy(),
            candidate.loc[candidate["quarter"] == quarter].copy(),
            selected.loc[selected["quarter"] == quarter].copy(),
            quarter,
            gate_label,
        )
        for quarter in quarter_labels
    }
    years = {
        str(year): _summary_report(
            scored.loc[pd.to_datetime(scored["entry_timestamp"], errors="coerce").dt.year == year].copy(),
            candidate.loc[pd.to_datetime(candidate["entry_timestamp"], errors="coerce").dt.year == year].copy(),
            selected.loc[pd.to_datetime(selected["entry_timestamp"], errors="coerce").dt.year == year].copy(),
            str(year),
            gate_label,
        )
        for year in range(start_year, end_year + 1)
    }
    return {
        "overall": overall,
        "years": years,
        "quarters": quarters,
    }


def _summary_report(
    scored: pd.DataFrame,
    candidate: pd.DataFrame,
    selected: pd.DataFrame,
    label: str,
    gate_label: str,
) -> dict[str, Any]:
    universe = _selection_stats(scored)
    candidate_stats = _selection_stats(candidate)
    chosen = _selection_stats(selected)
    return {
        "label": label,
        "universe": universe,
        gate_label: candidate_stats,
        "selected": chosen,
        "candidate_rejection_rate": _rate_from_counts(
            universe["rows"] - candidate_stats["rows"],
            universe["rows"],
        ),
        "selection_rate": _rate_from_counts(
            chosen["rows"],
            candidate_stats["rows"],
        ),
        "baseline_mean_pnl_delta": _delta(chosen.get("mean_pnl"), universe.get("mean_pnl")),
        "baseline_profit_factor_delta": _delta(
            chosen.get("profit_factor"),
            universe.get("profit_factor"),
        ),
        "gate_stage_counts": _gate_stage_counts(scored),
        "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(scored),
        "quantity_reduction_counts": _quantity_reduction_counts(scored),
    }


def _build_comparison(
    ml_selected: pd.DataFrame,
    deterministic_selected: pd.DataFrame,
    quarter_labels: list[str],
) -> dict[str, Any]:
    overall = _selected_delta_report(ml_selected, deterministic_selected, "overall")
    quarters = {
        quarter: _selected_delta_report(
            ml_selected.loc[ml_selected["quarter"] == quarter].copy(),
            deterministic_selected.loc[deterministic_selected["quarter"] == quarter].copy(),
            quarter,
        )
        for quarter in quarter_labels
    }
    wins = _win_counts(quarters)
    return {
        "overall": overall,
        "quarters": quarters,
        "win_counts": wins,
    }


def _selected_delta_report(
    ml_selected: pd.DataFrame,
    deterministic_selected: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    ml_stats = _selection_stats(ml_selected)
    deterministic_stats = _selection_stats(deterministic_selected)
    return {
        "label": label,
        "ml_selected": ml_stats,
        "deterministic_selected": deterministic_stats,
        "delta_ml_minus_det": {
            "rows": ml_stats["rows"] - deterministic_stats["rows"],
            "entry_dates": ml_stats["entry_dates"] - deterministic_stats["entry_dates"],
            "avg_trades_per_entry_date": _delta(
                ml_stats.get("avg_trades_per_entry_date"),
                deterministic_stats.get("avg_trades_per_entry_date"),
            ),
            "total_pnl": _delta(ml_stats.get("total_pnl"), deterministic_stats.get("total_pnl")),
            "mean_pnl": _delta(ml_stats.get("mean_pnl"), deterministic_stats.get("mean_pnl")),
            "profit_factor": _delta(
                ml_stats.get("profit_factor"),
                deterministic_stats.get("profit_factor"),
            ),
            "win_rate": _delta(ml_stats.get("win_rate"), deterministic_stats.get("win_rate")),
            "mean_return_on_risk": _delta(
                ml_stats.get("mean_return_on_risk"),
                deterministic_stats.get("mean_return_on_risk"),
            ),
            "large_loss_rate": _delta(
                ml_stats.get("large_loss_rate"),
                deterministic_stats.get("large_loss_rate"),
            ),
            "stop_loss_rate": _delta(
                ml_stats.get("stop_loss_rate"),
                deterministic_stats.get("stop_loss_rate"),
            ),
            "p05_pnl": _delta(ml_stats.get("p05_pnl"), deterministic_stats.get("p05_pnl")),
            "worst_pnl": _delta(ml_stats.get("worst_pnl"), deterministic_stats.get("worst_pnl")),
            "max_drawdown": _delta(
                ml_stats.get("max_drawdown"),
                deterministic_stats.get("max_drawdown"),
            ),
        },
    }


def _win_counts(quarters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "total_pnl": 0,
        "mean_pnl": 0,
        "profit_factor": 0,
        "win_rate": 0,
        "mean_return_on_risk": 0,
    }
    losses = {key: 0 for key in metrics}
    ties = {key: 0 for key in metrics}
    for report in quarters.values():
        delta = report["delta_ml_minus_det"]
        for key in metrics:
            value = delta.get(key)
            if value is None:
                ties[key] += 1
            elif value > 0:
                metrics[key] += 1
            elif value < 0:
                losses[key] += 1
            else:
                ties[key] += 1
    return {"wins": metrics, "losses": losses, "ties": ties}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ML vs Deterministic Backtest Comparison",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Window: {report['window']['start_year']} to {report['window']['end_year']}",
        "",
        "## Overall",
        "",
    ]
    overall = report["comparison"]["overall"]
    delta = overall["delta_ml_minus_det"]
    lines.extend(
        [
            f"- ML selected rows: {overall['ml_selected']['rows']}",
            f"- Deterministic selected rows: {overall['deterministic_selected']['rows']}",
            f"- Delta total PnL (ML - deterministic): {_fmt(delta.get('total_pnl'))}",
            f"- Delta mean PnL: {_fmt(delta.get('mean_pnl'))}",
            f"- Delta profit factor: {_fmt(delta.get('profit_factor'))}",
            f"- Delta win rate: {_fmt(delta.get('win_rate'))}",
            f"- Delta max drawdown: {_fmt(delta.get('max_drawdown'))}",
            "",
            "## Quarter by Quarter",
            "",
            "| Quarter | ML rows | Det rows | Delta total PnL | Delta mean PnL | Delta PF | Delta win rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for quarter in report["window"]["quarter_labels"]:
        q = report["comparison"]["quarters"][quarter]
        qd = q["delta_ml_minus_det"]
        lines.append(
            f"| {quarter} | {q['ml_selected']['rows']} | {q['deterministic_selected']['rows']} | "
            f"{_fmt(qd.get('total_pnl'))} | {_fmt(qd.get('mean_pnl'))} | "
            f"{_fmt(qd.get('profit_factor'))} | {_fmt(qd.get('win_rate'))} |"
        )
    lines.extend(
        [
            "",
            "## Replay Notes",
            "",
            "- Per-entry-date replay only; neither side carries a rolling open book across dates.",
            "- Quantities affect gate diagnostics but not PnL aggregation, matching the existing agent-like backtest semantics.",
            "- Deterministic replay proxies `prob_win` as `1 - abs(option_delta)` and skips the live open-interest gate because that field is not stored historically.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _strip_frame(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "selected_frame"}


def _filter_year_range(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    if "entry_timestamp" not in df:
        raise ValueError("Dataset is missing entry_timestamp; cannot run calendar comparison.")
    working = df.copy()
    working["entry_timestamp"] = pd.to_datetime(working["entry_timestamp"], errors="coerce", utc=True)
    working = working.dropna(subset=["entry_timestamp"])
    years = working["entry_timestamp"].dt.year
    return working.loc[years.between(start_year, end_year, inclusive="both")].copy()


def _quarter_labels(start_year: int, end_year: int) -> list[str]:
    labels: list[str] = []
    for year in range(start_year, end_year + 1):
        for quarter in range(1, 5):
            labels.append(f"{year}Q{quarter}")
    return labels


def _deterministic_top_n(config: dict[str, Any]) -> int:
    per_strategy = int(config.get("top_n_per_strategy", config.get("top_n_picks", 10)) or 10)
    enabled = 0
    for strategy in config.get("strategies", {}).values():
        if isinstance(strategy, dict) and strategy.get("enabled", False):
            enabled += 1
    return per_strategy * max(1, enabled)


def _primary_mask(group: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    strategies = config.get("strategies", {})
    pcs = strategies.get("put_credit_spread", {})
    ccs = strategies.get("call_credit_spread", {})

    premium = pd.to_numeric(group.get("entry_credit"), errors="coerce")
    delta = pd.to_numeric(group.get("option_delta"), errors="coerce").abs()
    prob = pd.to_numeric(group.get("prob_win_proxy"), errors="coerce")
    strategy = group.get("strategy", pd.Series(index=group.index, dtype=object)).astype("string").str.upper()

    pcs_mask = (
        (strategy == "PCS")
        & bool(pcs.get("enabled", False))
        & premium.ge(float(pcs.get("min_net_credit", 0.10)))
        & delta.le(float(pcs.get("max_delta_short_leg", 0.15)))
        & prob.ge(float(pcs.get("min_prob_profit", 0.80)))
    )
    ccs_mask = (
        (strategy == "CCS")
        & bool(ccs.get("enabled", False))
        & premium.ge(float(ccs.get("min_net_credit", 0.10)))
        & delta.le(float(ccs.get("max_delta_short_leg", 0.15)))
        & prob.ge(float(ccs.get("min_prob_profit", 0.80)))
    )
    return (pcs_mask | ccs_mask).fillna(False)


def _deterministic_scores(df: pd.DataFrame) -> pd.Series:
    premium = pd.to_numeric(df.get("entry_credit"), errors="coerce")
    width = pd.to_numeric(df.get("spread_width"), errors="coerce")
    credit_to_width = pd.to_numeric(df.get("credit_to_width"), errors="coerce")
    fallback = premium / width.replace({0: np.nan})
    normalized = credit_to_width.fillna(fallback)
    prob = pd.to_numeric(df.get("prob_win_proxy"), errors="coerce")
    return (normalized * np.square(prob)).astype(float)


def _pick_from_row(index: Any, row: pd.Series) -> dict[str, Any] | None:
    strategy = str(row.get("strategy") or "").upper()
    if strategy not in {"PCS", "CCS"}:
        return None
    spot = _positive_float(row.get("underlying_close"))
    short_strike = _positive_float(row.get("short_strike") or row.get("strike"))
    long_strike = _positive_float(row.get("long_strike"))
    dte = _positive_int(row.get("dte"))
    score = _float_value(row.get("score"))
    premium = _positive_float(row.get("entry_credit"))
    if None in {spot, short_strike, long_strike, dte, score, premium}:
        return None
    short_iv = _positive_float(row.get("implied_volatility")) or 0.25
    iv_skew = _float_value(row.get("iv_skew_wing")) or 0.0
    pick: dict[str, Any] = {
        "_row_index": index,
        "strategy": strategy,
        "symbol": str(row.get("underlying") or row.get("symbol") or "?"),
        "expiry": (date.today() + timedelta(days=dte)).isoformat(),
        "current_price": spot,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "premium": premium,
        "quantity": 1,
        "score": score,
        "prob_win": max(0.0, min(1.0, _float_value(row.get("prob_win_proxy")) or 0.0)),
        "short_iv": short_iv,
        "long_iv": max(0.01, short_iv + iv_skew),
    }
    if strategy == "PCS":
        pick["short_put"] = short_strike
        pick["long_put"] = long_strike
    else:
        pick["short_call"] = short_strike
        pick["long_call"] = long_strike
    return pick


def _group_regime_label(group: pd.DataFrame) -> str:
    series = group.get("allocator_regime_label", pd.Series(index=group.index, dtype=object))
    non_null = series.dropna()
    if non_null.empty:
        return "GREEN"
    return str(non_null.iloc[0]).strip().upper() or "GREEN"


def _regime_profile(config: dict[str, Any], label: str) -> RegimeProfile:
    cfg = config.get("risk_parameters", {}).get("regime_filter", {})
    normalized = label if label in {"GREEN", "YELLOW", "ORANGE", "RED"} else "GREEN"
    if normalized == "RED":
        return RegimeProfile(normalized, 1.0, 1.0, False)
    if normalized == "YELLOW":
        return RegimeProfile(
            normalized,
            float(cfg.get("yellow_quantity_multiplier", 0.65)),
            1.0,
            False,
        )
    if normalized == "ORANGE":
        return RegimeProfile(
            normalized,
            float(cfg.get("orange_quantity_multiplier", 0.30)),
            1.0,
            False,
        )
    return RegimeProfile("GREEN", 1.0, 1.0, False)


def _capital_for_pick(pick: dict[str, Any]) -> float:
    ss = float(pick.get("short_strike") or 0.0)
    ls = float(pick.get("long_strike") or 0.0)
    return abs(ss - ls) * 100.0


def _max_loss_per_contract_for_pick(pick: dict[str, Any]) -> float:
    width = abs(float(pick.get("short_strike") or 0.0) - float(pick.get("long_strike") or 0.0))
    premium = max(0.0, float(pick.get("premium") or 0.0))
    return max(0.0, round((width - premium) * 100.0, 2))


def _max_loss_multiple_for_pick(pick: dict[str, Any]) -> float:
    credit = max(0.0, float(pick.get("premium") or 0.0) * 100.0)
    if credit <= 0:
        return float("inf")
    return round(_max_loss_per_contract_for_pick(pick) / credit, 4)


def _max_loss_limit(config: dict[str, Any], strategy: str) -> float:
    cfg = config.get("risk_parameters", {}).get("max_loss_multiple", {})
    by_strategy = cfg.get("by_strategy", {})
    default = float(cfg.get("default", cfg.get("limit", 6.0)))
    return float(by_strategy.get(strategy.upper(), default))


def _strategy_sides(strategy: str) -> set[str]:
    normalized = strategy.upper()
    if normalized == "PCS":
        return {"put"}
    if normalized == "CCS":
        return {"call"}
    return set()


def _apply_directional_exposure_caps(
    picks: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    account_capital: float,
) -> list[dict[str, Any]]:
    cfg = config.get("risk_parameters", {}).get("directional_exposure_caps", {})
    if not cfg.get("enabled", True):
        return picks
    min_side_cap = float(cfg.get("min_side_cap_dollars", 0.0) or 0.0)
    put_limit = max(
        float(cfg.get("put", cfg.get("max_put_pct", 0.04))) * account_capital,
        float(cfg.get("min_put_cap_dollars", min_side_cap) or 0.0),
    )
    call_limit = max(
        float(cfg.get("call", cfg.get("max_call_pct", 0.04))) * account_capital,
        float(cfg.get("min_call_cap_dollars", min_side_cap) or 0.0),
    )
    limits = {"put": put_limit, "call": call_limit}
    used = {"put": 0.0, "call": 0.0}
    out: list[dict[str, Any]] = []
    for pick in sorted(picks, key=lambda item: item.get("score", 0.0), reverse=True):
        sides = _strategy_sides(str(pick.get("strategy") or ""))
        if not sides:
            continue
        requested_qty = int(pick.get("quantity") or 1)
        per_contract_loss = _max_loss_per_contract_for_pick(pick)
        if per_contract_loss <= 0:
            continue
        per_side_loss = per_contract_loss / len(sides)
        max_qty = requested_qty
        for side in sides:
            remaining = limits[side] - used[side]
            max_qty = min(max_qty, int(remaining // per_side_loss))
        if max_qty <= 0:
            continue
        pick["quantity"] = max_qty
        for side in sides:
            used[side] += per_side_loss * max_qty
        out.append(pick)
    return out


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _float_value(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _rate_from_counts(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a - b), 6)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
