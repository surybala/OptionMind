"""Evaluate a ranker after penalizing large-loss and stop-loss risk."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.evaluate_exit_criteria import ExitCriteriaConfig, _selection_metrics, _score_holdout
from ml.models.portfolio_controls import (
    apply_portfolio_risk_controls,
    load_scanner_config,
    _entry_dates,
    _float_value,
    _pick_from_scored_row,
    _positive_float,
    _positive_int,
    _scanner_top_n,
)
from ml.models.train_large_loss_classifier import _predict_prob, _transform_frame
from ml.models.train_xgboost import _engineer_features

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class RiskAdjustedConfig:
    selection_fraction: float = 0.10
    large_loss_penalty_multiple: float = 0.0
    stop_loss_penalty_multiple: float = 0.0
    max_large_loss_probability: float | None = 0.70
    max_stop_loss_probability: float | None = 0.70
    portfolio_risk_controls: bool = True
    portfolio_account_capital: float = 50_000.0
    portfolio_max_candidates_per_entry_date: int = 50
    scanner_controls: bool = True
    scanner_config_path: str | None = "config.json"
    scanner_top_n_per_entry_date: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate risk-adjusted ranking from ranker and risk classifiers.")
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--ranker-artifact", required=True, help="Expected-P&L XGBoost artifact JSON.")
    parser.add_argument("--large-loss-artifact", default=None, help="Binary classifier artifact for large_loss_label.")
    parser.add_argument("--stop-loss-artifact", default=None, help="Binary classifier artifact for stop_loss_hit.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--selection-fraction", type=float, default=RiskAdjustedConfig.selection_fraction)
    parser.add_argument("--large-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.large_loss_penalty_multiple)
    parser.add_argument("--stop-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.stop_loss_penalty_multiple)
    parser.add_argument("--max-large-loss-probability", type=float, default=None)
    parser.add_argument("--max-stop-loss-probability", type=float, default=None)
    parser.add_argument(
        "--portfolio-risk-controls",
        action="store_true",
        default=True,
        help="Apply portfolio-level risk controls to selected spreads (default: on).",
    )
    parser.add_argument(
        "--no-portfolio-risk-controls",
        dest="portfolio_risk_controls",
        action="store_false",
        help="Disable portfolio-level risk controls.",
    )
    parser.add_argument("--portfolio-account-capital", type=float, default=RiskAdjustedConfig.portfolio_account_capital)
    parser.add_argument(
        "--portfolio-max-candidates-per-entry-date",
        type=int,
        default=RiskAdjustedConfig.portfolio_max_candidates_per_entry_date,
    )
    parser.add_argument("--scanner-config-path", default=RiskAdjustedConfig.scanner_config_path)
    parser.add_argument("--scanner-top-n-per-entry-date", type=int, default=None)
    parser.add_argument("--no-scanner-controls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_risk_adjusted_ranking(
        Path(args.input),
        Path(args.ranker_artifact),
        large_loss_artifact=Path(args.large_loss_artifact) if args.large_loss_artifact else None,
        stop_loss_artifact=Path(args.stop_loss_artifact) if args.stop_loss_artifact else None,
        config=RiskAdjustedConfig(
            selection_fraction=args.selection_fraction,
            large_loss_penalty_multiple=args.large_loss_penalty_multiple,
            stop_loss_penalty_multiple=args.stop_loss_penalty_multiple,
            max_large_loss_probability=args.max_large_loss_probability,
            max_stop_loss_probability=args.max_stop_loss_probability,
            portfolio_risk_controls=args.portfolio_risk_controls,
            portfolio_account_capital=args.portfolio_account_capital,
            portfolio_max_candidates_per_entry_date=args.portfolio_max_candidates_per_entry_date,
            scanner_controls=not args.no_scanner_controls,
            scanner_config_path=args.scanner_config_path,
            scanner_top_n_per_entry_date=args.scanner_top_n_per_entry_date,
        ),
    )
    payload = _jsonable(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def evaluate_risk_adjusted_ranking(
    dataset_path: Path,
    ranker_artifact_path: Path,
    *,
    large_loss_artifact: Path | None = None,
    stop_loss_artifact: Path | None = None,
    config: RiskAdjustedConfig | None = None,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required to evaluate risk-adjusted ranking.")
    cfg = config or RiskAdjustedConfig()
    df = load_dataset(dataset_path)
    ranker_artifact = json.loads(ranker_artifact_path.read_text(encoding="utf-8"))
    scored = _score_holdout(df, ranker_artifact)
    if large_loss_artifact:
        scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact)
    else:
        scored["large_loss_probability"] = 0.0
    if stop_loss_artifact:
        scored["stop_loss_probability"] = _score_classifier(scored, stop_loss_artifact)
    else:
        scored["stop_loss_probability"] = 0.0

    scored["risk_adjusted_score"] = risk_adjusted_score(
        scored["prediction"],
        scored.get("max_loss", 0.0),
        scored["large_loss_probability"],
        scored["stop_loss_probability"],
        large_loss_penalty_multiple=cfg.large_loss_penalty_multiple,
        stop_loss_penalty_multiple=cfg.stop_loss_penalty_multiple,
    )
    scored["risk_adjusted_score"] = apply_probability_caps(
        scored["risk_adjusted_score"],
        scored["large_loss_probability"],
        scored["stop_loss_probability"],
        max_large_loss_probability=cfg.max_large_loss_probability,
        max_stop_loss_probability=cfg.max_stop_loss_probability,
    )
    gate_cfg = ExitCriteriaConfig(selection_fraction=cfg.selection_fraction)
    raw = _selection_metrics(scored, "prediction", gate_cfg)
    adjusted = _selection_metrics(scored, "risk_adjusted_score", gate_cfg)
    portfolio_selection = None
    portfolio_deltas = None
    portfolio_eligible_rows = None
    scanner_config = load_scanner_config(cfg.scanner_config_path)
    if cfg.portfolio_risk_controls:
        scored["portfolio_risk_score"] = apply_portfolio_risk_controls(
            scored,
            "risk_adjusted_score",
            account_capital=cfg.portfolio_account_capital,
            max_candidates_per_entry_date=cfg.portfolio_max_candidates_per_entry_date,
            scanner_controls=cfg.scanner_controls,
            scanner_config=scanner_config,
            scanner_top_n_per_entry_date=cfg.scanner_top_n_per_entry_date,
        )
        portfolio_eligible = scored[np.isfinite(pd.to_numeric(scored["portfolio_risk_score"], errors="coerce"))]
        portfolio_eligible_rows = int(len(portfolio_eligible))
        portfolio_selection = (
            _selection_metrics(portfolio_eligible, "portfolio_risk_score", ExitCriteriaConfig(selection_fraction=1.0))
            if portfolio_eligible_rows
            else {"rows": int(len(scored)), "selected_rows": 0}
        )
        portfolio_deltas = _metric_deltas(raw, portfolio_selection)
    return {
        "dataset_path": str(dataset_path),
        "ranker_artifact": str(ranker_artifact_path),
        "large_loss_artifact": str(large_loss_artifact) if large_loss_artifact else None,
        "stop_loss_artifact": str(stop_loss_artifact) if stop_loss_artifact else None,
        "config": asdict(cfg),
        "holdout_rows": int(len(scored)),
        "risk_eligible_rows": int(np.isfinite(pd.to_numeric(scored["risk_adjusted_score"], errors="coerce")).sum()),
        "raw_selection": raw,
        "risk_adjusted_selection": adjusted,
        "portfolio_risk_selection": portfolio_selection,
        "deltas": _metric_deltas(raw, adjusted),
        "portfolio_risk_deltas": portfolio_deltas,
        "risk_probability_summary": {
            "large_loss_probability": _prob_summary(scored["large_loss_probability"]),
            "stop_loss_probability": _prob_summary(scored["stop_loss_probability"]),
        },
        "portfolio_risk_eligible_rows": portfolio_eligible_rows,
    }


def risk_adjusted_score(
    prediction: Any,
    max_loss: Any,
    large_loss_probability: Any,
    stop_loss_probability: Any,
    *,
    large_loss_penalty_multiple: float = 1.0,
    stop_loss_penalty_multiple: float = 0.50,
) -> pd.Series:
    pred = pd.to_numeric(pd.Series(prediction), errors="coerce").fillna(-np.inf)
    loss = pd.to_numeric(pd.Series(max_loss), errors="coerce").fillna(0.0).clip(lower=0.0)
    large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    stop = pd.to_numeric(pd.Series(stop_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    penalty = loss * (large_loss_penalty_multiple * large + stop_loss_penalty_multiple * stop)
    return pred - penalty


def apply_probability_caps(
    score: Any,
    large_loss_probability: Any,
    stop_loss_probability: Any,
    *,
    max_large_loss_probability: float | None = None,
    max_stop_loss_probability: float | None = None,
) -> pd.Series:
    capped = pd.to_numeric(pd.Series(score), errors="coerce").fillna(-np.inf).copy()
    if max_large_loss_probability is not None:
        large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(1.0)
        capped.loc[large > max_large_loss_probability] = -np.inf
    if max_stop_loss_probability is not None:
        stop = pd.to_numeric(pd.Series(stop_loss_probability), errors="coerce").fillna(1.0)
        capped.loc[stop > max_stop_loss_probability] = -np.inf
    return capped




def _score_classifier(df: pd.DataFrame, artifact_path: Path) -> np.ndarray:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    engineered = _engineer_features(df.copy())
    feature_columns = list(artifact["feature_columns"])
    frame = _transform_frame(engineered, feature_columns, dict(artifact.get("fill_values") or {}))
    booster = xgb.Booster()
    booster.load_model(str(artifact["model_path"]))
    return _predict_prob(booster, frame)


def _metric_deltas(raw: dict[str, Any], adjusted: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "mean_pnl",
        "slippage_adjusted_mean_pnl",
        "profit_factor",
        "slippage_adjusted_profit_factor",
        "win_rate",
        "large_loss_rate",
        "stop_loss_rate",
        "p05_pnl",
        "worst_pnl",
        "max_drawdown",
        "single_underlying_share",
    )
    out: dict[str, Any] = {}
    for metric in metrics:
        raw_value = _float_or_none(raw.get(metric))
        adjusted_value = _float_or_none(adjusted.get(metric))
        out[metric] = None if raw_value is None or adjusted_value is None else round(adjusted_value - raw_value, 6)
    return out


def _prob_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "mean": round(float(numeric.mean()), 6),
        "p50": round(float(numeric.quantile(0.50)), 6),
        "p90": round(float(numeric.quantile(0.90)), 6),
        "p95": round(float(numeric.quantile(0.95)), 6),
        "p99": round(float(numeric.quantile(0.99)), 6),
        "max": round(float(numeric.max()), 6),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)




def _jsonable(report: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(report, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
