"""Evaluate the ML trade pipeline used to select credit-spread trades.

The production-shaped funnel is:
1. Score candidates with the XGBoost ranker.
2. Remove candidates with excessive large-tail-loss probability.
3. Apply the same portfolio gamma stress caps used by live selection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

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
    risk_penalty_basis: Literal["auto", "dollars", "return_on_risk"] = "auto"
    max_large_loss_probability: float | None = 0.70
    max_stop_loss_probability: float | None = 0.70
    portfolio_risk_controls: bool = True
    portfolio_account_capital: float = 50_000.0
    portfolio_max_candidates_per_entry_date: int = 50
    scanner_controls: bool = True
    scanner_config_path: str | None = "config.json"
    scanner_top_n_per_entry_date: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ML trade pipeline from ranker, risk classifiers, and portfolio risk gates.")
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--ranker-artifact", required=True, help="XGBoost ranker artifact JSON.")
    parser.add_argument("--large-loss-artifact", default=None, help="Binary classifier artifact for large_loss_label.")
    parser.add_argument("--stop-loss-artifact", default=None, help="Optional binary classifier artifact for stop_loss_hit diagnostics.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--selection-fraction", type=float, default=RiskAdjustedConfig.selection_fraction)
    parser.add_argument("--large-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.large_loss_penalty_multiple)
    parser.add_argument("--stop-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.stop_loss_penalty_multiple)
    parser.add_argument(
        "--risk-penalty-basis",
        choices=["auto", "dollars", "return_on_risk"],
        default=RiskAdjustedConfig.risk_penalty_basis,
        help=(
            "Unit used for probability penalties. 'auto' uses dollars for expected_pnl "
            "rankers and return_on_risk units for return_on_risk rankers."
        ),
    )
    parser.add_argument("--max-large-loss-probability", type=float, default=None)
    parser.add_argument("--max-stop-loss-probability", type=float, default=None)
    parser.add_argument(
        "--no-large-loss-probability-cap",
        action="store_true",
        help="Disable the large-loss probability gate in the trade pipeline.",
    )
    parser.add_argument(
        "--no-stop-loss-probability-cap",
        action="store_true",
        help="Disable the stop-loss probability cap.",
    )
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
            risk_penalty_basis=args.risk_penalty_basis,
            max_large_loss_probability=_cap_value(
                args.max_large_loss_probability,
                RiskAdjustedConfig.max_large_loss_probability,
                disabled=args.no_large_loss_probability_cap,
            ),
            max_stop_loss_probability=_cap_value(
                args.max_stop_loss_probability,
                RiskAdjustedConfig.max_stop_loss_probability,
                disabled=args.no_stop_loss_probability_cap,
            ),
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
    risk_penalty_basis = _resolve_risk_penalty_basis(
        cfg.risk_penalty_basis,
        str(ranker_artifact.get("target_column") or ""),
    )
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
        risk_penalty_basis=risk_penalty_basis,
    )
    scored["risk_adjusted_score"] = apply_probability_caps(
        scored["risk_adjusted_score"],
        scored["large_loss_probability"],
        scored["stop_loss_probability"],
        max_large_loss_probability=cfg.max_large_loss_probability,
        max_stop_loss_probability=cfg.max_stop_loss_probability,
    )
    gate_cfg = ExitCriteriaConfig(selection_fraction=cfg.selection_fraction)
    ror_selection = _selection_metrics(scored, "risk_adjusted_score", gate_cfg)
    scored["sortino_score"] = compute_sortino_score(scored["risk_adjusted_score"], scored)
    sortino_selection = _selection_metrics(scored, "sortino_score", gate_cfg)
    portfolio_selection = None
    portfolio_deltas = None
    portfolio_eligible_rows = None
    scanner_config = load_scanner_config(cfg.scanner_config_path)
    trade_pipeline = _evaluate_trade_pipeline(scored, cfg, scanner_config)
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
        portfolio_deltas = _metric_deltas(ror_selection, portfolio_selection)
    return {
        "dataset_path": str(dataset_path),
        "ranker_artifact": str(ranker_artifact_path),
        "large_loss_artifact": str(large_loss_artifact) if large_loss_artifact else None,
        "stop_loss_artifact": str(stop_loss_artifact) if stop_loss_artifact else None,
        "config": asdict(cfg),
        "resolved_risk_penalty_basis": risk_penalty_basis,
        "holdout_rows": int(len(scored)),
        "risk_eligible_rows": int(np.isfinite(pd.to_numeric(scored["risk_adjusted_score"], errors="coerce")).sum()),
        "ror_selection": ror_selection,
        "sortino_selection": sortino_selection,
        "ror_vs_sortino_deltas": _metric_deltas(ror_selection, sortino_selection),
        "portfolio_risk_selection": portfolio_selection,
        "large_loss_gate_selection": trade_pipeline["large_loss_gate_selection"],
        "trade_pipeline_selection": trade_pipeline["trade_pipeline_selection"],
        "portfolio_risk_deltas": portfolio_deltas,
        "large_loss_gate_deltas": _metric_deltas(ror_selection, trade_pipeline["large_loss_gate_selection"]),
        "trade_pipeline_deltas": _metric_deltas(ror_selection, trade_pipeline["trade_pipeline_selection"]),
        "risk_probability_summary": {
            "large_loss_probability": _prob_summary(scored["large_loss_probability"]),
            "stop_loss_probability": _prob_summary(scored["stop_loss_probability"]),
        },
        "portfolio_risk_eligible_rows": portfolio_eligible_rows,
        "large_loss_gate_eligible_rows": trade_pipeline["large_loss_gate_eligible_rows"],
        "trade_pipeline_eligible_rows": trade_pipeline["trade_pipeline_eligible_rows"],
        "trade_pipeline": {
            "stages": [
                "xgboost_ranker",
                "large_loss_classifier_gate",
                "portfolio_gamma_stress_caps",
            ],
            "ranker_score_column": "prediction",
            "large_loss_gate_score_column": "large_loss_gate_score",
            "final_score_column": "trade_pipeline_score",
            "large_loss_probability_cap": cfg.max_large_loss_probability,
            "portfolio_gamma_stress_caps": bool(cfg.portfolio_risk_controls),
        },
    }


def risk_adjusted_score(
    prediction: Any,
    max_loss: Any,
    large_loss_probability: Any,
    stop_loss_probability: Any,
    *,
    large_loss_penalty_multiple: float = 1.0,
    stop_loss_penalty_multiple: float = 0.50,
    risk_penalty_basis: Literal["dollars", "return_on_risk"] = "dollars",
) -> pd.Series:
    pred = pd.to_numeric(pd.Series(prediction), errors="coerce").fillna(-np.inf)
    loss = pd.to_numeric(pd.Series(max_loss), errors="coerce").fillna(0.0).clip(lower=0.0)
    large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    stop = pd.to_numeric(pd.Series(stop_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    if risk_penalty_basis == "return_on_risk":
        penalty_base = pd.Series(1.0, index=pred.index, dtype=float)
    elif risk_penalty_basis == "dollars":
        penalty_base = loss
    else:
        raise ValueError("risk_penalty_basis must be 'dollars' or 'return_on_risk'")
    penalty = penalty_base * (large_loss_penalty_multiple * large + stop_loss_penalty_multiple * stop)
    return pred - penalty


def _resolve_risk_penalty_basis(
    requested: Literal["auto", "dollars", "return_on_risk"],
    target_column: str,
) -> Literal["dollars", "return_on_risk"]:
    if requested != "auto":
        return requested
    return "return_on_risk" if target_column == "return_on_risk" else "dollars"


def _cap_value(value: float | None, default: float | None, *, disabled: bool = False) -> float | None:
    if disabled:
        return None
    return default if value is None else value


def _evaluate_trade_pipeline(
    scored: pd.DataFrame,
    cfg: RiskAdjustedConfig,
    scanner_config: dict[str, Any],
) -> dict[str, Any]:
    scored["large_loss_gate_score"] = apply_large_loss_gate(
        scored["prediction"],
        scored["large_loss_probability"],
        max_large_loss_probability=cfg.max_large_loss_probability,
    )
    large_loss_eligible = scored[np.isfinite(pd.to_numeric(scored["large_loss_gate_score"], errors="coerce"))]
    large_loss_selection = (
        _selection_metrics(large_loss_eligible, "large_loss_gate_score", ExitCriteriaConfig(selection_fraction=1.0))
        if len(large_loss_eligible)
        else {"rows": int(len(scored)), "selected_rows": 0}
    )

    if cfg.portfolio_risk_controls:
        scored["trade_pipeline_score"] = apply_portfolio_risk_controls(
            scored,
            "large_loss_gate_score",
            account_capital=cfg.portfolio_account_capital,
            max_candidates_per_entry_date=cfg.portfolio_max_candidates_per_entry_date,
            scanner_controls=cfg.scanner_controls,
            scanner_config=scanner_config,
            scanner_top_n_per_entry_date=cfg.scanner_top_n_per_entry_date,
        )
    else:
        scored["trade_pipeline_score"] = scored["large_loss_gate_score"]

    trade_pipeline_eligible = scored[np.isfinite(pd.to_numeric(scored["trade_pipeline_score"], errors="coerce"))]
    trade_pipeline_selection = (
        _selection_metrics(trade_pipeline_eligible, "trade_pipeline_score", ExitCriteriaConfig(selection_fraction=1.0))
        if len(trade_pipeline_eligible)
        else {"rows": int(len(scored)), "selected_rows": 0}
    )
    return {
        "large_loss_gate_eligible_rows": int(len(large_loss_eligible)),
        "large_loss_gate_selection": large_loss_selection,
        "trade_pipeline_eligible_rows": int(len(trade_pipeline_eligible)),
        "trade_pipeline_selection": trade_pipeline_selection,
    }


def apply_large_loss_gate(
    score: Any,
    large_loss_probability: Any,
    *,
    max_large_loss_probability: float | None = 0.70,
) -> pd.Series:
    gated = pd.to_numeric(pd.Series(score), errors="coerce").fillna(-np.inf).copy()
    if max_large_loss_probability is None:
        return gated
    large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(1.0)
    gated.loc[large > max_large_loss_probability] = -np.inf
    return gated


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


def compute_sortino_score(risk_adjusted_score: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Per-trade Sortino proxy: risk_adjusted_score / downside-vol factor.

    sortino_score = risk_adjusted_score / max(iv × sqrt(dte / 252), ε)

    The denominator (Black-Scholes expected-move factor) is a per-trade proxy for
    downside standard deviation over the spread's remaining life.  It up-ranks
    trades that earn the same predicted RoR with lower underlying volatility or
    shorter DTE, favouring low-risk capital efficiency.

    Classifier-vetoed rows (-inf score) remain -inf after division.
    """
    pred = pd.to_numeric(risk_adjusted_score, errors="coerce").fillna(-np.inf)
    iv = (
        pd.to_numeric(df.get("implied_volatility", pd.Series(dtype=float)), errors="coerce")
        .fillna(0.25)
        .clip(lower=0.01)
    )
    dte = (
        pd.to_numeric(df.get("dte", pd.Series(dtype=float)), errors="coerce")
        .fillna(30.0)
        .clip(lower=1.0)
    )
    vol_factor = (iv * np.sqrt(dte / 252.0)).clip(lower=1e-4)
    result = pd.Series(-np.inf, index=df.index, dtype=float)
    finite_mask = np.isfinite(pred)
    result.loc[finite_mask] = pred.loc[finite_mask] / vol_factor.loc[finite_mask]
    return result


def _metric_deltas(raw: dict[str, Any], adjusted: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "mean_pnl",
        "mean_return_on_risk",
        "sortino_ratio",
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
