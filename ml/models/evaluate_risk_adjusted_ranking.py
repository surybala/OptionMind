"""Compare two candidate-selection paths on the same holdout set.

Both paths share the same first two stages:

  1. XGBoost ranker  — scores every candidate with the trained model.
  2. Large-loss gate — removes candidates whose predicted large-loss
     probability exceeds ``max_large_loss_probability`` (default 0.70).

The surviving candidates are then ranked two ways and the top
``selection_fraction`` (default 10%) of each is evaluated:

  RoR path     — rank by the ranker score directly (predicted return-on-risk).
  Sortino path — rank by ranker score / (iv × √(dte/252)), i.e. predicted
                 RoR per unit of implied downside volatility.

Running both paths on every evaluation lets you compare which selection
criterion produces better realised outcomes on the holdout.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.evaluate_exit_criteria import ExitCriteriaConfig, _selection_metrics, _score_holdout
from ml.models.train_large_loss_classifier import _predict_prob, _transform_frame
from ml.models.train_xgboost import _engineer_features

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class RiskAdjustedConfig:
    selection_fraction: float = 0.10
    max_large_loss_probability: float | None = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RoR vs Sortino selection paths after large-loss classifier gating."
    )
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--ranker-artifact", required=True, help="XGBoost ranker artifact JSON.")
    parser.add_argument("--large-loss-artifact", default=None, help="Large-loss classifier artifact JSON.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--selection-fraction", type=float, default=RiskAdjustedConfig.selection_fraction)
    parser.add_argument(
        "--max-large-loss-probability",
        type=float,
        default=RiskAdjustedConfig.max_large_loss_probability,
        help="Veto candidates whose large-loss probability exceeds this threshold (default 0.70).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_risk_adjusted_ranking(
        Path(args.input),
        Path(args.ranker_artifact),
        large_loss_artifact=Path(args.large_loss_artifact) if args.large_loss_artifact else None,
        config=RiskAdjustedConfig(
            selection_fraction=args.selection_fraction,
            max_large_loss_probability=args.max_large_loss_probability,
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
    config: RiskAdjustedConfig | None = None,
) -> dict[str, Any]:
    """Run both selection paths and return side-by-side metrics.

    Parameters
    ----------
    dataset_path:
        Holdout dataset (same version used to train the ranker).
    ranker_artifact_path:
        XGBoost ranker artifact JSON produced by ``train_xgboost``.
    large_loss_artifact:
        Large-loss classifier artifact JSON.  When omitted the gate is skipped
        and all candidates proceed to ranking.
    config:
        Evaluation configuration.  Defaults to ``RiskAdjustedConfig()``.
    """
    if xgb is None:
        raise ImportError("xgboost is required to evaluate risk-adjusted ranking.")
    cfg = config or RiskAdjustedConfig()
    df = load_dataset(dataset_path)
    ranker_artifact = json.loads(ranker_artifact_path.read_text(encoding="utf-8"))

    # Stage 1 — XGBoost ranker
    scored = _score_holdout(df, ranker_artifact)

    # Stage 2 — large-loss classifier gate
    if large_loss_artifact:
        scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact)
    else:
        scored["large_loss_probability"] = 0.0
    scored["gated_score"] = apply_large_loss_gate(
        scored["prediction"],
        scored["large_loss_probability"],
        max_large_loss_probability=cfg.max_large_loss_probability,
    )

    gate_cfg = ExitCriteriaConfig(selection_fraction=cfg.selection_fraction)
    eligible_rows = int(np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce")).sum())

    # Stage 3a — RoR selection: top-N by gated ranker score (predicted RoR)
    ror_selection = _selection_metrics(scored, "gated_score", gate_cfg)

    # Stage 3b — Sortino selection: top-N by gated score / downside-vol factor
    scored["sortino_score"] = compute_sortino_score(scored["gated_score"], scored)
    sortino_selection = _selection_metrics(scored, "sortino_score", gate_cfg)

    return {
        "dataset_path": str(dataset_path),
        "ranker_artifact": str(ranker_artifact_path),
        "large_loss_artifact": str(large_loss_artifact) if large_loss_artifact else None,
        "config": asdict(cfg),
        "holdout_rows": int(len(scored)),
        "large_loss_eligible_rows": eligible_rows,
        "ror_selection": ror_selection,
        "sortino_selection": sortino_selection,
        "ror_vs_sortino_deltas": _metric_deltas(ror_selection, sortino_selection),
        "large_loss_probability_summary": _prob_summary(scored["large_loss_probability"]),
    }


def apply_large_loss_gate(
    score: Any,
    large_loss_probability: Any,
    *,
    max_large_loss_probability: float | None = 0.70,
) -> pd.Series:
    """Return score with high-large-loss-probability rows set to -inf."""
    gated = pd.to_numeric(pd.Series(score), errors="coerce").fillna(-np.inf).copy()
    if max_large_loss_probability is None:
        return gated
    large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(1.0)
    gated.loc[large > max_large_loss_probability] = -np.inf
    return gated


def compute_sortino_score(gated_score: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Per-trade Sortino proxy: gated_score / downside-vol factor.

    sortino_score = gated_score / max(iv × sqrt(dte / 252), ε)

    The denominator is the Black-Scholes expected-move factor — a per-trade
    proxy for downside standard deviation over the spread's remaining life.
    Trades vetoed by the large-loss gate (-inf score) remain -inf.
    """
    pred = pd.to_numeric(gated_score, errors="coerce").fillna(-np.inf)
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


def _score_classifier(df: pd.DataFrame, artifact_path: Path) -> np.ndarray:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    engineered = _engineer_features(df.copy())
    feature_columns = list(artifact["feature_columns"])
    frame = _transform_frame(engineered, feature_columns, dict(artifact.get("fill_values") or {}))
    booster = xgb.Booster()
    booster.load_model(str(artifact["model_path"]))
    return _predict_prob(booster, frame)


def _metric_deltas(ror: dict[str, Any], sortino: dict[str, Any]) -> dict[str, Any]:
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
        a = _float_or_none(ror.get(metric))
        b = _float_or_none(sortino.get(metric))
        out[metric] = None if a is None or b is None else round(b - a, 6)
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
