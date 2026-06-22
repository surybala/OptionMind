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
from ml.models.portfolio_controls import apply_portfolio_risk_controls
from ml.models.train_large_loss_classifier import _predict_prob, _transform_frame
from ml.models.train_xgboost import _engineer_features
from src.utils import load_config

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class RiskAdjustedConfig:
    selection_fraction: float = 0.10
    max_large_loss_probability: float | None = 0.70
    max_stop_loss_probability: float | None = None
    apply_portfolio_risk_controls: bool = False
    account_capital: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RoR vs Sortino selection paths after large-loss classifier gating."
    )
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--ranker-artifact", required=True, help="XGBoost ranker artifact JSON.")
    parser.add_argument("--large-loss-artifact", default=None, help="Large-loss classifier artifact JSON.")
    parser.add_argument("--stop-loss-artifact", default=None, help="Stop-loss classifier artifact JSON.")
    parser.add_argument("--runtime-config", default=None, help="Optional runtime config.json path for live-faithful thresholds and portfolio controls.")
    parser.add_argument("--apply-portfolio-risk-controls", action="store_true")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--selection-fraction", type=float, default=RiskAdjustedConfig.selection_fraction)
    parser.add_argument(
        "--max-large-loss-probability",
        type=float,
        default=RiskAdjustedConfig.max_large_loss_probability,
        help="Veto candidates whose large-loss probability exceeds this threshold (default 0.70).",
    )
    parser.add_argument(
        "--max-stop-loss-probability",
        type=float,
        default=None,
        help="Optional stop-loss veto threshold. When omitted, runtime config or artifact metadata can supply it.",
    )
    parser.add_argument(
        "--account-capital",
        type=float,
        default=None,
        help="Optional capital budget passed to portfolio controls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_risk_adjusted_ranking(
        Path(args.input),
        Path(args.ranker_artifact),
        large_loss_artifact=Path(args.large_loss_artifact) if args.large_loss_artifact else None,
        stop_loss_artifact=Path(args.stop_loss_artifact) if args.stop_loss_artifact else None,
        runtime_config_path=Path(args.runtime_config) if args.runtime_config else None,
        config=RiskAdjustedConfig(
            selection_fraction=args.selection_fraction,
            max_large_loss_probability=args.max_large_loss_probability,
            max_stop_loss_probability=args.max_stop_loss_probability,
            apply_portfolio_risk_controls=args.apply_portfolio_risk_controls,
            account_capital=args.account_capital,
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
    runtime_config_path: Path | None = None,
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
    cfg = config or RiskAdjustedConfig()
    df = load_dataset(dataset_path)
    ranker_artifact = json.loads(ranker_artifact_path.read_text(encoding="utf-8"))
    runtime_config = load_config(str(runtime_config_path)) if runtime_config_path else {}
    large_loss_threshold = _resolve_threshold(
        explicit=cfg.max_large_loss_probability,
        runtime_config=runtime_config,
        config_key="large_loss_veto_threshold",
    )
    stop_loss_threshold = _resolve_threshold(
        explicit=cfg.max_stop_loss_probability,
        runtime_config=runtime_config,
        config_key="stop_loss_veto_threshold",
    )

    # Stage 1 — XGBoost ranker
    scored = _score_holdout(df, ranker_artifact)
    gate_cfg = ExitCriteriaConfig(selection_fraction=cfg.selection_fraction)
    raw_selection = _selection_metrics(scored, "prediction", gate_cfg)

    # Stage 2 — large-loss classifier gate
    if large_loss_artifact:
        scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact)
    else:
        scored["large_loss_probability"] = 0.0
    scored["large_loss_gate_score"] = apply_large_loss_gate(
        scored["prediction"],
        scored["large_loss_probability"],
        max_large_loss_probability=large_loss_threshold,
    )
    large_loss_gate_selection = _selection_metrics(scored, "large_loss_gate_score", gate_cfg)
    large_loss_eligible_rows = int(np.isfinite(pd.to_numeric(scored["large_loss_gate_score"], errors="coerce")).sum())

    # Stage 3 — stop-loss classifier gate
    if stop_loss_artifact:
        scored["stop_loss_probability"] = _score_classifier(scored, stop_loss_artifact)
    else:
        scored["stop_loss_probability"] = 0.0
    scored["stop_loss_gate_score"] = apply_large_loss_gate(
        scored["large_loss_gate_score"],
        scored["stop_loss_probability"],
        max_large_loss_probability=stop_loss_threshold,
    )
    stop_loss_gate_selection = _selection_metrics(scored, "stop_loss_gate_score", gate_cfg)
    stop_loss_eligible_rows = int(np.isfinite(pd.to_numeric(scored["stop_loss_gate_score"], errors="coerce")).sum())

    # Stage 4 — portfolio controls using live thresholds/config when requested
    if cfg.apply_portfolio_risk_controls:
        scored["allocator_regime_label"] = _allocator_regime_labels(scored)
        account_capital = float(
            cfg.account_capital
            if cfg.account_capital is not None
            else runtime_config.get("max_capital_per_period", 50_000.0)
        )
        portfolio_score, portfolio_diagnostics = apply_portfolio_risk_controls(
            scored,
            "stop_loss_gate_score",
            account_capital=account_capital,
            scanner_controls=True,
            scanner_config=runtime_config,
            regime_label_column="allocator_regime_label",
            return_diagnostics=True,
        )
        scored["portfolio_score"] = portfolio_score
        portfolio_diagnostics_summary: dict[str, Any] = {}
        for key, value in portfolio_diagnostics.items():
            if isinstance(value, pd.Series) and len(value) == len(scored):
                scored[key] = value
            else:
                portfolio_diagnostics_summary[key] = value
    else:
        account_capital = cfg.account_capital
        scored["portfolio_score"] = scored["stop_loss_gate_score"]
        portfolio_diagnostics_summary = {}

    trade_pipeline_selection = _selection_metrics(scored, "portfolio_score", gate_cfg)
    trade_pipeline_eligible_rows = int(np.isfinite(pd.to_numeric(scored["portfolio_score"], errors="coerce")).sum())

    # Legacy side-by-side path: RoR vs Sortino after both ML gates.
    ror_selection = stop_loss_gate_selection
    scored["sortino_score"] = compute_sortino_score(scored["stop_loss_gate_score"], scored)
    sortino_selection = _selection_metrics(scored, "sortino_score", gate_cfg)

    return {
        "dataset_path": str(dataset_path),
        "ranker_artifact": str(ranker_artifact_path),
        "large_loss_artifact": str(large_loss_artifact) if large_loss_artifact else None,
        "stop_loss_artifact": str(stop_loss_artifact) if stop_loss_artifact else None,
        "runtime_config_path": str(runtime_config_path) if runtime_config_path else None,
        "config": asdict(cfg),
        "holdout_rows": int(len(scored)),
        "raw_selection": raw_selection,
        "large_loss_gate_selection": large_loss_gate_selection,
        "stop_loss_gate_selection": stop_loss_gate_selection,
        "trade_pipeline_selection": trade_pipeline_selection,
        "trade_pipeline_eligible_rows": trade_pipeline_eligible_rows,
        "trade_pipeline_deltas": {
            "rows_after_large_loss_gate": large_loss_eligible_rows,
            "rows_after_stop_loss_gate": stop_loss_eligible_rows,
            "rows_after_trade_pipeline": trade_pipeline_eligible_rows,
            "mean_pnl_delta_vs_raw": _delta_metric(trade_pipeline_selection, raw_selection, "mean_pnl"),
            "profit_factor_delta_vs_raw": _delta_metric(trade_pipeline_selection, raw_selection, "profit_factor"),
            "win_rate_delta_vs_raw": _delta_metric(trade_pipeline_selection, raw_selection, "win_rate"),
        },
        "trade_pipeline": {
            "stages": [
                "ranker",
                "large_loss_gate",
                "stop_loss_gate",
                "portfolio_controls" if cfg.apply_portfolio_risk_controls else "no_portfolio_controls",
            ],
            "final_score_column": "portfolio_score",
            "large_loss_threshold": large_loss_threshold,
            "stop_loss_threshold": stop_loss_threshold,
            "portfolio_controls_applied": bool(cfg.apply_portfolio_risk_controls),
            "account_capital": account_capital,
            "portfolio_diagnostics_summary": portfolio_diagnostics_summary,
        },
        "large_loss_eligible_rows": large_loss_eligible_rows,
        "stop_loss_eligible_rows": stop_loss_eligible_rows,
        "ror_selection": ror_selection,
        "sortino_selection": sortino_selection,
        "ror_vs_sortino_deltas": _metric_deltas(ror_selection, sortino_selection),
        "large_loss_probability_summary": _prob_summary(scored["large_loss_probability"]),
        "stop_loss_probability_summary": _prob_summary(scored["stop_loss_probability"]),
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
    if xgb is None:
        raise ImportError("xgboost is required to score risk classifier artifacts.")
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


def _delta_metric(current: dict[str, Any], baseline: dict[str, Any], metric: str) -> float | None:
    a = _float_or_none(current.get(metric))
    b = _float_or_none(baseline.get(metric))
    if a is None or b is None:
        return None
    return round(a - b, 6)


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


def _resolve_threshold(
    *,
    explicit: float | None,
    runtime_config: dict[str, Any],
    config_key: str,
) -> float | None:
    if explicit is not None:
        return explicit
    scanner_cfg = runtime_config.get("ml_scanner", {}) if isinstance(runtime_config.get("ml_scanner", {}), dict) else {}
    value = scanner_cfg.get(config_key)
    return float(value) if value is not None else None


def _allocator_regime_labels(df: pd.DataFrame) -> pd.Series:
    trend = df.get("market_trend_regime", pd.Series(index=df.index, dtype=object)).astype("string").str.lower()
    vol = df.get("market_volatility_regime", pd.Series(index=df.index, dtype=object)).astype("string").str.lower()
    vix = pd.to_numeric(df.get("vix_regime", pd.Series(index=df.index, dtype=float)), errors="coerce")

    labels = pd.Series("GREEN", index=df.index, dtype="string")
    yellow = trend.isin(["downtrend", "sideways"]) | vol.isin(["high", "normal"]) | (vix >= 1.0)
    orange = ((trend == "downtrend") & (vol.isin(["high", "normal"]))) | (vix >= 2.0)
    labels.loc[yellow.fillna(False)] = "YELLOW"
    labels.loc[orange.fillna(False)] = "ORANGE"
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
