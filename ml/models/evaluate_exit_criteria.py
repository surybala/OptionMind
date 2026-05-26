"""Evaluate whether an XGBoost credit-spread ranker is production-candidate ready.

This module is intentionally strict: AutoML should keep searching until every
hard gate passes.  The score is useful for ranking failures, but it is never a
substitute for the pass/fail gates.
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
from ml.models.train_baseline import _max_drawdown, _profit_factor, _split_index
from ml.models.train_xgboost import (
    AsymmetricLossConfig,
    _engineer_features,
    _inverse_transform_target,
    _transform_xgb_frame,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - only exercised when native dependency is missing
    xgb = None


@dataclass(frozen=True)
class ExitCriteriaConfig:
    """Hard gates for a production-candidate credit-spread ranker."""

    min_dataset_rows: int = 500_000
    min_test_rows: int = 100_000
    min_top_selection_rows: int = 100
    min_top_selection_entry_dates: int = 60
    min_walk_forward_folds: int = 3
    selection_fraction: float = 0.10
    max_single_underlying_share: float = 0.25
    max_top5_underlying_share: float = 0.65
    slippage_penalty_fraction: float = 0.20
    catastrophic_account_limit: float = 4_000.0
    max_drawdown_to_catastrophic_limit: float = 0.50
    max_mae_to_catastrophic_limit: float = 0.50
    feature_stability_top_k: int = 3
    min_feature_top_k_overlap: int = 3

    min_holdout_top_mean_pnl: float = 20.0
    min_holdout_top_slippage_adjusted_mean_pnl: float = 0.0
    min_holdout_top_slippage_adjusted_profit_factor: float = 1.0
    min_holdout_top_profit_factor: float = 1.35
    min_holdout_top_win_rate: float = 0.58
    min_holdout_top_return_on_risk: float = 0.20
    min_holdout_top_p05_pnl: float = -250.0
    min_holdout_top_p01_pnl: float = -600.0
    min_holdout_top_worst_pnl: float = -1_500.0
    max_holdout_top_large_loss_rate: float = 0.15
    max_holdout_top_stop_loss_rate: float = 0.30
    max_holdout_top_drawdown_to_gross_profit: float = 0.45

    min_walk_forward_top_mean_pnl: float = 10.0
    min_walk_forward_top_profit_factor: float = 1.20
    min_walk_forward_top_win_rate: float = 0.55
    min_walk_forward_top_p05_pnl: float = -300.0
    min_walk_forward_top_worst_pnl: float = -2_000.0
    min_walk_forward_avg_top_mean_pnl: float = 25.0
    min_walk_forward_avg_top_profit_factor: float = 1.35

    max_train_holdout_top_mean_ratio: float = 2.50
    max_train_holdout_profit_factor_ratio: float = 2.50


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    actual: Any
    threshold: Any
    direction: str


@dataclass(frozen=True)
class ExitCriteriaReport:
    passed: bool
    score: float
    artifact_path: str
    model_type: str
    target_column: str
    dataset_rows: int
    holdout: dict[str, Any]
    walk_forward: dict[str, Any]
    criteria: list[CriterionResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate production exit criteria for an XGBoost ranker.")
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--artifact", required=True, help="XGBoost model artifact JSON.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report path.")
    parser.add_argument("--selection-fraction", type=float, default=ExitCriteriaConfig.selection_fraction)
    parser.add_argument("--catastrophic-account-limit", type=float, default=ExitCriteriaConfig.catastrophic_account_limit)
    parser.add_argument("--slippage-penalty-fraction", type=float, default=ExitCriteriaConfig.slippage_penalty_fraction)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExitCriteriaConfig(
        selection_fraction=args.selection_fraction,
        catastrophic_account_limit=args.catastrophic_account_limit,
        slippage_penalty_fraction=args.slippage_penalty_fraction,
    )
    report = evaluate_exit_criteria(Path(args.input), Path(args.artifact), config=config)
    payload = _report_to_dict(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.passed else 2


def evaluate_exit_criteria(
    dataset_path: Path,
    artifact_path: Path,
    *,
    config: ExitCriteriaConfig | None = None,
) -> ExitCriteriaReport:
    if xgb is None:
        raise ImportError("xgboost is required to score model exit criteria")
    cfg = config or ExitCriteriaConfig()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    df = load_dataset(dataset_path)
    scored = _score_holdout(df, artifact)
    holdout = _selection_metrics(scored, "prediction", cfg)
    walk_forward = _walk_forward_summary(artifact)
    criteria = _evaluate_gates(df, artifact, holdout, walk_forward, cfg)
    score = _automl_score(holdout, walk_forward, criteria)
    return ExitCriteriaReport(
        passed=all(item.passed for item in criteria),
        score=score,
        artifact_path=str(artifact_path),
        model_type=str(artifact.get("model_type") or ""),
        target_column=str(artifact.get("target_column") or ""),
        dataset_rows=int(len(df)),
        holdout=holdout,
        walk_forward=walk_forward,
        criteria=criteria,
    )


def _score_holdout(df: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    clean = df.copy()
    if "entry_timestamp" in clean:
        clean = clean.sort_values("entry_timestamp")
    clean = _engineer_features(clean)
    test_rows = int(artifact.get("test_rows") or 0)
    split = len(clean) - test_rows if 0 < test_rows < len(clean) else _split_index(len(clean), 0.25)
    holdout = clean.iloc[split:].copy()
    feature_columns = list(artifact["feature_columns"])
    frame = _transform_xgb_frame(holdout, feature_columns, dict(artifact.get("fill_values") or {}))
    booster = xgb.Booster()
    booster.load_model(str(artifact["model_path"]))
    raw_prediction = booster.predict(xgb.DMatrix(frame, feature_names=list(frame.columns)))
    holdout["prediction"] = _prediction_to_target_space(raw_prediction, artifact)
    return holdout


def _prediction_to_target_space(raw_prediction: np.ndarray, artifact: dict[str, Any]) -> np.ndarray:
    model_type = str(artifact.get("model_type") or "")
    loss_config = artifact.get("loss_config") or {}
    if "pseudohuber" not in model_type or not loss_config:
        return raw_prediction
    return _inverse_transform_target(raw_prediction, AsymmetricLossConfig(**loss_config))


def _selection_metrics(df: pd.DataFrame, prediction_column: str, config: ExitCriteriaConfig) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "selected_rows": 0}
    pred = pd.to_numeric(df[prediction_column], errors="coerce").fillna(-np.inf).to_numpy()
    selected_rows = max(1, int(np.ceil(len(df) * config.selection_fraction)))
    selected_index = np.argsort(pred)[-selected_rows:]
    selected = df.iloc[selected_index].copy()
    if "entry_timestamp" in selected:
        selected = selected.sort_values("entry_timestamp")
    pnl = pd.to_numeric(selected["expected_pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    slippage_adjusted_pnl = _slippage_adjusted_pnl(selected, config)
    all_pnl = pd.to_numeric(df["expected_pnl"], errors="coerce").dropna().to_numpy(dtype=float)
    gross_profit = float(np.sum(pnl[pnl > 0]))
    drawdown = _max_drawdown(pnl) or 0.0
    max_adverse_excursion = _column_max(selected, "max_adverse_excursion")
    underlying_counts = selected["underlying"].value_counts() if "underlying" in selected else pd.Series(dtype=int)
    top5_share = float(underlying_counts.head(5).sum() / len(selected)) if len(selected) else 0.0
    top1_share = float(underlying_counts.iloc[0] / len(selected)) if len(underlying_counts) else 0.0
    return {
        "rows": int(len(df)),
        "selected_rows": int(len(selected)),
        "selected_entry_dates": _selected_entry_dates(selected),
        "baseline_mean_pnl": _mean(all_pnl),
        "mean_pnl": _mean(pnl),
        "slippage_penalty_fraction": config.slippage_penalty_fraction,
        "slippage_adjusted_mean_pnl": _mean(slippage_adjusted_pnl),
        "slippage_adjusted_profit_factor": _profit_factor(slippage_adjusted_pnl),
        "slippage_adjusted_total_pnl": round(float(np.sum(slippage_adjusted_pnl)), 6) if len(slippage_adjusted_pnl) else None,
        "median_pnl": _percentile(pnl, 50),
        "total_pnl": round(float(np.sum(pnl)), 6) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "win_rate": _rate(pnl > 0),
        "p05_pnl": _percentile(pnl, 5),
        "p01_pnl": _percentile(pnl, 1),
        "worst_pnl": _min(pnl),
        "max_drawdown": drawdown,
        "max_adverse_excursion": max_adverse_excursion,
        "catastrophic_account_limit": config.catastrophic_account_limit,
        "catastrophic_limit_half": round(float(config.catastrophic_account_limit * 0.5), 6),
        "drawdown_to_gross_profit": round(drawdown / gross_profit, 6) if gross_profit > 0 else None,
        "mean_return_on_risk": _column_mean(selected, "return_on_risk"),
        "large_loss_rate": _column_mean(selected, "large_loss_label"),
        "stop_loss_rate": _column_mean(selected, "stop_loss_hit"),
        "single_underlying_share": round(top1_share, 6),
        "top5_underlying_share": round(top5_share, 6),
        "strategy_counts": _counts(selected, "strategy"),
        "top_underlying_counts": _counts(selected, "underlying", limit=10),
    }


def _walk_forward_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    folds = list(artifact.get("walk_forward") or [])
    top_mean_values = _fold_metric_values(folds, "top_decile_actual_mean")
    top_pf_values = _fold_metric_values(folds, "top_decile_profit_factor")
    top_win_values = _fold_metric_values(folds, "top_decile_win_rate")
    top_p05_values = _fold_metric_values(folds, "top_decile_tail_loss_p05")
    top_worst_values = _fold_metric_values(folds, "top_decile_worst_actual")
    top_drawdown_values = _fold_metric_values(folds, "top_decile_max_drawdown")
    top_mae_values = _fold_metric_values(folds, "top_decile_max_adverse_excursion")
    feature_stability = _feature_stability(folds)
    return {
        "folds": int(len(folds)),
        "test_rows": int(sum(int(fold.get("test_rows") or 0) for fold in folds)),
        "top_mean_pnl_min": _min(top_mean_values),
        "top_mean_pnl_mean": _mean(top_mean_values),
        "top_profit_factor_min": _min(top_pf_values),
        "top_profit_factor_mean": _mean(top_pf_values),
        "top_win_rate_min": _min(top_win_values),
        "top_win_rate_mean": _mean(top_win_values),
        "top_p05_pnl_min": _min(top_p05_values),
        "top_worst_pnl_min": _min(top_worst_values),
        "top_max_drawdown_max": _max(top_drawdown_values),
        "top_max_adverse_excursion_max": _max(top_mae_values),
        "feature_stability": feature_stability,
        "folds_detail": [
            {
                "fold": fold.get("fold"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
                "top_mean_pnl": (fold.get("metrics") or {}).get("top_decile_actual_mean"),
                "top_profit_factor": (fold.get("metrics") or {}).get("top_decile_profit_factor"),
                "top_win_rate": (fold.get("metrics") or {}).get("top_decile_win_rate"),
                "top_p05_pnl": (fold.get("metrics") or {}).get("top_decile_tail_loss_p05"),
                "top_worst_pnl": (fold.get("metrics") or {}).get("top_decile_worst_actual"),
                "top_max_drawdown": (fold.get("metrics") or {}).get("top_decile_max_drawdown"),
                "top_max_adverse_excursion": (fold.get("metrics") or {}).get("top_decile_max_adverse_excursion"),
                "top_features": _top_features(fold.get("feature_importance") or {}, 3),
            }
            for fold in folds
        ],
    }


def _evaluate_gates(
    df: pd.DataFrame,
    artifact: dict[str, Any],
    holdout: dict[str, Any],
    walk_forward: dict[str, Any],
    cfg: ExitCriteriaConfig,
) -> list[CriterionResult]:
    metrics = dict(artifact.get("metrics") or {})
    train_top_mean = _float_or_none(metrics.get("train_top_decile_actual_mean"))
    holdout_top_mean = _float_or_none(holdout.get("mean_pnl"))
    train_pf = _float_or_none(metrics.get("train_top_decile_profit_factor"))
    holdout_pf = _float_or_none(holdout.get("profit_factor"))
    catastrophic_half = cfg.catastrophic_account_limit * cfg.max_drawdown_to_catastrophic_limit
    mae_half = cfg.catastrophic_account_limit * cfg.max_mae_to_catastrophic_limit
    feature_stability = walk_forward.get("feature_stability") or {}
    return [
        _gte("dataset_rows", len(df), cfg.min_dataset_rows),
        _gte("holdout_rows", holdout.get("rows"), cfg.min_test_rows),
        _gte("holdout_selected_rows", holdout.get("selected_rows"), cfg.min_top_selection_rows),
        _gte("holdout_selected_entry_dates", holdout.get("selected_entry_dates"), cfg.min_top_selection_entry_dates),
        _gte("walk_forward_folds", walk_forward.get("folds"), cfg.min_walk_forward_folds),
        _lte("holdout_single_underlying_share", holdout.get("single_underlying_share"), cfg.max_single_underlying_share),
        _lte("holdout_top5_underlying_share", holdout.get("top5_underlying_share"), cfg.max_top5_underlying_share),
        _gte("holdout_top_mean_pnl", holdout.get("mean_pnl"), cfg.min_holdout_top_mean_pnl),
        _gte(
            "holdout_slippage_adjusted_mean_pnl",
            holdout.get("slippage_adjusted_mean_pnl"),
            cfg.min_holdout_top_slippage_adjusted_mean_pnl,
        ),
        _gte(
            "holdout_slippage_adjusted_profit_factor",
            holdout.get("slippage_adjusted_profit_factor"),
            cfg.min_holdout_top_slippage_adjusted_profit_factor,
        ),
        _gte("holdout_top_profit_factor", holdout.get("profit_factor"), cfg.min_holdout_top_profit_factor),
        _gte("holdout_top_win_rate", holdout.get("win_rate"), cfg.min_holdout_top_win_rate),
        _gte("holdout_top_return_on_risk", holdout.get("mean_return_on_risk"), cfg.min_holdout_top_return_on_risk),
        _gte("holdout_top_p05_pnl", holdout.get("p05_pnl"), cfg.min_holdout_top_p05_pnl),
        _gte("holdout_top_p01_pnl", holdout.get("p01_pnl"), cfg.min_holdout_top_p01_pnl),
        _gte("holdout_top_worst_pnl", holdout.get("worst_pnl"), cfg.min_holdout_top_worst_pnl),
        _lte("holdout_large_loss_rate", holdout.get("large_loss_rate"), cfg.max_holdout_top_large_loss_rate),
        _lte("holdout_stop_loss_rate", holdout.get("stop_loss_rate"), cfg.max_holdout_top_stop_loss_rate),
        _lte("holdout_max_drawdown_to_half_catastrophic_limit", holdout.get("max_drawdown"), catastrophic_half),
        _lte("holdout_mae_to_half_catastrophic_limit", holdout.get("max_adverse_excursion"), mae_half),
        _lte(
            "holdout_drawdown_to_gross_profit",
            holdout.get("drawdown_to_gross_profit"),
            cfg.max_holdout_top_drawdown_to_gross_profit,
        ),
        _gte("walk_forward_min_top_mean_pnl", walk_forward.get("top_mean_pnl_min"), cfg.min_walk_forward_top_mean_pnl),
        _gte("walk_forward_avg_top_mean_pnl", walk_forward.get("top_mean_pnl_mean"), cfg.min_walk_forward_avg_top_mean_pnl),
        _gte("walk_forward_min_top_profit_factor", walk_forward.get("top_profit_factor_min"), cfg.min_walk_forward_top_profit_factor),
        _gte("walk_forward_avg_top_profit_factor", walk_forward.get("top_profit_factor_mean"), cfg.min_walk_forward_avg_top_profit_factor),
        _gte("walk_forward_min_top_win_rate", walk_forward.get("top_win_rate_min"), cfg.min_walk_forward_top_win_rate),
        _gte("walk_forward_min_top_p05_pnl", walk_forward.get("top_p05_pnl_min"), cfg.min_walk_forward_top_p05_pnl),
        _gte("walk_forward_min_top_worst_pnl", walk_forward.get("top_worst_pnl_min"), cfg.min_walk_forward_top_worst_pnl),
        _lte("walk_forward_max_drawdown_to_half_catastrophic_limit", walk_forward.get("top_max_drawdown_max"), catastrophic_half),
        _lte("walk_forward_mae_to_half_catastrophic_limit", walk_forward.get("top_max_adverse_excursion_max"), mae_half),
        _gte(
            "walk_forward_top_feature_overlap",
            feature_stability.get("top_feature_overlap"),
            cfg.min_feature_top_k_overlap,
        ),
        _lte(
            "train_holdout_top_mean_ratio",
            _positive_ratio(train_top_mean, holdout_top_mean),
            cfg.max_train_holdout_top_mean_ratio,
        ),
        _lte(
            "train_holdout_profit_factor_ratio",
            _positive_ratio(train_pf, holdout_pf),
            cfg.max_train_holdout_profit_factor_ratio,
        ),
    ]


def _automl_score(holdout: dict[str, Any], walk_forward: dict[str, Any], criteria: list[CriterionResult]) -> float:
    holdout_pf = _float_or_none(holdout.get("profit_factor")) or 0.0
    holdout_mean = _float_or_none(holdout.get("mean_pnl")) or 0.0
    walk_pf = _float_or_none(walk_forward.get("top_profit_factor_mean")) or 0.0
    walk_mean = _float_or_none(walk_forward.get("top_mean_pnl_mean")) or 0.0
    large_loss = _float_or_none(holdout.get("large_loss_rate")) or 1.0
    p05 = _float_or_none(holdout.get("p05_pnl")) or -10_000.0
    failed = sum(1 for item in criteria if not item.passed)
    slippage_mean = _float_or_none(holdout.get("slippage_adjusted_mean_pnl")) or 0.0
    raw = 20.0 * holdout_pf + 20.0 * walk_pf + 0.2 * holdout_mean + 0.3 * walk_mean + 0.2 * slippage_mean - 60.0 * large_loss + 0.02 * p05
    return round(float(raw - 10.0 * failed), 6)


def _gte(name: str, actual: Any, threshold: Any) -> CriterionResult:
    value = _float_or_none(actual)
    passed = value is not None and value >= float(threshold)
    return CriterionResult(name=name, passed=passed, actual=actual, threshold=threshold, direction=">=")


def _lte(name: str, actual: Any, threshold: Any) -> CriterionResult:
    value = _float_or_none(actual)
    passed = value is not None and value <= float(threshold)
    return CriterionResult(name=name, passed=passed, actual=actual, threshold=threshold, direction="<=")


def _fold_metric_values(folds: list[dict[str, Any]], name: str) -> np.ndarray:
    values = [_float_or_none((fold.get("metrics") or {}).get(name)) for fold in folds]
    return np.array([value for value in values if value is not None], dtype=float)


def _feature_stability(folds: list[dict[str, Any]]) -> dict[str, Any]:
    folds_with_importance = [fold for fold in folds if fold.get("feature_importance")]
    if len(folds_with_importance) < 2:
        return {
            "top_feature_overlap": None,
            "first_fold_top_features": [],
            "last_fold_top_features": [],
            "reason": "missing_fold_feature_importance",
        }
    first = _top_features(folds_with_importance[0].get("feature_importance") or {}, 3)
    last = _top_features(folds_with_importance[-1].get("feature_importance") or {}, 3)
    overlap = len(set(first) & set(last))
    return {
        "top_feature_overlap": int(overlap),
        "first_fold_top_features": first,
        "last_fold_top_features": last,
    }


def _top_features(feature_importance: dict[str, Any], count: int) -> list[str]:
    return [
        str(name)
        for name, _ in sorted(
            feature_importance.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )[:count]
    ]


def _positive_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0 or numerator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _report_to_dict(report: ExitCriteriaReport) -> dict[str, Any]:
    payload = asdict(report)
    payload["criteria"] = [asdict(item) for item in report.criteria]
    return payload


def _column_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _column_max(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    numeric = pd.to_numeric(df[column], errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.max()), 6)


def _slippage_adjusted_pnl(df: pd.DataFrame, config: ExitCriteriaConfig) -> np.ndarray:
    pnl = pd.to_numeric(df["expected_pnl"], errors="coerce").to_numpy(dtype=float)
    if "max_profit" in df:
        credit_dollars = pd.to_numeric(df["max_profit"], errors="coerce").to_numpy(dtype=float)
    elif "entry_credit" in df:
        credit_dollars = pd.to_numeric(df["entry_credit"], errors="coerce").to_numpy(dtype=float) * 100.0
    else:
        credit_dollars = np.zeros(len(df), dtype=float)
    credit_dollars = np.nan_to_num(credit_dollars, nan=0.0, posinf=0.0, neginf=0.0)
    adjusted = pnl - config.slippage_penalty_fraction * np.maximum(credit_dollars, 0.0)
    return adjusted[np.isfinite(adjusted)]


def _selected_entry_dates(df: pd.DataFrame) -> int | None:
    if "entry_timestamp" not in df:
        return None
    dates = pd.to_datetime(df["entry_timestamp"], errors="coerce").dropna().dt.date
    return int(dates.nunique())


def _counts(df: pd.DataFrame, column: str, *, limit: int | None = None) -> dict[str, int]:
    if column not in df:
        return {}
    counts = df[column].value_counts()
    if limit is not None:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def _mean(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.mean(values)), 6)


def _min(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.min(values)), 6)


def _max(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.max(values)), 6)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.percentile(values, percentile)), 6)


def _rate(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.mean(values)), 6)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
