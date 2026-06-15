"""Train a binary XGBoost classifier to predict large-loss trades.

Purpose
-------
The main RoR-target ranker (train_xgboost.py) selects high-expected-return
trades, but the top decile still carries tail risk — the five worst trades in
the current 590k-row corpus are all the same SOXX option with losses exceeding
$3,000/contract.  This classifier learns which trades are likely to hit the
``large_loss_label`` flag (realized PnL < −max_loss, i.e. a loss worse than
the max-loss boundary) and is used as a second-stage filter at inference time.

Inference pattern
-----------------
1. Score all candidate spreads with the RoR ranker.
2. Run each candidate through this classifier to obtain p(large_loss).
3. Only take trades where rank score is high AND p(large_loss) < threshold
   (e.g. 0.15).

Usage
-----
python -m ml.models.train_large_loss_classifier \\
    --input  artifacts/datasets/candidate_rows/... \\
    --output artifacts/models/large_loss_classifier_v001.json \\
    --test-fraction 0.25 \\
    --walk-forward-folds 3 \\
    --embargo-days 30
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.train_baseline import (
    _artifact_metadata,
    _artifact_fingerprint,
    _chronological_holdout_split,
    _engineer_features,
    _feature_importance,
    _gap_days,
    _select_feature_columns,
    _split_index,
    _walk_forward_splits_by_timestamp,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


TARGET_COLUMN = "large_loss_label"


@dataclass(frozen=True)
class LargeLossClassifierArtifact:
    model_type: str
    created_at: str
    target_column: str
    feature_version: str
    label_version: str
    data_range: dict[str, str | None]
    model_path: str
    feature_columns: list[str]
    fill_values: dict[str, float]
    train_rows: int
    test_rows: int
    train_positive_rate: float
    test_positive_rate: float
    dataset: dict[str, Any]
    training_command: str | None
    data_fingerprint: str
    data_quality_filters: dict[str, Any]
    split_summary: dict[str, Any]
    params: dict[str, Any]
    feature_importance: dict[str, float]
    metrics: dict[str, Any]
    walk_forward: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a binary classifier to predict large-loss credit spread trades."
    )
    parser.add_argument("--input", required=True, help="Parquet dataset directory or file.")
    parser.add_argument("--output", required=True, help="Output artifact JSON path.")
    parser.add_argument("--model-output", default=None, help="XGBoost model output path.")
    parser.add_argument("--target", default=TARGET_COLUMN, choices=["large_loss_label", "stop_loss_hit"])
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--min-walk-forward-train-rows", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=30)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    parser.add_argument("--reg-lambda", type=float, default=20.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help=(
            "XGBoost scale_pos_weight to rebalance the 12.5%% large-loss minority class. "
            "Defaults to (negative_count / positive_count) from training data."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Probability threshold for classification metrics (recall, precision, F1). "
             "Should match the veto threshold used at inference time.",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help=(
            "Comma-separated feature names to drop before training. "
            "Used by the feature-selection optimizer to mask feature groups. "
            "Example: --exclude-features vix_regime,market_return_5d"
        ),
    )
    parser.add_argument(
        "--max-dte",
        type=int,
        default=None,
        help="Maximum DTE filter: drop rows where dte > this value before training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    model_output = (
        Path(args.model_output)
        if args.model_output
        else output.with_name(f"{output.stem}.xgboost.json")
    )
    df = load_dataset(Path(args.input))
    if args.max_dte is not None and "dte" in df.columns:
        dte_col = pd.to_numeric(df["dte"], errors="coerce")
        pre = len(df)
        df = df[dte_col <= args.max_dte].reset_index(drop=True)
        print(f"DTE filter: {pre:,} -> {len(df):,} rows (max_dte={args.max_dte})", file=sys.stderr)
    exclude_features = {f.strip() for f in args.exclude_features.split(",") if f.strip()}
    artifact = train_large_loss_classifier(
        df,
        model_output=model_output,
        dataset_path=str(args.input),
        training_command=" ".join(shlex.quote(part) for part in sys.argv),
        target_column=args.target,
        test_fraction=args.test_fraction,
        walk_forward_folds=args.walk_forward_folds,
        min_walk_forward_train_rows=args.min_walk_forward_train_rows,
        embargo_days=args.embargo_days,
        num_boost_round=args.num_boost_round,
        val_fraction=args.val_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
        scale_pos_weight=args.scale_pos_weight,
        exclude_features=exclude_features,
        threshold=args.threshold,
        params={
            "max_depth": args.max_depth,
            "eta": args.eta,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "lambda": args.reg_lambda,
            "alpha": args.reg_alpha,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(artifact), indent=2, sort_keys=True))
    return 0


def train_large_loss_classifier(
    df: pd.DataFrame,
    *,
    model_output: Path,
    dataset_path: str | None = None,
    training_command: str | None = None,
    target_column: str = TARGET_COLUMN,
    test_fraction: float = 0.25,
    min_rows: int = 20,
    walk_forward_folds: int = 3,
    min_walk_forward_train_rows: int | None = None,
    embargo_days: int = 30,
    val_fraction: float = 0.15,
    early_stopping_rounds: int = 20,
    num_boost_round: int = 200,
    scale_pos_weight: float | None = None,
    exclude_features: set[str] | None = None,
    threshold: float = 0.15,
    params: dict[str, Any] | None = None,
) -> LargeLossClassifierArtifact:
    """Train and evaluate a binary large-loss classifier.

    Args:
        df: Candidate dataset with ``large_loss_label`` column.
        model_output: Where to save the XGBoost booster.
        test_fraction: Fraction of rows held out as chronological test set.
        walk_forward_folds: Number of walk-forward CV folds.
        embargo_days: Calendar days excluded between train and test in each fold.
        val_fraction: Fraction of training data used for early stopping.
        early_stopping_rounds: Patience for early stopping (0 = disabled).
        num_boost_round: Maximum boosting rounds.
        scale_pos_weight: Class weight for positive (large-loss) class.
            Defaults to (negatives / positives) to handle the 12.5% prevalence.
        params: XGBoost hyperparameter overrides.

    Returns:
        A ``LargeLossClassifierArtifact`` with metrics and walk-forward results.
    """
    if xgb is None:
        raise ImportError("xgboost is required. Install xgboost and its native runtime dependencies.")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column '{target_column}' in dataset.")

    clean = df.copy()
    clean[target_column] = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.dropna(subset=[target_column])
    if len(clean) < min_rows:
        raise ValueError(f"Need at least {min_rows} labeled rows, found {len(clean)}")
    if "entry_timestamp" in clean.columns:
        clean = clean.sort_values("entry_timestamp")

    clean = _engineer_features(clean)
    feature_columns = _select_feature_columns(clean)
    if exclude_features:
        feature_columns = [f for f in feature_columns if f not in exclude_features]
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    train_df, test_df, split_summary = _chronological_holdout_split(
        clean,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
    )
    y_train = train_df[target_column].to_numpy(dtype=float)
    y_test = test_df[target_column].to_numpy(dtype=float)

    # Auto-compute scale_pos_weight from training data if not provided.
    neg = float(np.sum(y_train == 0))
    pos = float(np.sum(y_train == 1))
    spw = scale_pos_weight if scale_pos_weight is not None else (neg / pos if pos > 0 else 1.0)

    model_params = _default_params(spw)
    if params:
        model_params.update(params)

    fill_values = _compute_fill_values(train_df, feature_columns)
    train_count = len(train_df)
    val_split_index = _split_index(train_count, val_fraction) if val_fraction > 0 and train_count > 1 else train_count

    x_train_sub = _build_dmatrix(train_df.iloc[:val_split_index], feature_columns, fill_values, y_train[:val_split_index])
    x_val = _build_dmatrix(train_df.iloc[val_split_index:], feature_columns, fill_values, y_train[val_split_index:])
    x_train_full = _build_dmatrix(train_df, feature_columns, fill_values, y_train)
    x_test_frame = _transform_frame(test_df, feature_columns, fill_values)

    has_val = x_val.num_row() > 0 and early_stopping_rounds > 0
    booster = xgb.train(
        model_params,
        x_train_sub,
        num_boost_round=num_boost_round,
        evals=[(x_val, "val")] if has_val else [],
        early_stopping_rounds=early_stopping_rounds if has_val else None,
        verbose_eval=False,
    )
    best_rounds = getattr(booster, "best_iteration", num_boost_round - 1) + 1

    train_prob = _predict_prob(booster, _transform_frame(train_df, feature_columns, fill_values))
    test_prob = _predict_prob(booster, x_test_frame) if len(x_test_frame) > 0 else np.array([])

    metrics = _prefixed_clf_metrics("train", y_train, train_prob, threshold)
    if len(test_prob):
        metrics.update(_prefixed_clf_metrics("test", y_test, test_prob, threshold))

    walk_forward = _walk_forward_clf(
        clean,
        target_column=target_column,
        feature_columns=feature_columns,
        fold_count=walk_forward_folds,
        min_train_rows=min_walk_forward_train_rows or max(min_rows, len(train_df)),
        params=model_params,
        num_boost_round=best_rounds,
        embargo_days=embargo_days,
        threshold=threshold,
    )
    metrics.update(_walk_forward_summary_clf(walk_forward))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_output)

    artifact_metadata = _artifact_metadata(clean)
    dataset_info = dict(artifact_metadata["dataset"])
    if dataset_path:
        dataset_info["input_path"] = dataset_path
    return LargeLossClassifierArtifact(
        model_type="xgboost_binary_large_loss_v001" if target_column == TARGET_COLUMN else "xgboost_binary_risk_v001",
        created_at=datetime.now(UTC).isoformat(),
        target_column=target_column,
        feature_version=artifact_metadata["feature_version"],
        label_version=artifact_metadata["label_version"],
        data_range=artifact_metadata["data_range"],
        model_path=str(model_output),
        feature_columns=feature_columns,
        fill_values=fill_values,
        train_rows=int(len(y_train)),
        test_rows=int(len(y_test)),
        train_positive_rate=round(float(np.mean(y_train)), 6),
        test_positive_rate=round(float(np.mean(y_test)), 6) if len(y_test) else 0.0,
        dataset=dataset_info,
        training_command=training_command,
        data_fingerprint=_artifact_fingerprint(
            dataset=dataset_info,
            target_column=target_column,
            feature_columns=feature_columns,
            data_quality_filters=artifact_metadata["data_quality_filters"],
            split_summary=split_summary,
        ),
        data_quality_filters=artifact_metadata["data_quality_filters"],
        split_summary=split_summary,
        params={**model_params, "num_boost_round": int(best_rounds)},
        feature_importance=_feature_importance(booster),
        metrics=metrics,
        walk_forward=walk_forward,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_params(scale_pos_weight: float = 1.0) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 20.0,
        "lambda": 20.0,
        "alpha": 0.0,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "seed": 17,
    }


def _compute_fill_values(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    return {
        col: round(float(frame[col].median()) if frame[col].notna().any() else 0.0, 10)
        for col in feature_columns
    }


def _transform_frame(df: pd.DataFrame, feature_columns: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    return frame.astype(float)


def _build_dmatrix(df: pd.DataFrame, feature_columns: list[str], fill_values: dict[str, float], labels: np.ndarray):
    frame = _transform_frame(df, feature_columns, fill_values)
    return xgb.DMatrix(frame, label=labels, feature_names=feature_columns)


def _predict_prob(booster, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([])
    matrix = xgb.DMatrix(frame, feature_names=list(frame.columns))
    return booster.predict(matrix)


def _clf_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.15) -> dict[str, Any]:
    """Classification metrics at a given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    auc = _auc_rank(y_true, y_prob)

    return {
        "rows": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 6),
        "threshold": threshold,
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
        "auc": round(auc, 6) if auc is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "mean_prob_positive": round(float(np.mean(y_prob[y_true == 1])), 6) if np.any(y_true == 1) else None,
        "mean_prob_negative": round(float(np.mean(y_prob[y_true == 0])), 6) if np.any(y_true == 0) else None,
    }


def _prefixed_clf_metrics(prefix: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.15) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in _clf_metrics(y_true, y_prob, threshold).items()}


def _walk_forward_clf(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    fold_count: int,
    min_train_rows: int,
    params: dict[str, Any],
    num_boost_round: int,
    embargo_days: int,
    threshold: float = 0.15,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    y_all = df[target_column].to_numpy(dtype=float)

    for fold_number, train_start, train_end, test_start, test_end in _walk_forward_splits_by_timestamp(
        df,
        fold_count=fold_count,
        min_train_rows=min_train_rows,
        embargo_days=embargo_days,
    ):
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        fold_fill = _compute_fill_values(train_df, feature_columns)

        # Use the optimized scale_pos_weight from params unchanged so walk-forward
        # metrics reflect the same model configuration that will be deployed.
        y_fold_train = y_all[train_start:train_end]
        dtrain = _build_dmatrix(train_df, feature_columns, fold_fill, y_fold_train)
        booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=False)

        x_test_frame = _transform_frame(test_df, feature_columns, fold_fill)
        prob = _predict_prob(booster, x_test_frame)
        fold_metrics = _clf_metrics(y_all[test_start:test_end], prob, threshold)

        from ml.models.train_baseline import _row_timestamp
        folds.append({
            "fold": fold_number,
            "train_start": _row_timestamp(df, train_start),
            "train_end": _row_timestamp(df, train_end - 1),
            "embargo_rows": int(test_start - train_end),
            "actual_gap_days": _gap_days(df, train_end - 1, test_start),
            "test_start": _row_timestamp(df, test_start),
            "test_end": _row_timestamp(df, test_end - 1),
            "train_rows": int(train_end - train_start),
            "test_rows": int(test_end - test_start),
            "metrics": fold_metrics,
        })
    return folds


def _walk_forward_summary_clf(folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"walk_forward_folds": int(len(folds))}
    if not folds:
        return summary
    for key in ("auc", "precision", "recall", "f1"):
        vals = [fold["metrics"].get(key) for fold in folds]
        numeric = [float(v) for v in vals if isinstance(v, (int, float))]
        summary[f"walk_forward_{key}_mean"] = round(float(np.mean(numeric)), 6) if numeric else None
    summary["walk_forward_test_rows"] = int(sum(f["test_rows"] for f in folds))
    return summary


def _auc_rank(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Return ROC AUC using average ranks, O(n log n) memory-safe for large corpora."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(y_prob).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


if __name__ == "__main__":
    raise SystemExit(main())
