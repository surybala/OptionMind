"""Train an asymmetric XGBoost model on candidate dataset rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.train_baseline import (
    _embargo_rows_from_days,
    _empty_test_metrics,
    _evaluation_metrics,
    _prefixed_metrics,
    _row_timestamp,
    _select_feature_columns,
    _split_index,
    _walk_forward_splits,
    _walk_forward_summary,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - exercised only in missing native dependency environments
    xgb = None


@dataclass(frozen=True)
class AsymmetricLossConfig:
    downside_scale: float = 1000.0
    error_scale: float = 1000.0
    downside_penalty: float = 1.5
    overprediction_penalty: float = 1.0
    max_multiplier: float = 30.0


@dataclass(frozen=True)
class XGBoostModelArtifact:
    model_type: str
    created_at: str
    target_column: str
    model_path: str
    feature_columns: list[str]
    fill_values: dict[str, float]
    train_rows: int
    test_rows: int
    params: dict[str, Any]
    loss_config: dict[str, float]
    feature_importance: dict[str, float]
    metrics: dict[str, Any]
    walk_forward: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an asymmetric XGBoost model.")
    parser.add_argument("--input", required=True, help="JSONL file, parquet file, or dataset directory.")
    parser.add_argument("--output", required=True, help="Output artifact JSON path.")
    parser.add_argument("--model-output", default=None, help="Optional XGBoost model output path.")
    parser.add_argument("--target", default="expected_pnl")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--min-walk-forward-train-rows", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=30, help="Calendar days to exclude between train and test in each walk-forward fold (should match forward_days).")
    parser.add_argument("--early-stopping-rounds", type=int, default=0, help="Stop training when val mae has not improved for this many rounds. 0 disables early stopping.")
    parser.add_argument("--val-fraction", type=float, default=0.0, help="Fraction of training data held out as a validation set for early stopping. 0 disables.")
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=5.0)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--downside-scale", type=float, default=1000.0)
    parser.add_argument("--error-scale", type=float, default=1000.0)
    parser.add_argument("--downside-penalty", type=float, default=1.5)
    parser.add_argument("--overprediction-penalty", type=float, default=1.0)
    parser.add_argument("--max-multiplier", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    model_output = Path(args.model_output) if args.model_output else output.with_name(f"{output.stem}.xgboost.json")
    df = load_dataset(Path(args.input))
    artifact = train_xgboost(
        df,
        model_output=model_output,
        target_column=args.target,
        test_fraction=args.test_fraction,
        min_rows=args.min_rows,
        walk_forward_folds=args.walk_forward_folds,
        min_walk_forward_train_rows=args.min_walk_forward_train_rows,
        embargo_days=args.embargo_days,
        val_fraction=args.val_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
        params=_params_from_args(args),
        num_boost_round=args.num_boost_round,
        loss_config=AsymmetricLossConfig(
            downside_scale=args.downside_scale,
            error_scale=args.error_scale,
            downside_penalty=args.downside_penalty,
            overprediction_penalty=args.overprediction_penalty,
            max_multiplier=args.max_multiplier,
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(artifact), indent=2, sort_keys=True))
    return 0


def train_xgboost(
    df: pd.DataFrame,
    *,
    model_output: Path,
    target_column: str = "expected_pnl",
    test_fraction: float = 0.25,
    min_rows: int = 20,
    walk_forward_folds: int = 3,
    min_walk_forward_train_rows: int | None = None,
    embargo_days: int = 0,
    val_fraction: float = 0.15,
    early_stopping_rounds: int = 20,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 300,
    loss_config: AsymmetricLossConfig | None = None,
) -> XGBoostModelArtifact:
    if xgb is None:
        raise ImportError("xgboost is required and must be loadable. Install xgboost and its native runtime dependencies.")
    if target_column not in df:
        raise ValueError(f"Missing target column: {target_column}")

    clean = df.copy()
    clean[target_column] = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.dropna(subset=[target_column])
    if len(clean) < min_rows:
        raise ValueError(f"Need at least {min_rows} labeled rows, found {len(clean)}")
    if "entry_timestamp" in clean:
        clean = clean.sort_values("entry_timestamp")

    feature_columns = _select_feature_columns(clean)
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found")

    y_all = clean[target_column].to_numpy(dtype=float)
    split_index = _split_index(len(clean), test_fraction)

    # Hold out the last val_fraction of training rows as a validation set for
    # early stopping. Fill values are always derived from the full training set.
    _, fill_values = _fit_xgb_frame(clean.iloc[:split_index], feature_columns)
    val_split_index = _split_index(split_index, val_fraction) if val_fraction > 0 and split_index > 1 else split_index
    x_train_sub = _transform_xgb_frame(clean.iloc[:val_split_index], feature_columns, fill_values)
    x_val = _transform_xgb_frame(clean.iloc[val_split_index:split_index], feature_columns, fill_values)
    x_train = _transform_xgb_frame(clean.iloc[:split_index], feature_columns, fill_values)
    x_test = _transform_xgb_frame(clean.iloc[split_index:], feature_columns, fill_values)

    y_train_sub = y_all[:val_split_index]
    y_val = y_all[val_split_index:split_index]
    y_train = y_all[:split_index]
    y_test = y_all[split_index:]

    model_params = _default_params()
    if params:
        model_params.update(params)
    loss = loss_config or AsymmetricLossConfig()
    booster = _fit_booster(
        x_train_sub, y_train_sub, model_params, num_boost_round, loss,
        x_val=x_val if len(x_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
        early_stopping_rounds=early_stopping_rounds,
    )
    best_iteration = getattr(booster, "best_iteration", num_boost_round - 1) + 1
    train_pred = _predict(booster, x_train)
    test_pred = _predict(booster, x_test) if len(x_test) else np.array([])

    metrics = _prefixed_metrics("train", _evaluation_metrics(y_train, train_pred))
    if len(x_test):
        metrics.update(_prefixed_metrics("test", _evaluation_metrics(y_test, test_pred)))
    else:
        metrics.update(_empty_test_metrics("test"))

    walk_forward = _walk_forward_validation(
        clean,
        target_column=target_column,
        feature_columns=feature_columns,
        fold_count=walk_forward_folds,
        min_train_rows=min_walk_forward_train_rows or max(min_rows, split_index),
        params=model_params,
        num_boost_round=best_iteration,
        loss_config=loss,
        embargo_days=embargo_days,
    )
    metrics.update(_walk_forward_summary(walk_forward))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_output)
    return XGBoostModelArtifact(
        model_type="xgboost_asymmetric_v001",
        created_at=datetime.now(UTC).isoformat(),
        target_column=target_column,
        model_path=str(model_output),
        feature_columns=feature_columns,
        fill_values=fill_values,
        train_rows=int(len(y_train)),
        test_rows=int(len(y_test)),
        params={**model_params, "num_boost_round": int(best_iteration)},
        loss_config=asdict(loss),
        feature_importance=_feature_importance(booster),
        metrics=metrics,
        walk_forward=walk_forward,
    )


def _params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "lambda": args.reg_lambda,
        "alpha": args.reg_alpha,
    }


def _default_params() -> dict[str, Any]:
    return {
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5.0,
        "lambda": 10.0,
        "alpha": 0.0,
        "disable_default_eval_metric": 1,
        "seed": 17,
    }


def _fit_booster(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any],
    num_boost_round: int,
    loss_config: AsymmetricLossConfig,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    early_stopping_rounds: int | None = None,
):
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=list(x_train.columns))
    has_val = x_val is not None and y_val is not None and len(x_val) > 0 and early_stopping_rounds
    evals = []
    train_params = dict(params)
    if has_val:
        dval = xgb.DMatrix(x_val, label=y_val, feature_names=list(x_val.columns))
        evals = [(dval, "val")]
        # mae is more stable than rmse with the asymmetric custom objective because
        # asymmetric gradients push predictions negative, inflating squared errors
        # even when the model is improving on the actual loss.
        train_params["eval_metric"] = "mae"
    return xgb.train(
        train_params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=_asymmetric_objective(loss_config),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if has_val else None,
        verbose_eval=False,
    )


def _asymmetric_objective(config: AsymmetricLossConfig):
    def objective(predt: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        residual = predt - labels
        downside = np.clip((-labels) / config.downside_scale, 0.0, None)
        optimistic_error = np.clip(residual / config.error_scale, 0.0, None)
        exponent = config.downside_penalty * downside + config.overprediction_penalty * optimistic_error
        multiplier = np.where(residual > 0.0, np.exp(np.clip(exponent, 0.0, np.log(config.max_multiplier))), 1.0)
        grad = multiplier * residual
        hess = multiplier * (1.0 + np.where(residual > 0.0, config.overprediction_penalty * optimistic_error, 0.0))
        return grad.astype(float), np.maximum(hess, 1e-6).astype(float)

    return objective


def _fit_xgb_frame(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = {
        column: round(float(frame[column].median()) if frame[column].notna().any() else 0.0, 10)
        for column in feature_columns
    }
    return _transform_xgb_frame(df, feature_columns, fill_values), fill_values


def _transform_xgb_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> pd.DataFrame:
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)


def _predict(booster, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([])
    matrix = xgb.DMatrix(frame, feature_names=list(frame.columns))
    return booster.predict(matrix)


def _walk_forward_validation(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    fold_count: int,
    min_train_rows: int,
    params: dict[str, Any],
    num_boost_round: int,
    loss_config: AsymmetricLossConfig,
    embargo_days: int = 0,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    y_all = df[target_column].to_numpy(dtype=float)
    embargo_rows = _embargo_rows_from_days(df, embargo_days)
    for fold_number, train_start, train_end, test_start, test_end in _walk_forward_splits(
        len(df),
        fold_count=fold_count,
        min_train_rows=min_train_rows,
        embargo_rows=embargo_rows,
    ):
        x_train, fold_fill_values = _fit_xgb_frame(df.iloc[train_start:train_end], feature_columns)
        x_test = _transform_xgb_frame(df.iloc[test_start:test_end], feature_columns, fold_fill_values)
        booster = _fit_booster(
            x_train,
            y_all[train_start:train_end],
            params,
            num_boost_round,
            loss_config,
        )
        pred = _predict(booster, x_test)
        fold_metrics = _evaluation_metrics(y_all[test_start:test_end], pred)
        folds.append(
            {
                "fold": fold_number,
                "train_start": _row_timestamp(df, train_start),
                "train_end": _row_timestamp(df, train_end - 1),
                "embargo_rows": int(test_start - train_end),
                "test_start": _row_timestamp(df, test_start),
                "test_end": _row_timestamp(df, test_end - 1),
                "train_rows": int(train_end - train_start),
                "test_rows": int(test_end - test_start),
                "metrics": fold_metrics,
            }
        )
    return folds


def _feature_importance(booster) -> dict[str, float]:
    scores = booster.get_score(importance_type="gain")
    return {
        key: round(float(value), 6)
        for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    }


if __name__ == "__main__":
    raise SystemExit(main())
