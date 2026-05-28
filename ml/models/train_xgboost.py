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
    _artifact_metadata,
    _embargo_rows_from_days,
    _empty_test_metrics,
    _engineer_features,
    _evaluation_metrics,
    _max_drawdown,
    _prefixed_metrics,
    _profit_factor,
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
    downside_penalty: float = 1.4
    overprediction_penalty: float = 1.0
    max_multiplier: float = 30.0
    target_scale: float = 100.0
    target_clip: float = 5000.0
    huber_delta: float = 1.0
    gradient_clip: float = 10.0
    hessian_floor: float = 1e-6


@dataclass(frozen=True)
class XGBoostModelArtifact:
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
    parser.add_argument("--target", default="return_on_risk")
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
    parser.add_argument("--target-scale", type=float, default=0.10, help="Signed log1p target scaling denominator for the custom objective. Use 0.10 for return_on_risk (0–1 range), 100.0 for expected_pnl (dollar range).")
    parser.add_argument("--target-clip", type=float, default=5.0, help="Absolute target cap before signed log1p scaling. Use 5.0 for return_on_risk, 5000.0 for expected_pnl.")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Pseudo-Huber transition point in transformed target units.")
    parser.add_argument("--gradient-clip", type=float, default=10.0, help="Absolute custom-objective gradient clip.")
    parser.add_argument("--hessian-floor", type=float, default=1e-6, help="Minimum custom-objective hessian.")
    parser.add_argument(
        "--max-rows-per-underlying",
        type=int,
        default=None,
        help=(
            "Hard cap on training rows per underlying symbol before the train/test split. "
            "Prevents any single ETF from dominating the training set (e.g. SMH at 22%%). "
            "Set to e.g. 30000 for a roughly uniform distribution across 39 underlyings."
        ),
    )
    parser.add_argument(
        "--high-vol-oversample-factor",
        type=int,
        default=1,
        help=(
            "Replicate high-volatility training rows (vix_regime >= 2, i.e. VIX >= 30) "
            "this many additional times. Factor=5 means high-vol rows appear 5x in training. "
            "Applied only to the training split to avoid contaminating the test evaluation. "
            "Interim measure until 2022-2023 high-vol data is added to the corpus."
        ),
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help=(
            "Comma-separated feature names to drop before training. "
            "Used by the feature-selection optimizer to mask feature groups. "
            "Example: --exclude-features vix_regime,vix_return_5d,option_gamma"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    model_output = Path(args.model_output) if args.model_output else output.with_name(f"{output.stem}.xgboost.json")
    df = load_dataset(Path(args.input))
    exclude_features = {f.strip() for f in args.exclude_features.split(",") if f.strip()}
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
        max_rows_per_underlying=args.max_rows_per_underlying,
        high_vol_oversample_factor=args.high_vol_oversample_factor,
        exclude_features=exclude_features,
        loss_config=AsymmetricLossConfig(
            downside_scale=args.downside_scale,
            error_scale=args.error_scale,
            downside_penalty=args.downside_penalty,
            overprediction_penalty=args.overprediction_penalty,
            max_multiplier=args.max_multiplier,
            target_scale=args.target_scale,
            target_clip=args.target_clip,
            huber_delta=args.huber_delta,
            gradient_clip=args.gradient_clip,
            hessian_floor=args.hessian_floor,
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
    target_column: str = "return_on_risk",
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
    max_rows_per_underlying: int | None = None,
    high_vol_oversample_factor: int = 1,
    exclude_features: set[str] | None = None,
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

    # Per-underlying row cap: prevents any single ETF from dominating training.
    # Applied before train/test split so the holdout reflects the capped distribution.
    if max_rows_per_underlying is not None and "underlying" in clean.columns:
        clean = _cap_rows_per_underlying(clean, max_rows_per_underlying)

    clean = _engineer_features(clean)
    feature_columns = _select_feature_columns(clean)
    if exclude_features:
        feature_columns = [f for f in feature_columns if f not in exclude_features]
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found")

    y_all = clean[target_column].to_numpy(dtype=float)
    loss = loss_config or AsymmetricLossConfig()
    y_model_all = _transform_target(y_all, loss)
    split_index = _split_index(len(clean), test_fraction)

    # Hold out the last val_fraction of training rows as a validation set for
    # early stopping. Fill values are always derived from the full training set.
    _, fill_values = _fit_xgb_frame(clean.iloc[:split_index], feature_columns)
    val_split_index = _split_index(split_index, val_fraction) if val_fraction > 0 and split_index > 1 else split_index

    # High-vol oversampling: replicate vix_regime >= 2 rows in the training split
    # only. The test split is never oversampled so evaluation reflects real distribution.
    train_df_sub = clean.iloc[:val_split_index]
    if high_vol_oversample_factor > 1:
        train_df_sub = _oversample_high_vol(train_df_sub, high_vol_oversample_factor)
        y_sub_oversampled = train_df_sub[target_column].to_numpy(dtype=float)
        y_train_sub = _transform_target(y_sub_oversampled, loss)
    else:
        y_train_sub = y_model_all[:val_split_index]

    x_train_sub = _transform_xgb_frame(train_df_sub, feature_columns, fill_values)
    x_val = _transform_xgb_frame(clean.iloc[val_split_index:split_index], feature_columns, fill_values)
    x_train = _transform_xgb_frame(clean.iloc[:split_index], feature_columns, fill_values)
    x_test = _transform_xgb_frame(clean.iloc[split_index:], feature_columns, fill_values)

    y_val = y_model_all[val_split_index:split_index]
    y_train = y_all[:split_index]
    y_test = y_all[split_index:]

    model_params = _default_params()
    if params:
        model_params.update(params)
    booster = _fit_booster(
        x_train_sub, y_train_sub, model_params, num_boost_round, loss,
        x_val=x_val if len(x_val) > 0 else None,
        y_val=y_val if len(y_val) > 0 else None,
        early_stopping_rounds=early_stopping_rounds,
    )
    best_iteration = getattr(booster, "best_iteration", num_boost_round - 1) + 1
    train_pred = _inverse_transform_target(_predict_model(booster, x_train), loss)
    test_pred = _inverse_transform_target(_predict_model(booster, x_test), loss) if len(x_test) else np.array([])

    train_metrics = _evaluation_metrics(y_train, train_pred)
    train_metrics.update(_credit_spread_selection_metrics(clean.iloc[:split_index], train_pred))
    metrics = _prefixed_metrics("train", train_metrics)
    if len(x_test):
        test_metrics = _evaluation_metrics(y_test, test_pred)
        test_metrics.update(_credit_spread_selection_metrics(clean.iloc[split_index:], test_pred))
        metrics.update(_prefixed_metrics("test", test_metrics))
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
    artifact_metadata = _artifact_metadata(clean)
    return XGBoostModelArtifact(
        model_type="xgboost_asymmetric_pseudohuber_v002",
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
        train_params.pop("eval_metric", None)
    return xgb.train(
        train_params,
        dtrain,
        num_boost_round=num_boost_round,
        obj=_asymmetric_objective(loss_config),
        custom_metric=_asymmetric_metric(loss_config),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if has_val else None,
        verbose_eval=False,
    )


def _asymmetric_objective(config: AsymmetricLossConfig):
    def objective(predt: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        labels = dtrain.get_label()
        residual = predt - labels
        weight = _asymmetric_weight(labels, predt, config)
        scaled = residual / config.huber_delta
        denom = np.sqrt(1.0 + np.square(scaled))
        grad = weight * residual / denom
        hess = weight / np.power(1.0 + np.square(scaled), 1.5)
        grad = np.clip(grad, -config.gradient_clip, config.gradient_clip)
        hess = np.maximum(hess, config.hessian_floor)
        return grad.astype(float), hess.astype(float)

    return objective


def _asymmetric_metric(config: AsymmetricLossConfig):
    def metric(predt: np.ndarray, dtrain) -> tuple[str, float]:
        labels = dtrain.get_label()
        residual = predt - labels
        weight = _asymmetric_weight(labels, predt, config)
        scaled = residual / config.huber_delta
        loss = weight * (config.huber_delta**2) * (np.sqrt(1.0 + np.square(scaled)) - 1.0)
        return "asym_pseudo_huber", float(np.mean(loss))

    return metric


def _asymmetric_weight(labels: np.ndarray, predt: np.ndarray, config: AsymmetricLossConfig) -> np.ndarray:
    raw_labels = _inverse_transform_target(labels, config)
    raw_pred = _inverse_transform_target(predt, config)
    downside = np.log1p(np.clip(-raw_labels, 0.0, config.target_clip) / config.downside_scale)
    optimistic_error = np.log1p(np.clip(raw_pred - raw_labels, 0.0, config.target_clip) / config.error_scale)
    weight = 1.0 + config.downside_penalty * downside + config.overprediction_penalty * optimistic_error
    return np.clip(weight, 1.0, config.max_multiplier)


def _transform_target(values: np.ndarray, config: AsymmetricLossConfig) -> np.ndarray:
    clipped = np.clip(values.astype(float), -config.target_clip, config.target_clip)
    return np.sign(clipped) * np.log1p(np.abs(clipped) / config.target_scale)


def _inverse_transform_target(values: np.ndarray, config: AsymmetricLossConfig) -> np.ndarray:
    clipped = np.clip(values.astype(float), -_transformed_target_cap(config), _transformed_target_cap(config))
    return np.sign(clipped) * np.expm1(np.abs(clipped)) * config.target_scale


def _transformed_target_cap(config: AsymmetricLossConfig) -> float:
    return float(np.log1p(config.target_clip / config.target_scale))


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


def _predict_model(booster, frame: pd.DataFrame) -> np.ndarray:
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
    y_model_all = _transform_target(y_all, loss_config)
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
            y_model_all[train_start:train_end],
            params,
            num_boost_round,
            loss_config,
        )
        pred = _inverse_transform_target(_predict_model(booster, x_test), loss_config)
        fold_metrics = _evaluation_metrics(y_all[test_start:test_end], pred)
        fold_metrics.update(_credit_spread_selection_metrics(df.iloc[test_start:test_end], pred))
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
                "feature_importance": _feature_importance(booster),
            }
        )
    return folds


def _feature_importance(booster) -> dict[str, float]:
    scores = booster.get_score(importance_type="gain")
    return {
        key: round(float(value), 6)
        for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    }


def _credit_spread_selection_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float | None]:
    """Top-decile selection metrics always reported in dollar PnL units.

    Overrides the target-unit values written by _evaluation_metrics so that
    exit criteria gates remain calibrated in dollars regardless of training target
    (e.g. return_on_risk vs expected_pnl).
    """
    if len(df) == 0 or len(y_pred) == 0:
        return {
            "top_decile_count": None,
            "top_decile_actual_mean": None,
            "top_decile_predicted_mean": None,
            "top_decile_win_rate": None,
            "top_decile_profit_factor": None,
            "top_decile_tail_loss_p05": None,
            "top_decile_worst_actual": None,
            "top_decile_max_drawdown": None,
            "top_decile_max_adverse_excursion": None,
            "top_decile_large_loss_rate": None,
            "top_decile_stop_loss_rate": None,
            "top_decile_return_on_risk_mean": None,
        }
    top_n = max(1, int(np.ceil(len(y_pred) * 0.1)))
    top_indices = np.argsort(y_pred)[-top_n:]
    selected = df.iloc[top_indices]
    # Dollar PnL from the dataset column — independent of what the training target is.
    pnl = pd.to_numeric(selected.get("expected_pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    return {
        "top_decile_count": int(len(selected)),
        "top_decile_actual_mean": round(float(np.mean(pnl)), 6) if len(pnl) else None,
        "top_decile_predicted_mean": round(float(np.mean(y_pred[top_indices])), 6),
        "top_decile_win_rate": round(float(np.mean(pnl > 0)), 6) if len(pnl) else None,
        "top_decile_profit_factor": _profit_factor(pnl),
        "top_decile_tail_loss_p05": round(float(np.percentile(pnl, 5)), 6) if len(pnl) >= 20 else None,
        "top_decile_worst_actual": round(float(np.min(pnl)), 6) if len(pnl) else None,
        "top_decile_max_drawdown": _max_drawdown(pnl),
        "top_decile_max_adverse_excursion": _column_max(selected, "max_adverse_excursion"),
        "top_decile_large_loss_rate": _column_mean(selected, "large_loss_label"),
        "top_decile_stop_loss_rate": _column_mean(selected, "stop_loss_hit"),
        "top_decile_return_on_risk_mean": _column_mean(selected, "return_on_risk"),
    }


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


def _cap_rows_per_underlying(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Return df with at most max_rows rows per underlying, preserving time order.

    Rows are sorted by entry_timestamp before capping so the earliest trades are
    kept — consistent with the dataset builder's own per-underlying cap logic.
    """
    if "entry_timestamp" in df.columns:
        df = df.sort_values("entry_timestamp")
    capped = (
        df.groupby("underlying", sort=False)
        .head(max_rows)
        .sort_values("entry_timestamp" if "entry_timestamp" in df.columns else df.columns[0])
        .reset_index(drop=True)
    )
    dropped = len(df) - len(capped)
    if dropped > 0:
        print(
            f"Per-underlying cap ({max_rows:,} rows): removed {dropped:,} rows "
            f"({dropped / len(df) * 100:.1f}%%), {len(capped):,} remaining.",
            flush=True,
        )
    return capped


def _oversample_high_vol(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Replicate high-volatility rows (vix_regime >= 2) in the training split.

    factor=5 means high-vol rows appear 5× in the returned frame.  The original
    order is preserved with oversampled rows appended at the end; XGBoost
    training does not depend on row order within the DMatrix.
    """
    if "vix_regime" not in df.columns or factor <= 1:
        return df
    vix_regime = pd.to_numeric(df["vix_regime"], errors="coerce")
    high_vol_mask = vix_regime >= 2.0
    high_vol_rows = df[high_vol_mask]
    if high_vol_rows.empty:
        return df
    replicated = pd.concat([df] + [high_vol_rows] * (factor - 1), ignore_index=True)
    print(
        f"High-vol oversampling (factor={factor}): {high_vol_mask.sum():,} high-vol rows "
        f"→ training set grows from {len(df):,} to {len(replicated):,} rows.",
        flush=True,
    )
    return replicated


if __name__ == "__main__":
    raise SystemExit(main())
