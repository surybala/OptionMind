"""Train a transparent baseline model on candidate dataset rows."""
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


DEFAULT_FEATURE_COLUMNS = [
    "dte",
    "strike",
    "underlying_close",
    "underlying_return_1d",
    "underlying_return_5d",
    "underlying_return_20d",
    "underlying_range_pct",
    "underlying_realized_vol_5d",
    "underlying_realized_vol_20d",
    "underlying_sma_20_distance_pct",
    "underlying_above_sma_20",
    "underlying_volatility_ratio_5d_20d",
    "underlying_volume",
    "strike_distance_pct",
    "moneyness",
    "market_return_5d",
    "market_return_20d",
    "market_realized_vol_5d",
    "market_realized_vol_20d",
    "market_sma_20_distance_pct",
    "market_above_sma_20",
    "market_volatility_ratio_5d_20d",
    "option_entry_open",
    "option_entry_high",
    "option_entry_low",
    "option_entry_price",
    "option_entry_range_pct",
    "option_entry_volume",
    "option_entry_trade_count",
    "option_entry_vwap",
]


@dataclass(frozen=True)
class BaselineModelArtifact:
    model_type: str
    created_at: str
    target_column: str
    feature_columns: list[str]
    intercept: float
    coefficients: dict[str, float]
    fill_values: dict[str, float]
    train_rows: int
    test_rows: int
    metrics: dict[str, Any]
    walk_forward: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transparent baseline model.")
    parser.add_argument("--input", required=True, help="JSONL file, parquet file, or dataset directory.")
    parser.add_argument("--output", required=True, help="Output model JSON path.")
    parser.add_argument("--target", default="expected_pnl")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=2)
    parser.add_argument("--walk-forward-folds", type=int, default=3)
    parser.add_argument("--min-walk-forward-train-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(Path(args.input))
    artifact = train_baseline(
        df,
        target_column=args.target,
        test_fraction=args.test_fraction,
        min_rows=args.min_rows,
        walk_forward_folds=args.walk_forward_folds,
        min_walk_forward_train_rows=args.min_walk_forward_train_rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(asdict(artifact), indent=2, sort_keys=True))
    return 0


def train_baseline(
    df: pd.DataFrame,
    *,
    target_column: str = "expected_pnl",
    test_fraction: float = 0.25,
    min_rows: int = 2,
    walk_forward_folds: int = 3,
    min_walk_forward_train_rows: int | None = None,
) -> BaselineModelArtifact:
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

    y_train, y_test = y_all[:split_index], y_all[split_index:]
    x_train, fill_values = _fit_feature_matrix(clean.iloc[:split_index], feature_columns)
    x_test = _transform_feature_matrix(clean.iloc[split_index:], feature_columns, fill_values)
    weights = _fit_linear_regression(x_train, y_train)
    train_pred = _predict(x_train, weights)
    test_pred = _predict(x_test, weights) if len(x_test) else np.array([])

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
    )
    metrics.update(_walk_forward_summary(walk_forward))

    return BaselineModelArtifact(
        model_type="linear_least_squares_v001",
        created_at=datetime.now(UTC).isoformat(),
        target_column=target_column,
        feature_columns=feature_columns,
        intercept=round(float(weights[0]), 10),
        coefficients={
            column: round(float(weight), 10)
            for column, weight in zip(feature_columns, weights[1:])
        },
        fill_values=fill_values,
        train_rows=int(len(y_train)),
        test_rows=int(len(y_test)),
        metrics=metrics,
        walk_forward=walk_forward,
    )


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in DEFAULT_FEATURE_COLUMNS:
        if column in df and pd.to_numeric(df[column], errors="coerce").notna().any():
            columns.append(column)
    return columns


def _fit_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = {
        column: round(float(frame[column].median()) if frame[column].notna().any() else 0.0, 10)
        for column in feature_columns
    }
    return _transform_feature_matrix(df, feature_columns, fill_values), fill_values


def _transform_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    fill_values: dict[str, float],
) -> np.ndarray:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(fill_values)
    matrix = frame.to_numpy(dtype=float)
    intercept = np.ones((matrix.shape[0], 1), dtype=float)
    return np.hstack([intercept, matrix])


def _split_index(row_count: int, test_fraction: float) -> int:
    test_rows = max(1, int(round(row_count * test_fraction))) if row_count > 1 else 0
    return max(1, row_count - test_rows)


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return weights


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return x @ weights


def _walk_forward_validation(
    df: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    fold_count: int,
    min_train_rows: int,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    y_all = df[target_column].to_numpy(dtype=float)
    for fold_number, train_start, train_end, test_start, test_end in _walk_forward_splits(
        len(df),
        fold_count=fold_count,
        min_train_rows=min_train_rows,
    ):
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        x_train, fold_fill_values = _fit_feature_matrix(train_df, feature_columns)
        x_test = _transform_feature_matrix(test_df, feature_columns, fold_fill_values)
        weights = _fit_linear_regression(x_train, y_all[train_start:train_end])
        pred = _predict(x_test, weights)
        fold_metrics = _evaluation_metrics(y_all[test_start:test_end], pred)
        folds.append(
            {
                "fold": fold_number,
                "train_start": _row_timestamp(df, train_start),
                "train_end": _row_timestamp(df, train_end - 1),
                "test_start": _row_timestamp(df, test_start),
                "test_end": _row_timestamp(df, test_end - 1),
                "train_rows": int(train_end - train_start),
                "test_rows": int(test_end - test_start),
                "metrics": fold_metrics,
            }
        )
    return folds


def _walk_forward_splits(
    row_count: int,
    *,
    fold_count: int,
    min_train_rows: int,
) -> list[tuple[int, int, int, int, int]]:
    if fold_count <= 0 or row_count <= min_train_rows:
        return []
    test_indices = np.array_split(np.arange(min_train_rows, row_count), min(fold_count, row_count - min_train_rows))
    splits: list[tuple[int, int, int, int, int]] = []
    for fold_number, chunk in enumerate(test_indices, start=1):
        if len(chunk) == 0:
            continue
        test_start = int(chunk[0])
        test_end = int(chunk[-1]) + 1
        splits.append((fold_number, 0, test_start, test_start, test_end))
    return splits


def _evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    top_indices = np.sort(_top_decile_indices(y_pred))
    top_actual = y_true[top_indices]
    top_pred = y_pred[top_indices]
    return {
        "rows": int(len(y_true)),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "actual_mean": _mean(y_true),
        "predicted_mean": _mean(y_pred),
        "direction_accuracy": _direction_accuracy(y_true, y_pred),
        "win_rate": _win_rate(y_true),
        "profit_factor": _profit_factor(y_true),
        "max_drawdown": _max_drawdown(y_true),
        "tail_loss_p05": _percentile(y_true, 5),
        "worst_actual": _min(y_true),
        "top_decile_count": int(len(top_indices)),
        "top_decile_actual_mean": _mean(top_actual),
        "top_decile_predicted_mean": _mean(top_pred),
        "top_decile_win_rate": _win_rate(top_actual),
        "top_decile_profit_factor": _profit_factor(top_actual),
        "top_decile_max_drawdown": _max_drawdown(top_actual),
        "top_decile_tail_loss_p05": _percentile(top_actual, 5),
        "top_decile_worst_actual": _min(top_actual),
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _empty_test_metrics(prefix: str) -> dict[str, None]:
    names = [
        "mae",
        "rmse",
        "actual_mean",
        "predicted_mean",
        "direction_accuracy",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "tail_loss_p05",
        "worst_actual",
        "top_decile_count",
        "top_decile_actual_mean",
        "top_decile_predicted_mean",
        "top_decile_win_rate",
        "top_decile_profit_factor",
        "top_decile_max_drawdown",
        "top_decile_tail_loss_p05",
        "top_decile_worst_actual",
    ]
    return {f"{prefix}_{name}": None for name in names}


def _walk_forward_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"walk_forward_folds": int(len(folds))}
    if not folds:
        return summary

    metric_names = [
        "mae",
        "rmse",
        "direction_accuracy",
        "profit_factor",
        "max_drawdown",
        "top_decile_actual_mean",
        "top_decile_win_rate",
        "top_decile_profit_factor",
        "top_decile_max_drawdown",
        "top_decile_worst_actual",
    ]
    for name in metric_names:
        values = [fold["metrics"].get(name) for fold in folds]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        summary[f"walk_forward_{name}_mean"] = _round_or_none(np.mean(numeric_values)) if numeric_values else None
    summary["walk_forward_test_rows"] = int(sum(fold["test_rows"] for fold in folds))
    return summary


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.mean(np.abs(y_true - y_pred))), 6)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 6)


def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.mean((y_true > 0) == (y_pred > 0))), 6)


def _top_decile_indices(y_pred: np.ndarray) -> np.ndarray:
    top_n = max(1, int(np.ceil(len(y_pred) * 0.1)))
    return np.argsort(y_pred)[-top_n:]


def _mean(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.mean(values)), 6)


def _min(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.min(values)), 6)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.percentile(values, percentile)), 6)


def _win_rate(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.mean(values > 0)), 6)


def _profit_factor(values: np.ndarray) -> float | None:
    gains = float(np.sum(values[values > 0]))
    losses = float(np.sum(values[values < 0]))
    if losses == 0.0:
        return None
    return round(gains / abs(losses), 6)


def _max_drawdown(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    equity = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))[:-1]
    drawdowns = running_peak - equity
    return round(float(max(0.0, np.max(drawdowns))), 6)


def _round_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _row_timestamp(df: pd.DataFrame, index: int) -> str | None:
    if "entry_timestamp" not in df:
        return None
    value = df.iloc[index]["entry_timestamp"]
    if pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value)
    return timestamp.isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
