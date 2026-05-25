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
    "underlying_range_pct",
    "underlying_realized_vol_5d",
    "underlying_realized_vol_20d",
    "underlying_volume",
    "strike_distance_pct",
    "moneyness",
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
    metrics: dict[str, float | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transparent baseline model.")
    parser.add_argument("--input", required=True, help="JSONL file, parquet file, or dataset directory.")
    parser.add_argument("--output", required=True, help="Output model JSON path.")
    parser.add_argument("--target", default="expected_pnl")
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(Path(args.input))
    artifact = train_baseline(
        df,
        target_column=args.target,
        test_fraction=args.test_fraction,
        min_rows=args.min_rows,
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

    x_all, fill_values = _feature_matrix(clean, feature_columns)
    y_all = clean[target_column].to_numpy(dtype=float)
    split_index = _split_index(len(clean), test_fraction)

    x_train, x_test = x_all[:split_index], x_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]
    weights = _fit_linear_regression(x_train, y_train)
    train_pred = _predict(x_train, weights)
    test_pred = _predict(x_test, weights) if len(x_test) else np.array([])

    metrics = {
        "train_mae": _mae(y_train, train_pred),
        "train_rmse": _rmse(y_train, train_pred),
        "test_mae": _mae(y_test, test_pred) if len(x_test) else None,
        "test_rmse": _rmse(y_test, test_pred) if len(x_test) else None,
        "test_direction_accuracy": _direction_accuracy(y_test, test_pred) if len(x_test) else None,
        "test_top_decile_actual_mean": _top_decile_actual_mean(y_test, test_pred) if len(x_test) else None,
    }

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
    )


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in DEFAULT_FEATURE_COLUMNS:
        if column in df and pd.to_numeric(df[column], errors="coerce").notna().any():
            columns.append(column)
    return columns


def _feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, dict[str, float]]:
    frame = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values = {
        column: round(float(frame[column].median()) if frame[column].notna().any() else 0.0, 10)
        for column in feature_columns
    }
    frame = frame.fillna(fill_values)
    matrix = frame.to_numpy(dtype=float)
    intercept = np.ones((matrix.shape[0], 1), dtype=float)
    return np.hstack([intercept, matrix]), fill_values


def _split_index(row_count: int, test_fraction: float) -> int:
    test_rows = max(1, int(round(row_count * test_fraction))) if row_count > 1 else 0
    return max(1, row_count - test_rows)


def _fit_linear_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(x, y, rcond=None)
    return weights


def _predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return x @ weights


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.mean(np.abs(y_true - y_pred))), 6)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 6)


def _direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float(np.mean((y_true > 0) == (y_pred > 0))), 6)


def _top_decile_actual_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    top_n = max(1, int(np.ceil(len(y_pred) * 0.1)))
    indices = np.argsort(y_pred)[-top_n:]
    return round(float(np.mean(y_true[indices])), 6)


if __name__ == "__main__":
    raise SystemExit(main())
