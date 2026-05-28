"""Train a transparent baseline model on candidate dataset rows."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset


DEFAULT_FEATURE_COLUMNS = [
    # Contract structure
    "is_pcs",
    "is_ccs",
    "dte",
    "strike",
    "strike_distance_pct",
    "moneyness",
    # Credit spread structure
    "spread_width",
    "entry_credit",
    "max_profit",
    "max_loss",
    "credit_to_width",
    "long_option_entry_price",
    "long_option_entry_volume",
    "long_option_entry_trade_count",
    "long_option_entry_vwap",
    # Underlying price features
    "underlying_close",
    "underlying_return_1d",
    "underlying_return_3d",
    "underlying_return_5d",
    "underlying_return_20d",
    "underlying_range_pct",
    "underlying_realized_vol_5d",
    "underlying_realized_vol_10d",
    "underlying_realized_vol_20d",
    "underlying_sma_20_distance_pct",
    "underlying_above_sma_20",
    "underlying_volatility_ratio_5d_20d",
    "underlying_volume",
    "underlying_skew_5d",
    # Idiosyncratic vs market vol: ETF moving faster/slower than SPY
    "underlying_vol_vs_market",
    # Vol momentum: 5d realized vol relative to 10d — captures whether volatility
    # is currently accelerating (>1) or decelerating (<1) independent of level.
    "vol_acceleration",
    # Market regime
    "market_return_5d",
    "market_return_20d",
    "market_realized_vol_5d",
    "market_realized_vol_20d",
    "market_sma_20_distance_pct",
    "market_above_sma_20",
    "market_volatility_ratio_5d_20d",
    # Option entry bar — price level (open/high/low dropped: same-day OHLC, spurious importance)
    "option_entry_price",
    "option_entry_range_pct",
    "option_entry_volume",
    "option_entry_trade_count",
    "option_entry_vwap",
    # Option lookback
    "option_volume_5d_avg",
    "option_trade_count_5d_avg",
    # Unusual option activity today vs recent baseline
    "option_activity_spike",
    # Black-Scholes Greeks and implied volatility
    "implied_volatility",
    "option_delta",
    "option_gamma",
    "option_theta",
    "option_vega",
    "iv_vs_hv5d",
    "iv_vs_hv20d",
    # IV term structure: long-leg IV minus short-leg IV (vol surface slope proxy)
    "iv_skew_wing",
    # VIX market regime
    # vix_close (continuous) replaced by vix_regime (3-level bucket: 0=low <15,
    # 1=elevated 15-30, 2=high ≥30) to prevent memorising exact index levels.
    "vix_regime",
    "vix_return_5d",
    "vix_realized_vol_5d",
    # vix_above_20 and vix_above_30 dropped: fully absorbed by vix_regime (zero importance)
    # Event risk — earnings (continuous only; binary flags had zero importance)
    "days_to_earnings",
    # Ex-dividend risk
    "days_to_ex_dividend",
    # Macro event risk (continuous only; binary flags had zero importance)
    "days_to_fomc",
    "days_to_macro_event",
    # Credit efficiency: premium per DTE per dollar at risk
    # Normalises entry_credit by both time and capital exposure so the model can
    # compare trades across different DTEs and spread widths on equal footing.
    "credit_per_day_per_risk",
]

DEFAULT_FEATURE_VERSION = "features_v005"
DEFAULT_LABEL_VERSION = "short_option_labels_v002"


@dataclass(frozen=True)
class BaselineModelArtifact:
    model_type: str
    created_at: str
    target_column: str
    feature_version: str
    label_version: str
    data_range: dict[str, str | None]
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
    parser.add_argument("--embargo-days", type=int, default=0, help="Calendar days to exclude between train and test in each walk-forward fold.")
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
        embargo_days=args.embargo_days,
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
    embargo_days: int = 0,
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

    clean = _engineer_features(clean)
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
        embargo_days=embargo_days,
    )
    metrics.update(_walk_forward_summary(walk_forward))

    artifact_metadata = _artifact_metadata(clean)
    return BaselineModelArtifact(
        model_type="linear_least_squares_v001",
        created_at=datetime.now(UTC).isoformat(),
        target_column=target_column,
        feature_version=artifact_metadata["feature_version"],
        label_version=artifact_metadata["label_version"],
        data_range=artifact_metadata["data_range"],
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


def _artifact_metadata(df: pd.DataFrame) -> dict[str, Any]:
    manifest = df.attrs.get("dataset_manifest") if hasattr(df, "attrs") else None
    manifest_metadata = dict((manifest or {}).get("metadata") or {})
    return {
        "feature_version": str(manifest_metadata.get("feature_set_version") or DEFAULT_FEATURE_VERSION),
        "label_version": str(manifest_metadata.get("label_version") or _mode_value(df, "label_version") or DEFAULT_LABEL_VERSION),
        "data_range": _data_range(df, manifest_metadata),
    }


def _mode_value(df: pd.DataFrame, column: str) -> Any | None:
    if column not in df:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    return values.astype(str).mode().iloc[0]


def _data_range(df: pd.DataFrame, manifest_metadata: dict[str, Any]) -> dict[str, str | None]:
    start = manifest_metadata.get("entry_start")
    end = manifest_metadata.get("entry_end")
    if "entry_timestamp" in df and (not start or not end):
        timestamps = pd.to_datetime(df["entry_timestamp"], errors="coerce").dropna()
        if not timestamps.empty:
            start = start or timestamps.min().isoformat()
            end = end or timestamps.max().isoformat()
    return {"start": str(start) if start else None, "end": str(end) if end else None}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engineered features from raw dataset columns.

    Existing:
      - ``vix_regime``: 3-level ordinal bucket from ``vix_close`` (0=low <15,
        1=elevated 15-30, 2=high ≥30).  Prevents memorising exact VIX levels.
      - ``is_pcs`` / ``is_ccs``: one-hot from ``strategy`` column.

    New in features_v004:
      - ``underlying_vol_vs_market``: underlying 5d realized vol divided by
        market 5d realized vol.  Captures idiosyncratic ETF-level stress that
        is independent of broad market turbulence.
      - ``option_activity_spike``: today's option trade count divided by the
        5-day average.  Detects unusual positioning activity on entry day.
      - ``iv_skew_wing``: long-leg implied volatility minus short-leg IV,
        computed via Newton-Raphson Black-Scholes.  Approximates the slope of
        the vol surface between the two strikes — a wide positive spread on
        PCS signals steep put skew (protective demand).

    New in features_v005:
      - ``vol_acceleration``: 5d realized vol divided by 10d realized vol.
        Ratio > 1 means vol is currently expanding; < 1 means it is contracting.
        Complements ``underlying_volatility_ratio_5d_20d`` (5d/20d) by isolating
        near-term momentum before the longer window absorbs the move.
    """
    df = df.copy()

    # vix_regime bucket
    if "vix_close" in df.columns:
        vix = pd.to_numeric(df["vix_close"], errors="coerce")
        df["vix_regime"] = (
            (vix >= 15).astype("Int64") + (vix >= 30).astype("Int64")
        ).astype(float)

    # is_pcs / is_ccs flags
    if "strategy" in df.columns:
        strategy = df["strategy"].astype(str).str.upper()
        df["is_pcs"] = (strategy == "PCS").astype(float)
        df["is_ccs"] = (strategy == "CCS").astype(float)

    # Idiosyncratic vol: ETF vs market
    if "underlying_realized_vol_5d" in df.columns and "market_realized_vol_5d" in df.columns:
        und_vol = pd.to_numeric(df["underlying_realized_vol_5d"], errors="coerce")
        mkt_vol = pd.to_numeric(df["market_realized_vol_5d"], errors="coerce")
        df["underlying_vol_vs_market"] = (und_vol / mkt_vol.replace(0.0, np.nan)).round(8)

    # Vol momentum: 5d realized vol divided by 10d realized vol.
    # Ratio > 1 signals vol is expanding (spike in progress); < 1 signals mean-reversion.
    # Distinct from underlying_volatility_ratio_5d_20d (5d/20d) — the 5d/10d window
    # catches near-term acceleration that the longer ratio smooths away.
    if "underlying_realized_vol_5d" in df.columns and "underlying_realized_vol_10d" in df.columns:
        vol_5d = pd.to_numeric(df["underlying_realized_vol_5d"], errors="coerce")
        vol_10d = pd.to_numeric(df["underlying_realized_vol_10d"], errors="coerce")
        df["vol_acceleration"] = (vol_5d / vol_10d.replace(0.0, np.nan)).round(8)

    # Unusual option activity relative to recent baseline
    if "option_entry_trade_count" in df.columns and "option_trade_count_5d_avg" in df.columns:
        today_cnt = pd.to_numeric(df["option_entry_trade_count"], errors="coerce")
        avg_cnt = pd.to_numeric(df["option_trade_count_5d_avg"], errors="coerce")
        df["option_activity_spike"] = (today_cnt / avg_cnt.replace(0.0, np.nan)).round(8)

    # Credit efficiency: entry credit per DTE per dollar at risk.
    # Normalises raw credit across different expirations and spread widths so
    # a 7-DTE trade paying $0.50 on a $5-wide spread compares fairly to a
    # 45-DTE trade paying $1.50 on a $10-wide spread.
    if "entry_credit" in df.columns and "dte" in df.columns and "max_loss" in df.columns:
        credit = pd.to_numeric(df["entry_credit"], errors="coerce")
        dte = pd.to_numeric(df["dte"], errors="coerce")
        max_loss = pd.to_numeric(df["max_loss"], errors="coerce")
        df["credit_per_day_per_risk"] = (
            credit / (dte.replace(0.0, np.nan) * max_loss.replace(0.0, np.nan))
        ).round(8)

    # IV skew wing: long-leg IV minus short-leg IV.
    # Requires columns present in credit-spread rows only; silently skipped for
    # short-option datasets that lack long_strike / long_option_entry_price.
    _iv_skew_required = {"long_option_entry_price", "underlying_close", "long_strike", "option_type", "dte"}
    if _iv_skew_required.issubset(df.columns) and "iv_skew_wing" not in df.columns:
        df["iv_skew_wing"] = _compute_iv_skew_wing(df)

    return df


def _compute_iv_skew_wing(df: pd.DataFrame, risk_free_rate: float = 0.045) -> pd.Series:
    """Return long-leg IV minus short-leg IV for each credit-spread row.

    Uses the same Newton-Raphson Black-Scholes solver as the dataset builder.
    Rows where computation fails (non-positive inputs, unconverged) yield NaN.
    """
    results = np.full(len(df), np.nan)
    short_iv = pd.to_numeric(df.get("implied_volatility"), errors="coerce").to_numpy()
    long_price = pd.to_numeric(df["long_option_entry_price"], errors="coerce").to_numpy()
    spot = pd.to_numeric(df["underlying_close"], errors="coerce").to_numpy()
    k_long = pd.to_numeric(df["long_strike"], errors="coerce").to_numpy()
    dte_arr = pd.to_numeric(df["dte"], errors="coerce").to_numpy()
    opt_type = df["option_type"].astype(str).str.lower().to_numpy()

    for i in range(len(df)):
        try:
            S, K, T = float(spot[i]), float(k_long[i]), float(dte_arr[i]) / 365.0
            price = float(long_price[i])
            otype = str(opt_type[i])
            siv = short_iv[i]
            if not (S > 0 and K > 0 and T > 0 and price > 0 and otype in {"call", "put"}):
                continue
            liv = _bs_iv_simple(price, S, K, T, risk_free_rate, otype)
            if liv is None or np.isnan(siv):
                continue
            results[i] = round(liv - float(siv), 8)
        except Exception:
            pass
    return pd.Series(results, index=df.index)


def _bs_iv_simple(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    max_iter: int = 60,
    tol: float = 1e-6,
) -> float | None:
    """Newton-Raphson Black-Scholes IV solver (training-time use only)."""
    sigma = 0.25
    for _ in range(max_iter):
        try:
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            nd1 = (1.0 + math.erf(d1 / math.sqrt(2.0))) / 2.0
            nd2 = (1.0 + math.erf(d2 / math.sqrt(2.0))) / 2.0
            pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
            disc = math.exp(-r * T)
            if option_type == "call":
                price = S * nd1 - K * disc * nd2
            else:
                price = K * disc * (1.0 - nd2) - S * (1.0 - nd1)
            vega = S * pdf_d1 * sqrt_T
            if abs(vega) < 1e-10:
                return None
            diff = price - market_price
            if abs(diff) < tol:
                return round(max(0.001, min(sigma, 10.0)), 8)
            sigma = max(0.001, min(sigma - diff / vega, 10.0))
        except (ValueError, ZeroDivisionError):
            return None
    return None


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
                "embargo_rows": int(test_start - train_end),
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
    embargo_rows: int = 0,
) -> list[tuple[int, int, int, int, int]]:
    if fold_count <= 0 or row_count <= min_train_rows:
        return []
    test_indices = np.array_split(np.arange(min_train_rows, row_count), min(fold_count, row_count - min_train_rows))
    splits: list[tuple[int, int, int, int, int]] = []
    for fold_number, chunk in enumerate(test_indices, start=1):
        if len(chunk) == 0:
            continue
        train_end = int(chunk[0])
        test_start = min(train_end + embargo_rows, int(chunk[-1]) + 1)
        test_end = int(chunk[-1]) + 1
        if test_start >= test_end:
            continue
        splits.append((fold_number, 0, train_end, test_start, test_end))
    return splits


def _embargo_rows_from_days(df: pd.DataFrame, embargo_days: int) -> int:
    """Estimate how many rows correspond to embargo_days of calendar time."""
    if embargo_days <= 0 or "entry_timestamp" not in df.columns:
        return 0
    ts = pd.to_datetime(df["entry_timestamp"], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return 0
    total_days = max(1.0, (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400)
    rows_per_day = len(ts) / total_days
    return max(0, int(np.ceil(rows_per_day * embargo_days)))


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
