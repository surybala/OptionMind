"""Train a grouped XGBoost early-exit model on intraday risk rows.

The dataset contains many minute-state rows for the same spread candidate.
To avoid leakage, all rows belonging to the same entry must stay in the same
train/validation/test fold. This trainer therefore splits chronologically by
trade entry rather than by raw row index.
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
    _artifact_fingerprint,
    _artifact_metadata,
    _feature_importance,
)

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


TARGET_CHOICES = (
    "stop_loss_hit_5m",
    "stop_loss_hit_15m",
    "stop_loss_hit_30m",
)

CORE_FEATURE_COLUMNS = [
    "is_pcs",
    "is_ccs",
    "dte",
    "spread_width",
    "entry_credit",
    "max_loss",
    "stop_debit",
    "profit_take_debit",
    "current_debit",
    "pnl_per_contract",
    "profit_captured_pct",
    "stop_distance_pct",
    "minutes_since_entry",
    "minutes_to_expiry",
    "underlying_close",
    "current_debit_to_stop",
    "current_debit_to_profit_take",
    "debit_to_width",
    "loss_pct_of_max_loss",
    "credit_retained_pct",
]

OPTIONAL_FEATURE_COLUMNS = [
    "underlying_return_5m",
    "underlying_return_15m",
    "underlying_return_30m",
    "abs_underlying_return_5m",
    "abs_underlying_return_15m",
    "abs_underlying_return_30m",
    "underlying_realized_vol_15m",
    "underlying_realized_vol_30m",
    "underlying_vol_ratio_15m_30m",
    "short_leg_close",
    "long_leg_close",
    "short_leg_share_of_debit",
    "long_leg_share_of_debit",
    "short_leg_volume",
    "long_leg_volume",
    "short_leg_trade_count",
    "long_leg_trade_count",
    "leg_volume_imbalance",
    "leg_trade_count_imbalance",
    "market_trend_uptrend",
    "market_trend_sideways",
    "market_trend_downtrend",
    "market_volatility_low",
    "market_volatility_medium",
    "market_volatility_high",
]

ALL_FEATURE_COLUMNS = CORE_FEATURE_COLUMNS + OPTIONAL_FEATURE_COLUMNS


@dataclass(frozen=True)
class IntradayRiskMonitorArtifact:
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
    train_entries: int
    test_entries: int
    recommended_close_threshold: float
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
        description="Train a grouped XGBoost intraday risk monitor on intraday_risk_rows."
    )
    parser.add_argument("--input", required=True, help="Parquet dataset directory or file.")
    parser.add_argument("--output", required=True, help="Output artifact JSON path.")
    parser.add_argument("--model-output", default=None, help="XGBoost model output path.")
    parser.add_argument("--target", default="stop_loss_hit_30m", choices=list(TARGET_CHOICES))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--min-walk-forward-train-groups", type=int, default=None)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-weight", type=float, default=20.0)
    parser.add_argument("--reg-lambda", type=float, default=20.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--scale-pos-weight", type=float, default=None)
    parser.add_argument(
        "--min-threshold-recall",
        type=float,
        default=0.75,
        help="Recall floor used when choosing the close threshold from validation predictions.",
    )
    parser.add_argument(
        "--max-threshold-close-rate",
        type=float,
        default=0.15,
        help="Soft upper bound on close rate used when choosing the close threshold.",
    )
    parser.add_argument(
        "--max-threshold-false-close-rate",
        type=float,
        default=0.12,
        help="Soft upper bound on false-close rate used when choosing the close threshold.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Use a fixed probability threshold instead of auto-selecting via validation set. "
        "All metrics (train, test, walk-forward) are evaluated at this threshold.",
    )
    parser.add_argument(
        "--include-features",
        default=None,
        help="Comma-separated feature columns to include. Defaults to all engineered features.",
    )
    parser.add_argument(
        "--exclude-features",
        default=None,
        help="Comma-separated feature columns to exclude after engineering.",
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
    artifact = train_intraday_risk_monitor(
        df,
        model_output=model_output,
        dataset_path=str(args.input),
        training_command=" ".join(shlex.quote(part) for part in sys.argv),
        target_column=args.target,
        test_fraction=args.test_fraction,
        min_rows=args.min_rows,
        walk_forward_folds=args.walk_forward_folds,
        min_walk_forward_train_groups=args.min_walk_forward_train_groups,
        embargo_days=args.embargo_days,
        num_boost_round=args.num_boost_round,
        val_fraction=args.val_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
        scale_pos_weight=args.scale_pos_weight,
        fixed_threshold=args.fixed_threshold,
        min_threshold_recall=args.min_threshold_recall,
        max_threshold_close_rate=args.max_threshold_close_rate,
        max_threshold_false_close_rate=args.max_threshold_false_close_rate,
        include_features=_parse_feature_list(args.include_features),
        exclude_features=_parse_feature_list(args.exclude_features),
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


def train_intraday_risk_monitor(
    df: pd.DataFrame,
    *,
    model_output: Path,
    dataset_path: str | None = None,
    training_command: str | None = None,
    target_column: str = "stop_loss_hit_30m",
    test_fraction: float = 0.25,
    min_rows: int = 20,
    walk_forward_folds: int = 4,
    min_walk_forward_train_groups: int | None = None,
    embargo_days: int = 1,
    val_fraction: float = 0.15,
    early_stopping_rounds: int = 20,
    num_boost_round: int = 300,
    scale_pos_weight: float | None = None,
    fixed_threshold: float | None = None,
    min_threshold_recall: float = 0.75,
    max_threshold_close_rate: float | None = 0.15,
    max_threshold_false_close_rate: float | None = 0.12,
    include_features: list[str] | None = None,
    exclude_features: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> IntradayRiskMonitorArtifact:
    if xgb is None:
        raise ImportError("xgboost is required. Install xgboost and its native runtime dependencies.")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column '{target_column}' in dataset.")

    clean = _prepare_frame(df, target_column=target_column)
    if len(clean) < min_rows:
        raise ValueError(f"Need at least {min_rows} labeled rows, found {len(clean)}")

    clean = _engineer_intraday_features(clean)
    feature_columns = _select_feature_columns(
        clean,
        include_features=include_features,
        exclude_features=exclude_features,
    )
    if not feature_columns:
        raise ValueError("No usable numeric feature columns found.")

    clean, groups = _group_index(clean)
    train_groups_df, test_groups_df, split_summary = _group_holdout_split(
        groups,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
    )
    if train_groups_df.empty or test_groups_df.empty:
        raise ValueError("Not enough grouped entries to create a chronological holdout split.")

    train_groups = set(train_groups_df["group_key"])
    test_groups = set(test_groups_df["group_key"])
    train_df = clean[clean["_group_key"].isin(train_groups)].copy()
    test_df = clean[clean["_group_key"].isin(test_groups)].copy()

    y_train = train_df[target_column].to_numpy(dtype=float)
    y_test = test_df[target_column].to_numpy(dtype=float)
    neg = float(np.sum(y_train == 0))
    pos = float(np.sum(y_train == 1))
    spw = scale_pos_weight if scale_pos_weight is not None else (neg / pos if pos > 0 else 1.0)

    model_params = _default_params(spw)
    if params:
        model_params.update(params)

    train_group_count = len(train_groups_df)
    val_group_index = (
        _split_index(train_group_count, val_fraction)
        if val_fraction > 0 and train_group_count > 1
        else train_group_count
    )
    train_sub_groups = set(train_groups_df.iloc[:val_group_index]["group_key"])
    val_groups = set(train_groups_df.iloc[val_group_index:]["group_key"])

    train_sub_df = clean[clean["_group_key"].isin(train_sub_groups)].copy()
    val_df = clean[clean["_group_key"].isin(val_groups)].copy()

    fill_values = _compute_fill_values(train_df, feature_columns)
    x_train_sub = _build_dmatrix(train_sub_df, feature_columns, fill_values, train_sub_df[target_column].to_numpy(dtype=float))
    x_val = _build_dmatrix(val_df, feature_columns, fill_values, val_df[target_column].to_numpy(dtype=float))
    x_test_frame = _transform_frame(test_df, feature_columns, fill_values)
    x_train_frame = _transform_frame(train_df, feature_columns, fill_values)

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

    if fixed_threshold is not None:
        recommended_threshold = fixed_threshold
    else:
        threshold_frame = val_df if len(val_df) else train_sub_df
        threshold_prob = _predict_prob(booster, _transform_frame(threshold_frame, feature_columns, fill_values))
        recommended_threshold = _choose_close_threshold(
            threshold_frame,
            target_column=target_column,
            y_prob=threshold_prob,
            min_recall=min_threshold_recall,
            max_close_rate=max_threshold_close_rate,
            max_false_close_rate=max_threshold_false_close_rate,
        )

    train_prob = _predict_prob(booster, x_train_frame)
    test_prob = _predict_prob(booster, x_test_frame) if len(x_test_frame) else np.array([])

    metrics = _prefixed_metrics("train", _clf_metrics(y_train, train_prob, recommended_threshold))
    metrics.update(_prefixed_policy_metrics("train", train_df, target_column, train_prob, recommended_threshold))
    if len(test_prob):
        metrics.update(_prefixed_metrics("test", _clf_metrics(y_test, test_prob, recommended_threshold)))
        metrics.update(_prefixed_policy_metrics("test", test_df, target_column, test_prob, recommended_threshold))

    walk_forward = _walk_forward_grouped(
        clean,
        groups,
        target_column=target_column,
        feature_columns=feature_columns,
        fold_count=walk_forward_folds,
        min_train_groups=min_walk_forward_train_groups or max(25, len(train_groups_df)),
        params=model_params,
        num_boost_round=best_rounds,
        embargo_days=embargo_days,
        val_fraction=val_fraction,
        early_stopping_rounds=early_stopping_rounds,
        fixed_threshold=fixed_threshold,
        min_threshold_recall=min_threshold_recall,
        max_threshold_close_rate=max_threshold_close_rate,
        max_threshold_false_close_rate=max_threshold_false_close_rate,
    )
    metrics.update(_walk_forward_summary(walk_forward))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_output)

    metadata = _artifact_metadata(clean)
    dataset_info = dict(metadata["dataset"])
    if dataset_path:
        dataset_info["input_path"] = dataset_path
    return IntradayRiskMonitorArtifact(
        model_type="xgboost_intraday_risk_monitor_v001",
        created_at=datetime.now(UTC).isoformat(),
        target_column=target_column,
        feature_version="intraday_risk_live_monitor_features_v001",
        label_version=f"intraday_risk_{target_column}_v001",
        data_range=metadata["data_range"],
        model_path=str(model_output),
        feature_columns=feature_columns,
        fill_values=fill_values,
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        train_positive_rate=round(float(np.mean(y_train)), 6),
        test_positive_rate=round(float(np.mean(y_test)), 6) if len(y_test) else 0.0,
        train_entries=int(len(train_groups)),
        test_entries=int(len(test_groups)),
        recommended_close_threshold=round(float(recommended_threshold), 6),
        dataset=dataset_info,
        training_command=training_command,
        data_fingerprint=_artifact_fingerprint(
            dataset=dataset_info,
            target_column=target_column,
            feature_columns=feature_columns,
            data_quality_filters=metadata["data_quality_filters"],
            split_summary=split_summary,
        ),
        data_quality_filters=metadata["data_quality_filters"],
        split_summary=split_summary,
        params={**model_params, "num_boost_round": int(best_rounds)},
        feature_importance=_feature_importance(booster),
        metrics=metrics,
        walk_forward=walk_forward,
    )


def _prepare_frame(df: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    clean = df.copy()
    clean["entry_timestamp"] = pd.to_datetime(clean["entry_timestamp"], utc=True, errors="coerce")
    clean["state_timestamp"] = pd.to_datetime(clean["state_timestamp"], utc=True, errors="coerce")
    clean[target_column] = pd.to_numeric(clean[target_column], errors="coerce")
    clean = clean.dropna(subset=["entry_timestamp", "state_timestamp", target_column, "underlying", "short_option_symbol", "long_option_symbol"])
    clean[target_column] = clean[target_column].astype(int)
    return clean.sort_values(["entry_timestamp", "state_timestamp"]).reset_index(drop=True)


def _engineer_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    strategy = clean.get("strategy", pd.Series(index=clean.index, dtype=object)).astype(str).str.upper()
    clean["is_pcs"] = (strategy == "PCS").astype(float)
    clean["is_ccs"] = (strategy == "CCS").astype(float)

    current_debit = pd.to_numeric(clean.get("current_debit"), errors="coerce")
    stop_debit = pd.to_numeric(clean.get("stop_debit"), errors="coerce")
    profit_take_debit = pd.to_numeric(clean.get("profit_take_debit"), errors="coerce")
    spread_width = pd.to_numeric(clean.get("spread_width"), errors="coerce")
    entry_credit = pd.to_numeric(clean.get("entry_credit"), errors="coerce")
    max_loss = pd.to_numeric(clean.get("max_loss"), errors="coerce")
    pnl_per_contract = pd.to_numeric(clean.get("pnl_per_contract"), errors="coerce")

    clean["current_debit_to_stop"] = (current_debit / stop_debit.replace(0.0, np.nan)).round(8)
    clean["current_debit_to_profit_take"] = (current_debit / profit_take_debit.replace(0.0, np.nan)).round(8)
    clean["debit_to_width"] = (current_debit / spread_width.replace(0.0, np.nan)).round(8)
    clean["loss_pct_of_max_loss"] = (
        ((-pnl_per_contract).clip(lower=0.0) / (max_loss.replace(0.0, np.nan) * 100.0))
    ).round(8)
    clean["credit_retained_pct"] = (current_debit / entry_credit.replace(0.0, np.nan)).round(8)

    for minutes in (5, 15, 30):
        ret_col = f"underlying_return_{minutes}m"
        if ret_col in clean:
            clean[ret_col] = pd.to_numeric(clean[ret_col], errors="coerce")
            clean[f"abs_underlying_return_{minutes}m"] = clean[ret_col].abs().round(8)

    rv15 = pd.to_numeric(clean.get("underlying_realized_vol_15m"), errors="coerce")
    rv30 = pd.to_numeric(clean.get("underlying_realized_vol_30m"), errors="coerce")
    if rv15.notna().any() or rv30.notna().any():
        clean["underlying_realized_vol_15m"] = rv15
        clean["underlying_realized_vol_30m"] = rv30
        clean["underlying_vol_ratio_15m_30m"] = (rv15 / rv30.replace(0.0, np.nan)).round(8)

    short_leg_close = pd.to_numeric(clean.get("short_leg_close"), errors="coerce")
    long_leg_close = pd.to_numeric(clean.get("long_leg_close"), errors="coerce")
    if short_leg_close.notna().any() or long_leg_close.notna().any():
        clean["short_leg_close"] = short_leg_close
        clean["long_leg_close"] = long_leg_close
        clean["short_leg_share_of_debit"] = (short_leg_close / current_debit.replace(0.0, np.nan)).round(8)
        clean["long_leg_share_of_debit"] = (long_leg_close / current_debit.replace(0.0, np.nan)).round(8)

    short_leg_volume = pd.to_numeric(clean.get("short_leg_volume"), errors="coerce")
    long_leg_volume = pd.to_numeric(clean.get("long_leg_volume"), errors="coerce")
    if short_leg_volume.notna().any() or long_leg_volume.notna().any():
        clean["short_leg_volume"] = short_leg_volume
        clean["long_leg_volume"] = long_leg_volume
        clean["leg_volume_imbalance"] = (
            (short_leg_volume - long_leg_volume)
            / (short_leg_volume + long_leg_volume).replace(0.0, np.nan)
        ).round(8)

    short_leg_trade_count = pd.to_numeric(clean.get("short_leg_trade_count"), errors="coerce")
    long_leg_trade_count = pd.to_numeric(clean.get("long_leg_trade_count"), errors="coerce")
    if short_leg_trade_count.notna().any() or long_leg_trade_count.notna().any():
        clean["short_leg_trade_count"] = short_leg_trade_count
        clean["long_leg_trade_count"] = long_leg_trade_count
        clean["leg_trade_count_imbalance"] = (
            (short_leg_trade_count - long_leg_trade_count)
            / (short_leg_trade_count + long_leg_trade_count).replace(0.0, np.nan)
        ).round(8)

    trend = clean.get("market_trend_regime", pd.Series(index=clean.index, dtype=object)).astype("string").str.lower()
    clean["market_trend_uptrend"] = (trend == "uptrend").astype(float)
    clean["market_trend_sideways"] = (trend == "sideways").astype(float)
    clean["market_trend_downtrend"] = (trend == "downtrend").astype(float)

    vol_regime = clean.get("market_volatility_regime", pd.Series(index=clean.index, dtype=object)).astype("string").str.lower()
    clean["market_volatility_low"] = (vol_regime == "low").astype(float)
    clean["market_volatility_medium"] = (vol_regime == "medium").astype(float)
    clean["market_volatility_high"] = (vol_regime == "high").astype(float)
    return clean


def _select_feature_columns(
    df: pd.DataFrame,
    *,
    include_features: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> list[str]:
    include_set = set(include_features) if include_features else None
    exclude_set = set(exclude_features or [])
    columns: list[str] = []
    for column in ALL_FEATURE_COLUMNS:
        if include_set is not None and column not in include_set:
            continue
        if column in exclude_set:
            continue
        if column in df and pd.to_numeric(df[column], errors="coerce").notna().any():
            columns.append(column)
    return columns


def _group_index(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = df.copy()
    keyed["_group_key"] = keyed.apply(_trade_group_key, axis=1)
    groups = (
        keyed.groupby("_group_key", as_index=False)
        .agg(
            entry_timestamp=("entry_timestamp", "min"),
            state_timestamp=("state_timestamp", "min"),
            rows=("entry_timestamp", "size"),
        )
        .sort_values(["entry_timestamp", "state_timestamp", "_group_key"])
        .reset_index(drop=True)
        .rename(columns={"_group_key": "group_key"})
    )
    order = {key: idx for idx, key in enumerate(groups["group_key"])}
    keyed["_group_order"] = keyed["_group_key"].map(order)
    keyed.sort_values(["_group_order", "state_timestamp"], inplace=True)
    keyed.reset_index(drop=True, inplace=True)
    return keyed, groups


def _trade_group_key(row: pd.Series) -> str:
    return "||".join(
        [
            str(row.get("underlying") or ""),
            pd.Timestamp(row["entry_timestamp"]).isoformat(),
            str(row.get("short_option_symbol") or ""),
            str(row.get("long_option_symbol") or ""),
        ]
    )


def _split_index(count: int, fraction: float) -> int:
    test_rows = max(1, int(round(count * fraction))) if count > 1 else 0
    return max(1, count - test_rows)


def _group_holdout_split(
    groups: pd.DataFrame,
    *,
    test_fraction: float,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    split_group_index = _split_index(len(groups), test_fraction)
    raw_test_groups = groups.iloc[split_group_index:].copy()
    train_groups = groups.iloc[:split_group_index].copy()
    if embargo_days > 0 and not raw_test_groups.empty:
        test_start_ts = pd.to_datetime(raw_test_groups["entry_timestamp"], utc=True, errors="coerce").iloc[0]
        cutoff = test_start_ts - pd.Timedelta(days=embargo_days)
        train_ts = pd.to_datetime(train_groups["entry_timestamp"], utc=True, errors="coerce")
        train_groups = train_groups.loc[train_ts < cutoff].copy()
    return train_groups, raw_test_groups, _group_split_summary(train_groups, raw_test_groups, embargo_days=embargo_days)


def _group_split_summary(
    train_groups: pd.DataFrame,
    test_groups: pd.DataFrame,
    *,
    embargo_days: int,
) -> dict[str, Any]:
    train_start = _group_timestamp(train_groups, 0)
    train_end = _group_timestamp(train_groups, len(train_groups) - 1)
    test_start = _group_timestamp(test_groups, 0)
    test_end = _group_timestamp(test_groups, len(test_groups) - 1)
    gap_days = None
    if train_end is not None and test_start is not None:
        gap_days = round((pd.Timestamp(test_start) - pd.Timestamp(train_end)).total_seconds() / 86400.0, 6)
    return {
        "strategy": "grouped_chronological_timestamp_embargo",
        "embargo_days": int(embargo_days),
        "train_groups": int(len(train_groups)),
        "test_groups": int(len(test_groups)),
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "actual_gap_days": gap_days,
    }


def _group_timestamp(groups: pd.DataFrame, index: int) -> str | None:
    if groups.empty or index < 0 or index >= len(groups):
        return None
    ts = pd.to_datetime(groups.iloc[index]["entry_timestamp"], utc=True, errors="coerce")
    return ts.isoformat() if pd.notna(ts) else None


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
    return df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(fill_values)


def _build_dmatrix(df: pd.DataFrame, feature_columns: list[str], fill_values: dict[str, float], labels: np.ndarray):
    frame = _transform_frame(df, feature_columns, fill_values)
    return xgb.DMatrix(frame, label=labels, feature_names=feature_columns)


def _predict_prob(booster, frame: pd.DataFrame) -> np.ndarray:
    if len(frame) == 0:
        return np.array([])
    matrix = xgb.DMatrix(frame, feature_names=list(frame.columns))
    return booster.predict(matrix)


def _choose_close_threshold(
    df: pd.DataFrame,
    *,
    target_column: str,
    y_prob: np.ndarray,
    min_recall: float,
    max_close_rate: float | None = None,
    max_false_close_rate: float | None = None,
) -> float:
    best_threshold = 0.5
    best_key: tuple[float, float, float, float, float, float, float] | None = None
    y_true = df[target_column].to_numpy(dtype=float)
    for threshold in np.arange(0.05, 0.951, 0.05):
        metrics = _clf_metrics(y_true, y_prob, float(threshold))
        policy = _policy_metrics(df, target_column=target_column, y_prob=y_prob, threshold=float(threshold))
        recall = float(metrics.get("recall") or 0.0)
        precision = float(metrics.get("precision") or 0.0)
        f2 = float(metrics.get("f2") or 0.0)
        close_rate = float(metrics.get("close_rate") or 0.0)
        false_close_rate = float(policy.get("false_close_rate") or 0.0)
        tp_lead_minutes = float(policy.get("mean_true_positive_minutes_to_exit") or 0.0)
        meets_recall = 1.0 if recall >= min_recall else 0.0
        meets_close = 1.0 if max_close_rate is None or close_rate <= max_close_rate else 0.0
        meets_false_close = 1.0 if (
            max_false_close_rate is None or false_close_rate <= max_false_close_rate
        ) else 0.0
        key = (
            meets_recall + meets_close + meets_false_close,
            meets_recall,
            meets_false_close,
            meets_close,
            f2,
            precision,
            tp_lead_minutes - false_close_rate - close_rate,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _clf_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)
    auc = _auc_rank(y_true, y_prob)
    return {
        "rows": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 6),
        "threshold": round(float(threshold), 6),
        "precision": _round_or_none(precision),
        "recall": _round_or_none(recall),
        "f1": _round_or_none(f1),
        "f2": _round_or_none(f2),
        "auc": _round_or_none(auc),
        "close_rate": round(float(np.mean(y_pred)), 6) if len(y_pred) else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _policy_metrics(
    df: pd.DataFrame,
    *,
    target_column: str,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_true = df[target_column].to_numpy(dtype=float)
    close_flag = y_prob >= threshold
    profit_take_col = _profit_take_column(target_column)
    profit_take = pd.to_numeric(df.get(profit_take_col), errors="coerce").fillna(0).to_numpy(dtype=float)
    pnl = pd.to_numeric(df.get("pnl_per_contract"), errors="coerce").to_numpy(dtype=float)
    minutes_to_exit = pd.to_numeric(df.get("minutes_to_exit"), errors="coerce").to_numpy(dtype=float)
    return {
        "close_rate": _round_or_none(np.mean(close_flag) if len(close_flag) else None),
        "false_close_rate": _round_or_none(np.mean(close_flag & (y_true == 0)) if len(close_flag) else None),
        "profit_take_false_close_rate": _round_or_none(
            np.mean(close_flag & (y_true == 0) & (profit_take == 1)) if len(close_flag) else None
        ),
        "mean_true_positive_minutes_to_exit": _round_or_none(
            np.mean(minutes_to_exit[close_flag & (y_true == 1)])
            if np.any(close_flag & (y_true == 1))
            else None
        ),
        "mean_false_positive_minutes_to_exit": _round_or_none(
            np.mean(minutes_to_exit[close_flag & (y_true == 0)])
            if np.any(close_flag & (y_true == 0))
            else None
        ),
        "mean_flagged_pnl_per_contract": _round_or_none(np.mean(pnl[close_flag]) if np.any(close_flag) else None),
        "mean_missed_risk_pnl_per_contract": _round_or_none(
            np.mean(pnl[(~close_flag) & (y_true == 1)]) if np.any((~close_flag) & (y_true == 1)) else None
        ),
        "mean_flagged_minutes_to_exit": _round_or_none(
            np.mean(minutes_to_exit[close_flag]) if np.any(close_flag) else None
        ),
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _prefixed_policy_metrics(
    prefix: str,
    df: pd.DataFrame,
    target_column: str,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in _policy_metrics(df, target_column=target_column, y_prob=y_prob, threshold=threshold).items()
    }


def _walk_forward_grouped(
    df: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: list[str],
    fold_count: int,
    min_train_groups: int,
    params: dict[str, Any],
    num_boost_round: int,
    embargo_days: int,
    val_fraction: float,
    early_stopping_rounds: int,
    fixed_threshold: float | None = None,
    min_threshold_recall: float,
    max_threshold_close_rate: float | None,
    max_threshold_false_close_rate: float | None,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for fold_number, train_start, train_end, test_start, test_end in _group_walk_forward_splits_by_timestamp(
        groups,
        len(groups),
        fold_count=fold_count,
        min_train_groups=min_train_groups,
        embargo_days=embargo_days,
    ):
        train_group_keys = set(groups.iloc[train_start:train_end]["group_key"])
        test_group_keys = set(groups.iloc[test_start:test_end]["group_key"])
        train_df = df[df["_group_key"].isin(train_group_keys)].copy()
        test_df = df[df["_group_key"].isin(test_group_keys)].copy()
        if train_df.empty or test_df.empty:
            continue

        train_group_count = train_end - train_start
        val_group_index = (
            _split_index(train_group_count, val_fraction)
            if val_fraction > 0 and train_group_count > 1
            else train_group_count
        )
        val_group_keys = set(groups.iloc[train_start + val_group_index:train_end]["group_key"])
        train_sub_group_keys = set(groups.iloc[train_start:train_start + val_group_index]["group_key"])
        train_sub_df = df[df["_group_key"].isin(train_sub_group_keys)].copy()
        val_df = df[df["_group_key"].isin(val_group_keys)].copy()

        fill_values = _compute_fill_values(train_df, feature_columns)
        dtrain = _build_dmatrix(
            train_sub_df,
            feature_columns,
            fill_values,
            train_sub_df[target_column].to_numpy(dtype=float),
        )
        dval = _build_dmatrix(
            val_df,
            feature_columns,
            fill_values,
            val_df[target_column].to_numpy(dtype=float),
        )
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")] if dval.num_row() > 0 and early_stopping_rounds > 0 else [],
            early_stopping_rounds=early_stopping_rounds if dval.num_row() > 0 and early_stopping_rounds > 0 else None,
            verbose_eval=False,
        )

        if fixed_threshold is not None:
            threshold = fixed_threshold
        else:
            threshold_df = val_df if len(val_df) else train_sub_df
            threshold_prob = _predict_prob(booster, _transform_frame(threshold_df, feature_columns, fill_values))
            threshold = _choose_close_threshold(
                threshold_df,
                target_column=target_column,
                y_prob=threshold_prob,
                min_recall=min_threshold_recall,
                max_close_rate=max_threshold_close_rate,
                max_false_close_rate=max_threshold_false_close_rate,
            )
        test_prob = _predict_prob(booster, _transform_frame(test_df, feature_columns, fill_values))
        fold_metrics = _clf_metrics(test_df[target_column].to_numpy(dtype=float), test_prob, threshold)
        fold_metrics.update(_policy_metrics(test_df, target_column=target_column, y_prob=test_prob, threshold=threshold))
        folds.append(
            {
                "fold": fold_number,
                "train_start": groups.iloc[train_start]["entry_timestamp"].isoformat(),
                "train_end": groups.iloc[train_end - 1]["entry_timestamp"].isoformat(),
                "embargo_groups": int(test_start - train_end),
                "actual_gap_days": _group_gap_days(groups, train_end - 1, test_start),
                "test_start": groups.iloc[test_start]["entry_timestamp"].isoformat(),
                "test_end": groups.iloc[test_end - 1]["entry_timestamp"].isoformat(),
                "train_groups": int(train_end - train_start),
                "test_groups": int(test_end - test_start),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "threshold": round(float(threshold), 6),
                "metrics": fold_metrics,
            }
        )
    return folds


def _group_walk_forward_splits(
    group_count: int,
    *,
    fold_count: int,
    min_train_groups: int,
    embargo_groups: int = 0,
) -> list[tuple[int, int, int, int, int]]:
    if fold_count <= 0 or group_count <= min_train_groups:
        return []
    test_indices = np.array_split(
        np.arange(min_train_groups, group_count),
        min(fold_count, group_count - min_train_groups),
    )
    splits: list[tuple[int, int, int, int, int]] = []
    for fold_number, chunk in enumerate(test_indices, start=1):
        if len(chunk) == 0:
            continue
        train_end = int(chunk[0])
        test_start = min(train_end + embargo_groups, int(chunk[-1]) + 1)
        test_end = int(chunk[-1]) + 1
        if test_start >= test_end:
            continue
        splits.append((fold_number, 0, train_end, test_start, test_end))
    return splits


def _group_walk_forward_splits_by_timestamp(
    groups: pd.DataFrame,
    group_count: int,
    *,
    fold_count: int,
    min_train_groups: int,
    embargo_days: int = 0,
) -> list[tuple[int, int, int, int, int]]:
    if fold_count <= 0 or group_count <= min_train_groups:
        return []
    test_indices = np.array_split(
        np.arange(min_train_groups, group_count),
        min(fold_count, group_count - min_train_groups),
    )
    if embargo_days <= 0:
        return _group_walk_forward_splits(
            group_count,
            fold_count=fold_count,
            min_train_groups=min_train_groups,
            embargo_groups=0,
        )
    timestamps = pd.to_datetime(groups["entry_timestamp"], utc=True, errors="coerce")
    splits: list[tuple[int, int, int, int, int]] = []
    for fold_number, chunk in enumerate(test_indices, start=1):
        if len(chunk) == 0:
            continue
        test_start = int(chunk[0])
        test_end = int(chunk[-1]) + 1
        cutoff = timestamps.iloc[test_start] - pd.Timedelta(days=embargo_days)
        train_end = test_start
        for idx in range(test_start - 1, -1, -1):
            ts = timestamps.iloc[idx]
            if pd.notna(ts) and ts < cutoff:
                train_end = idx + 1
                break
        else:
            train_end = 0
        if train_end < min_train_groups:
            continue
        splits.append((fold_number, 0, train_end, test_start, test_end))
    return splits


def _group_embargo_from_days(groups: pd.DataFrame, embargo_days: int) -> int:
    if embargo_days <= 0 or len(groups) < 2:
        return 0
    timestamps = pd.to_datetime(groups["entry_timestamp"], utc=True, errors="coerce").dropna().sort_values()
    if len(timestamps) < 2:
        return 0
    total_days = max(1.0, (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86400.0)
    groups_per_day = len(timestamps) / total_days
    return max(0, int(np.ceil(groups_per_day * embargo_days)))


def _group_gap_days(groups: pd.DataFrame, train_end_idx: int, test_start_idx: int) -> float | None:
    if train_end_idx < 0 or test_start_idx < 0:
        return None
    timestamps = pd.to_datetime(groups["entry_timestamp"], utc=True, errors="coerce")
    if train_end_idx >= len(timestamps) or test_start_idx >= len(timestamps):
        return None
    train_end = timestamps.iloc[train_end_idx]
    test_start = timestamps.iloc[test_start_idx]
    if pd.isna(train_end) or pd.isna(test_start):
        return None
    return round((test_start - train_end).total_seconds() / 86400.0, 6)


def _walk_forward_summary(folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"walk_forward_folds": int(len(folds))}
    if not folds:
        return summary
    mean_keys = [
        "auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "close_rate",
        "false_close_rate",
        "profit_take_false_close_rate",
        "mean_true_positive_minutes_to_exit",
        "mean_false_positive_minutes_to_exit",
    ]
    for key in mean_keys:
        values = [fold["metrics"].get(key) for fold in folds]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        summary[f"walk_forward_{key}_mean"] = _round_or_none(np.mean(numeric) if numeric else None)
        if key in {"auc", "recall", "f2"}:
            summary[f"walk_forward_{key}_min"] = _round_or_none(np.min(numeric) if numeric else None)
    summary["walk_forward_test_rows"] = int(sum(fold["test_rows"] for fold in folds))
    summary["walk_forward_test_groups"] = int(sum(fold["test_groups"] for fold in folds))
    return summary


def _profit_take_column(target_column: str) -> str:
    suffix = target_column.split("_")[-1]
    return f"profit_take_hit_{suffix}"


def _fbeta(precision: float | None, recall: float | None, *, beta: float) -> float | None:
    if precision is None or recall is None or precision <= 0 or recall <= 0:
        return None
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom <= 0:
        return None
    return ((1.0 + beta_sq) * precision * recall) / denom


def _auc_rank(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
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
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _round_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _parse_feature_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
