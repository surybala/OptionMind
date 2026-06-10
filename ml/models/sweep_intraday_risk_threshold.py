"""Threshold sweep for the intraday risk monitor.

Loads the champion model, reconstructs the holdout test set, and evaluates
classification + policy metrics at a range of thresholds to visualise the
recall vs false-close trade-off.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.train_intraday_risk_monitor import (
    _choose_close_threshold,
    _clf_metrics,
    _engineer_intraday_features,
    _group_index,
    _policy_metrics,
    _predict_prob,
    _prepare_frame,
    _select_feature_columns,
    _split_index,
    _transform_frame,
)

try:
    import xgboost as xgb
except ImportError:
    print("xgboost required", file=sys.stderr)
    sys.exit(1)


ARTIFACT_PATH = Path("artifacts/models/intraday_risk_monitor_stop30m_fullraw_v003.json")
MODEL_PATH = Path("artifacts/models/intraday_risk_monitor_stop30m_fullraw_v003.xgboost.json")
DATASET_PATH = Path(
    "artifacts/datasets/intraday_risk_rows/"
    "dataset_version=intraday_risk_rows_parquet_regime_balanced_1m_broad_etfs_20220602_20260601_v001"
)
TARGET = "stop_loss_hit_30m"
TEST_FRACTION = 0.25


def main() -> int:
    artifact = json.loads(ARTIFACT_PATH.read_text())
    feature_columns: list[str] = artifact["feature_columns"]
    fill_values: dict[str, float] = artifact["fill_values"]

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = load_dataset(DATASET_PATH)
    clean = _prepare_frame(df, target_column=TARGET)
    clean = _engineer_intraday_features(clean)
    clean, groups = _group_index(clean)

    split_idx = _split_index(len(groups), TEST_FRACTION)
    test_groups = set(groups.iloc[split_idx:]["group_key"])
    test_df = clean[clean["_group_key"].isin(test_groups)].copy()

    print(f"Test set: {len(test_df):,} rows, {len(test_groups):,} entries, "
          f"positive rate {test_df[TARGET].mean():.4%}")

    x_test = _transform_frame(test_df, feature_columns, fill_values)
    y_prob = _predict_prob(booster, x_test)
    y_true = test_df[TARGET].to_numpy(dtype=float)

    thresholds = [round(t, 3) for t in np.arange(0.01, 0.51, 0.01)]
    rows = []
    for t in thresholds:
        clf = _clf_metrics(y_true, y_prob, t)
        pol = _policy_metrics(test_df, target_column=TARGET, y_prob=y_prob, threshold=t)
        rows.append({
            "threshold": t,
            "recall": clf["recall"],
            "precision": clf["precision"],
            "f1": clf["f1"],
            "f2": clf["f2"],
            "close_rate": clf["close_rate"],
            "false_close_rate": pol["false_close_rate"],
            "profit_take_false_close_rate": pol["profit_take_false_close_rate"],
            "tp": clf["tp"],
            "fp": clf["fp"],
            "fn": clf["fn"],
            "tp_lead_min": pol["mean_true_positive_minutes_to_exit"],
            "flagged_pnl": pol["mean_flagged_pnl_per_contract"],
            "missed_risk_pnl": pol["mean_missed_risk_pnl_per_contract"],
        })

    result_df = pd.DataFrame(rows)

    print("\n" + "=" * 120)
    print("INTRADAY RISK MONITOR — THRESHOLD SWEEP (test holdout)")
    print("=" * 120)
    print(f"{'Thresh':>7} | {'Recall':>7} | {'Prec':>7} | {'F1':>7} | {'F2':>7} | "
          f"{'Close%':>7} | {'FalseClose%':>11} | {'ProfTakeFalse%':>14} | "
          f"{'TP':>5} | {'FP':>6} | {'FN':>5} | {'TP Lead(min)':>12} | {'Flag PnL':>9}")
    print("-" * 120)
    for _, r in result_df.iterrows():
        print(f"{r['threshold']:7.2f} | {r['recall'] or 0:7.4f} | {r['precision'] or 0:7.4f} | "
              f"{r['f1'] or 0:7.4f} | {r['f2'] or 0:7.4f} | "
              f"{r['close_rate'] or 0:7.4f} | {r['false_close_rate'] or 0:11.4f} | "
              f"{r['profit_take_false_close_rate'] or 0:14.6f} | "
              f"{r['tp']:5.0f} | {r['fp']:6.0f} | {r['fn']:5.0f} | "
              f"{r['tp_lead_min'] or 0:12.1f} | {r['flagged_pnl'] or 0:9.2f}")
    print("=" * 120)

    current_t = 0.05
    current = result_df[result_df["threshold"] == current_t].iloc[0]
    print(f"\n>>> Current operating threshold: {current_t}")
    print(f"    Recall={current['recall']:.4f}, FalseClose%={current['false_close_rate']:.4f}, "
          f"TP={int(current['tp'])}, FP={int(current['fp'])}, FN={int(current['fn'])}")

    for candidate in [0.08, 0.10, 0.12, 0.15]:
        c = result_df[result_df["threshold"] == candidate].iloc[0]
        recall_delta = (c["recall"] or 0) - (current["recall"] or 0)
        fc_delta = (c["false_close_rate"] or 0) - (current["false_close_rate"] or 0)
        print(f"\n    If threshold → {candidate:.2f}:")
        print(f"      Recall {current['recall']:.4f} → {c['recall']:.4f}  ({recall_delta:+.4f})")
        print(f"      FalseClose% {current['false_close_rate']:.4f} → {c['false_close_rate']:.4f}  ({fc_delta:+.4f})")
        print(f"      TP {int(current['tp'])} → {int(c['tp'])}, FP {int(current['fp'])} → {int(c['fp'])}, "
              f"FN {int(current['fn'])} → {int(c['fn'])}")

    out_path = Path("artifacts/models/intraday_risk_threshold_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nFull sweep saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
