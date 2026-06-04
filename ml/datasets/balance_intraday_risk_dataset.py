"""Materialize a balanced training view of intraday risk dataset rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.datasets.balance_candidate_dataset import (
    _mode_value,
    _next_seed,
    _present_group_columns,
    _sample_frame,
    _sample_within_subgroups,
    _allocate_quotas,
)
from ml.storage import ParquetDatasetWriter


@dataclass(frozen=True)
class IntradayRiskBalanceConfig:
    target_rows: int = 1_000_000
    group_columns: tuple[str, ...] = (
        "underlying",
        "market_volatility_regime",
        "market_trend_regime",
        "intraday_exit_reason",
        "entry_month",
    )
    max_underlying_share: float = 0.20
    max_oversample_factor: float = 1.0
    random_seed: int = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced parquet training view from intraday risk rows.")
    parser.add_argument("--input", required=True, help="Source intraday risk dataset directory, parquet, or JSONL.")
    parser.add_argument("--dataset-version", required=True, help="Output dataset version.")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--target-rows", type=int, default=IntradayRiskBalanceConfig.target_rows)
    parser.add_argument(
        "--group-columns",
        default="underlying,market_volatility_regime,market_trend_regime,intraday_exit_reason,entry_month",
        help="Comma-separated columns used for hierarchical balancing.",
    )
    parser.add_argument("--max-underlying-share", type=float, default=IntradayRiskBalanceConfig.max_underlying_share)
    parser.add_argument("--max-oversample-factor", type=float, default=IntradayRiskBalanceConfig.max_oversample_factor)
    parser.add_argument("--random-seed", type=int, default=IntradayRiskBalanceConfig.random_seed)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = IntradayRiskBalanceConfig(
        target_rows=args.target_rows,
        group_columns=tuple(item.strip() for item in args.group_columns.split(",") if item.strip()),
        max_underlying_share=args.max_underlying_share,
        max_oversample_factor=args.max_oversample_factor,
        random_seed=args.random_seed,
    )
    source_path = Path(args.input)
    df = load_dataset(source_path)
    balanced = balance_intraday_risk_frame(df, cfg)
    manifest = df.attrs.get("dataset_manifest") if hasattr(df, "attrs") else None
    source_metadata = dict((manifest or {}).get("metadata") or {})
    metadata = {
        **source_metadata,
        "balanced_from": str(source_path),
        "balance_method": "sqrt_frequency_hierarchical_sampling",
        "balance_group_columns": list(_present_group_columns(balanced, cfg.group_columns)),
        "balance_target_rows": cfg.target_rows,
        "balance_max_underlying_share": cfg.max_underlying_share,
        "balance_max_oversample_factor": cfg.max_oversample_factor,
        "balance_random_seed": cfg.random_seed,
        "label_version": source_metadata.get("label_version", _mode_value(balanced, "label_version")),
    }
    result = ParquetDatasetWriter(root_dir=args.output_dir).write(
        balanced.to_dict("records"),
        dataset_version=args.dataset_version,
        dataset_type="intraday_risk_rows",
        schema_columns=list(balanced.columns),
        metadata=metadata,
    )
    print(json.dumps({
        "root_path": str(result.root_path),
        "manifest_path": str(result.manifest_path),
        "row_count": result.row_count,
        "underlying_counts": balanced["underlying"].value_counts().to_dict() if "underlying" in balanced else {},
    }, indent=2, sort_keys=True))
    return 0


def balance_intraday_risk_frame(
    df: pd.DataFrame,
    config: IntradayRiskBalanceConfig,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Cannot balance an empty dataset.")
    if config.target_rows <= 0:
        raise ValueError("target_rows must be positive.")
    if config.max_underlying_share <= 0 or config.max_underlying_share > 1:
        raise ValueError("max_underlying_share must be in (0, 1].")
    if config.max_oversample_factor < 1:
        raise ValueError("max_oversample_factor must be >= 1.")

    clean = df.copy()
    if "entry_timestamp" in clean:
        clean["entry_timestamp"] = pd.to_datetime(clean["entry_timestamp"], utc=True, errors="coerce")
        clean = clean.sort_values(["entry_timestamp", "state_timestamp"], na_position="last").reset_index(drop=True)
    added_entry_date = False
    added_entry_month = False
    if "entry_date" not in clean and "entry_timestamp" in clean:
        clean["entry_date"] = clean["entry_timestamp"].dt.date.astype(str)
        added_entry_date = True
    if "entry_month" not in clean and "entry_timestamp" in clean:
        clean["entry_month"] = clean["entry_timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
        added_entry_month = True

    orig_column = "__balance_orig_idx__"
    clean[orig_column] = range(len(clean))

    group_columns = _present_group_columns(clean, config.group_columns)
    if not group_columns:
        result = _sample_frame(clean, min(config.target_rows, len(clean)), config.random_seed)
        return result.drop(columns=[orig_column], errors="ignore")

    primary = "underlying" if "underlying" in group_columns else group_columns[0]
    primary_counts = clean[primary].astype(str).value_counts()
    primary_caps = {
        value: max(1, int(config.target_rows * config.max_underlying_share))
        for value in primary_counts.index
    }
    primary_quota = _allocate_quotas(
        primary_counts,
        config.target_rows,
        caps=primary_caps,
        max_oversample_factor=config.max_oversample_factor,
    )

    rng = np.random.default_rng(config.random_seed)
    sampled_parts: list[pd.DataFrame] = []
    subgroup_columns = tuple(column for column in group_columns if column != primary)
    for value, quota in primary_quota.items():
        if quota <= 0:
            continue
        group = clean[clean[primary].astype(str) == str(value)]
        if subgroup_columns:
            sampled_parts.append(_sample_within_subgroups(group, subgroup_columns, quota, rng, config))
        else:
            sampled_parts.append(_sample_frame(group, quota, _next_seed(rng)))

    balanced = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else clean.iloc[:0].copy()
    if len(balanced) != config.target_rows:
        delta = config.target_rows - len(balanced)
        if delta > 0:
            if config.max_oversample_factor <= 1.0 and orig_column in balanced.columns:
                used = set(balanced[orig_column].dropna().astype(int))
                unused = clean[~clean[orig_column].isin(used)]
                fill = min(delta, len(unused))
                if fill > 0:
                    balanced = pd.concat(
                        [balanced, unused.sample(n=fill, replace=False, random_state=_next_seed(rng))],
                        ignore_index=True,
                    )
            else:
                balanced = pd.concat([balanced, _sample_frame(clean, delta, _next_seed(rng))], ignore_index=True)
        else:
            balanced = balanced.sample(n=config.target_rows, random_state=_next_seed(rng)).reset_index(drop=True)
    if "entry_timestamp" in balanced:
        balanced = balanced.sort_values(["entry_timestamp", "state_timestamp"], na_position="last").reset_index(drop=True)
    drop_columns = [orig_column]
    if added_entry_month:
        drop_columns.append("entry_month")
    if added_entry_date and "entry_date" not in df.columns:
        drop_columns.append("entry_date")
    return balanced.drop(columns=drop_columns, errors="ignore")


if __name__ == "__main__":
    raise SystemExit(main())
