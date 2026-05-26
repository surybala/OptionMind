"""Materialize a balanced training view of candidate dataset rows."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.storage import ParquetDatasetWriter


@dataclass(frozen=True)
class BalanceConfig:
    target_rows: int = 500_000
    group_columns: tuple[str, ...] = ("underlying", "market_volatility_regime", "market_trend_regime")
    max_underlying_share: float = 0.25
    max_oversample_factor: float = 1.0
    random_seed: int = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced parquet training view from candidate rows.")
    parser.add_argument("--input", required=True, help="Source candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--dataset-version", required=True, help="Output dataset version.")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--target-rows", type=int, default=BalanceConfig.target_rows)
    parser.add_argument(
        "--group-columns",
        default="underlying,market_volatility_regime,market_trend_regime",
        help="Comma-separated columns used for hierarchical balancing.",
    )
    parser.add_argument("--max-underlying-share", type=float, default=BalanceConfig.max_underlying_share)
    parser.add_argument("--max-oversample-factor", type=float, default=BalanceConfig.max_oversample_factor)
    parser.add_argument("--random-seed", type=int, default=BalanceConfig.random_seed)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BalanceConfig(
        target_rows=args.target_rows,
        group_columns=tuple(item.strip() for item in args.group_columns.split(",") if item.strip()),
        max_underlying_share=args.max_underlying_share,
        max_oversample_factor=args.max_oversample_factor,
        random_seed=args.random_seed,
    )
    source_path = Path(args.input)
    df = load_dataset(source_path)
    balanced = balance_candidate_frame(df, cfg)
    manifest = df.attrs.get("dataset_manifest") if hasattr(df, "attrs") else None
    source_metadata = dict((manifest or {}).get("metadata") or {})
    metadata = {
        **source_metadata,
        "balanced_from": str(source_path),
        "balance_method": "sqrt_frequency_hierarchical_sampling",
        "balance_group_columns": list(_present_group_columns(df, cfg.group_columns)),
        "balance_target_rows": cfg.target_rows,
        "balance_max_underlying_share": cfg.max_underlying_share,
        "balance_max_oversample_factor": cfg.max_oversample_factor,
        "balance_random_seed": cfg.random_seed,
        "feature_set_version": source_metadata.get("feature_set_version", "features_v005"),
        "label_version": source_metadata.get("label_version", _mode_value(df, "label_version")),
    }
    result = ParquetDatasetWriter(root_dir=args.output_dir).write(
        balanced.to_dict("records"),
        dataset_version=args.dataset_version,
        dataset_type="candidate_rows",
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


def balance_candidate_frame(df: pd.DataFrame, config: BalanceConfig) -> pd.DataFrame:
    """Return a balanced sample with deterministic replacement where needed.

    The sampler deliberately uses square-root group frequencies instead of a
    strict uniform target. That reduces dominant ETFs without turning sparse
    symbols into thousands of cloned rows.

    When max_oversample_factor=1.0 (the default), sampling is always without
    replacement: no authentic row is ever duplicated. The top-up step after
    quota allocation is restricted to genuinely unsampled rows, so sparse
    underlyings cannot exceed their natural count.
    """
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
        clean = clean.sort_values("entry_timestamp").reset_index(drop=True)
    if "entry_date" not in clean and "entry_timestamp" in clean:
        clean["entry_date"] = pd.to_datetime(clean["entry_timestamp"], errors="coerce").dt.date.astype(str)

    # Sentinel column lets the no-oversample top-up find genuinely unused rows.
    _ORIG = "__balance_orig_idx__"
    clean[_ORIG] = range(len(clean))

    group_columns = _present_group_columns(clean, config.group_columns)
    if not group_columns:
        result = _sample_frame(clean, config.target_rows, config.random_seed)
        return result.drop(columns=[_ORIG], errors="ignore")

    primary = "underlying" if "underlying" in group_columns else group_columns[0]
    primary_counts = clean[primary].astype(str).value_counts()
    primary_caps = {
        value: max(1, int(np.floor(config.target_rows * config.max_underlying_share)))
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
            if config.max_oversample_factor <= 1.0 and _ORIG in balanced.columns:
                # No-oversample: top-up only from rows not yet drawn in the main pass.
                # _ORIG values are unique per row when replace=False, so set membership
                # gives exact "unsampled" rows without index aliasing.
                used = set(balanced[_ORIG].dropna().astype(int))
                unused = clean[~clean[_ORIG].isin(used)]
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
        balanced = balanced.sort_values("entry_timestamp").reset_index(drop=True)
    return balanced.drop(columns=[_ORIG], errors="ignore")


def _sample_within_subgroups(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    target_rows: int,
    rng: np.random.Generator,
    config: BalanceConfig,
) -> pd.DataFrame:
    key = df[list(columns)].astype(str).agg("||".join, axis=1)
    counts = key.value_counts()
    quotas = _allocate_quotas(counts, target_rows, max_oversample_factor=config.max_oversample_factor)
    parts: list[pd.DataFrame] = []
    for group_key, quota in quotas.items():
        if quota <= 0:
            continue
        parts.append(_sample_frame(df[key == group_key], quota, _next_seed(rng)))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[:0].copy()


def _allocate_quotas(
    counts: pd.Series,
    target_rows: int,
    *,
    caps: dict[Any, int] | None = None,
    max_oversample_factor: float = 3.0,
) -> dict[Any, int]:
    if counts.empty:
        return {}
    counts = counts.astype(int)
    weights = np.sqrt(counts.to_numpy(dtype=float))
    raw = weights / weights.sum() * target_rows
    quotas = {key: int(np.floor(value)) for key, value in zip(counts.index, raw)}
    capacities = {
        key: max(1, int(np.floor(count * max_oversample_factor)))
        for key, count in counts.items()
    }
    if caps:
        capacities = {key: min(capacities[key], int(caps.get(key, capacities[key]))) for key in capacities}
    quotas = {key: min(max(0, quotas.get(key, 0)), capacities[key]) for key in counts.index}

    remaining = target_rows - sum(quotas.values())
    if remaining > 0:
        fractional = {key: raw_value - np.floor(raw_value) for key, raw_value in zip(counts.index, raw)}
        order = sorted(counts.index, key=lambda key: (-fractional[key], str(key)))
        while remaining > 0:
            changed = False
            for key in order:
                if remaining <= 0:
                    break
                if quotas[key] < capacities[key]:
                    quotas[key] += 1
                    remaining -= 1
                    changed = True
            if not changed:
                break
    elif remaining < 0:
        order = sorted(quotas, key=lambda key: (-quotas[key], str(key)))
        while remaining < 0:
            for key in order:
                if remaining >= 0:
                    break
                if quotas[key] > 0:
                    quotas[key] -= 1
                    remaining += 1

    return quotas


def _sample_frame(df: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    if rows <= 0:
        return df.iloc[:0].copy()
    replace = rows > len(df)
    return df.sample(n=rows, replace=replace, random_state=seed).copy()


def _present_group_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(column for column in columns if column in df.columns)


def _mode_value(df: pd.DataFrame, column: str) -> str | None:
    if column not in df:
        return None
    values = df[column].dropna().astype(str)
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def _next_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


if __name__ == "__main__":
    raise SystemExit(main())
