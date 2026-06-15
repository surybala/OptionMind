"""Shared quality gates for candidate training rows."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CandidateQualityFilterConfig:
    min_max_loss_dollars: float | None = 25.0
    max_credit_to_width: float | None = 0.90
    min_short_leg_volume: float | None = 5.0
    min_long_leg_volume: float | None = 5.0
    min_short_leg_trade_count: int | None = 2
    min_long_leg_trade_count: int | None = 2

    def to_metadata(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def apply_candidate_quality_filters(
    df: pd.DataFrame,
    config: CandidateQualityFilterConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a filtered frame plus stable audit stats for the applied gates."""
    cfg = config or CandidateQualityFilterConfig()
    if df.empty:
        return df.copy(), _stats_payload(df, pd.Series(dtype=bool), {}, cfg)

    clean = df.copy()
    drop_reasons: dict[str, int] = {}
    keep_mask = pd.Series(True, index=clean.index, dtype=bool)

    keep_mask &= _apply_numeric_lower_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="max_loss",
        minimum=cfg.min_max_loss_dollars,
        reason="min_max_loss_dollars",
    )
    keep_mask &= _apply_numeric_upper_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="credit_to_width",
        maximum=cfg.max_credit_to_width,
        reason="max_credit_to_width",
    )
    keep_mask &= _apply_numeric_lower_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="option_entry_volume",
        minimum=cfg.min_short_leg_volume,
        reason="min_short_leg_volume",
    )
    keep_mask &= _apply_numeric_lower_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="long_option_entry_volume",
        minimum=cfg.min_long_leg_volume,
        reason="min_long_leg_volume",
    )
    keep_mask &= _apply_numeric_lower_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="option_entry_trade_count",
        minimum=cfg.min_short_leg_trade_count,
        reason="min_short_leg_trade_count",
    )
    keep_mask &= _apply_numeric_lower_bound(
        clean,
        keep_mask,
        drop_reasons,
        column="long_option_entry_trade_count",
        minimum=cfg.min_long_leg_trade_count,
        reason="min_long_leg_trade_count",
    )

    filtered = clean.loc[keep_mask].copy()
    if "entry_timestamp" in filtered.columns:
        filtered = filtered.sort_values("entry_timestamp").reset_index(drop=True)
    stats = _stats_payload(clean, keep_mask, drop_reasons, cfg)
    filtered.attrs.update(clean.attrs)
    filtered.attrs["candidate_quality_filter_stats"] = stats
    return filtered, stats


def _apply_numeric_lower_bound(
    df: pd.DataFrame,
    current_keep_mask: pd.Series,
    drop_reasons: dict[str, int],
    *,
    column: str,
    minimum: float | int | None,
    reason: str,
) -> pd.Series:
    if minimum is None or column not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    numeric = pd.to_numeric(df[column], errors="coerce")
    mask = numeric.notna() & (numeric >= float(minimum))
    drop_reasons[reason] = int((current_keep_mask & ~mask).sum())
    return mask


def _apply_numeric_upper_bound(
    df: pd.DataFrame,
    current_keep_mask: pd.Series,
    drop_reasons: dict[str, int],
    *,
    column: str,
    maximum: float | None,
    reason: str,
) -> pd.Series:
    if maximum is None or column not in df.columns:
        return pd.Series(True, index=df.index, dtype=bool)
    numeric = pd.to_numeric(df[column], errors="coerce")
    mask = numeric.notna() & (numeric <= float(maximum))
    drop_reasons[reason] = int((current_keep_mask & ~mask).sum())
    return mask


def _stats_payload(
    input_df: pd.DataFrame,
    keep_mask: pd.Series,
    drop_reasons: dict[str, int],
    config: CandidateQualityFilterConfig,
) -> dict[str, Any]:
    kept_rows = int(keep_mask.sum()) if len(keep_mask) else int(len(input_df))
    return {
        "config": config.to_metadata(),
        "input_rows": int(len(input_df)),
        "output_rows": kept_rows,
        "dropped_rows": int(len(input_df) - kept_rows),
        "drop_reasons": {key: value for key, value in drop_reasons.items() if value > 0},
        "config_json": json.dumps(config.to_metadata(), sort_keys=True),
    }
