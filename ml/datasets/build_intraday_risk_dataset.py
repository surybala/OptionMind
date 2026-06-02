"""CLI for building an intraday spread-state dataset for live risk models."""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ml.datasets.intraday_risk_dataset import (
    IntradayRiskDatasetBuilder,
    IntradayRiskDatasetConfig,
    IntradayRiskRow,
)
from ml.providers import MassiveProvider
from ml.storage import ParquetDatasetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minute-level spread-state rows for intraday risk model training.")
    parser.add_argument("--input", required=True, help="Seed candidate_rows dataset directory, parquet, or JSONL.")
    parser.add_argument("--provider", default="massive", choices=["massive"], help="Historical provider for intraday stock/option bars.")
    parser.add_argument("--entry-start", default=None, help="Optional seed-entry lower bound, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--entry-end", default=None, help="Optional seed-entry upper bound, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--strategy-types", default="PCS,CCS", help="Comma-separated spread strategies to include.")
    parser.add_argument("--option-timeframe", default="1Min", help="Intraday option-bar timeframe, default 1Min.")
    parser.add_argument("--stock-timeframe", default="1Min", help="Intraday stock-bar timeframe, default 1Min.")
    parser.add_argument("--lookback-minutes", type=int, default=30, help="Trailing underlying lookback retained for state features.")
    parser.add_argument("--sample-every-n-candidates", type=int, default=1, help="Downsample the seed candidate list before intraday fetches.")
    parser.add_argument("--sample-every-n-minutes", type=int, default=5, help="Keep every Nth minute-state row per candidate.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap on processed seed candidates.")
    parser.add_argument("--max-forward-days", type=int, default=30, help="Cap intraday fetches at entry + this many days.")
    parser.add_argument("--min-state-rows-per-candidate", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--dataset-version", default="intraday_risk_v001")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--min-output-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    if args.provider != "massive":
        raise ValueError(f"Unsupported provider: {args.provider}")
    provider = MassiveProvider.from_env()
    config = IntradayRiskDatasetConfig(
        option_timeframe=args.option_timeframe,
        stock_timeframe=args.stock_timeframe,
        lookback_minutes=args.lookback_minutes,
        sample_every_n_candidates=args.sample_every_n_candidates,
        sample_every_n_minutes=args.sample_every_n_minutes,
        max_candidates=args.max_candidates,
        max_forward_days=args.max_forward_days,
        min_state_rows_per_candidate=args.min_state_rows_per_candidate,
        max_workers=args.max_workers,
    )
    rows = IntradayRiskDatasetBuilder(provider, provider).build(
        Path(args.input),
        config,
        entry_start=_parse_datetime(args.entry_start) if args.entry_start else None,
        entry_end=_parse_datetime(args.entry_end, end_of_day=True) if args.entry_end else None,
        strategy_types=tuple(item.strip().upper() for item in args.strategy_types.split(",") if item.strip()),
    )
    result = ParquetDatasetWriter(root_dir=args.output_dir).write(
        rows,
        dataset_version=args.dataset_version,
        dataset_type="intraday_risk_rows",
        schema_columns=[field.name for field in fields(IntradayRiskRow)] + ["entry_date"],
        metadata={
            "provider": args.provider,
            "input": args.input,
            "entry_start": args.entry_start,
            "entry_end": args.entry_end,
            "strategy_types": [item.strip().upper() for item in args.strategy_types.split(",") if item.strip()],
            "option_timeframe": args.option_timeframe,
            "stock_timeframe": args.stock_timeframe,
            "lookback_minutes": args.lookback_minutes,
            "sample_every_n_candidates": args.sample_every_n_candidates,
            "sample_every_n_minutes": args.sample_every_n_minutes,
            "max_candidates": args.max_candidates,
            "max_forward_days": args.max_forward_days,
            "min_state_rows_per_candidate": args.min_state_rows_per_candidate,
            "min_output_rows": args.min_output_rows,
        },
        append=args.append,
    )
    print(f"Wrote {result.row_count} row(s) to {result.root_path}")
    print(f"Manifest: {result.manifest_path}")
    if args.min_output_rows and result.row_count < args.min_output_rows:
        print(
            f"ERROR: dataset produced {result.row_count} row(s), below --min-output-rows={args.min_output_rows}",
        )
        return 2
    return 0


def _parse_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    if len(value) == 10:
        parsed_date = datetime.fromisoformat(value).replace(tzinfo=UTC)
        if end_of_day:
            return parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        return parsed_date
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
