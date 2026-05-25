"""CLI for building a prototype historical candidate dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.datasets.candidate_dataset import CandidateDatasetRow
from ml.providers import AlpacaProvider, MassiveProvider
from ml.storage import ParquetDatasetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype option candidate dataset rows.")
    parser.add_argument("--provider", default="alpaca", choices=["alpaca", "massive"], help="Provider adapter to use.")
    parser.add_argument("--underlyings", default="SPY", help="Comma-separated underlying symbols.")
    parser.add_argument("--entry-start", required=True, help="Entry window start, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--entry-end", required=True, help="Entry window end, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--contract-status", default="inactive", choices=["active", "inactive"], help="Option contract status.")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--max-contracts", type=int, default=25)
    parser.add_argument("--option-limit", type=int, default=100, help="Provider contract metadata page limit before local filtering.")
    parser.add_argument("--forward-days", type=int, default=30)
    parser.add_argument("--dataset-version", default="candidate_rows_v001")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--jsonl-output", default=None, help="Optional JSONL inspection copy.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    provider = _provider_from_args(args.provider)

    config = CandidateDatasetConfig(
        underlyings=[item.strip().upper() for item in args.underlyings.split(",") if item.strip()],
        entry_start=_parse_datetime(args.entry_start),
        entry_end=_parse_datetime(args.entry_end, end_of_day=True),
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        contract_status=args.contract_status,
        option_limit=args.option_limit,
        max_contracts_per_underlying=args.max_contracts,
        forward_days=args.forward_days,
    )
    rows = HistoricalCandidateDatasetBuilder(provider, provider, provider).build(config)
    result = ParquetDatasetWriter(root_dir=args.output_dir).write(
        rows,
        dataset_version=args.dataset_version,
        dataset_type="candidate_rows",
        schema_columns=list(CandidateDatasetRow.__dataclass_fields__.keys()) + ["entry_date"],
        metadata={
            "provider": args.provider,
            "underlyings": config.underlyings,
            "entry_start": config.entry_start.isoformat(),
            "entry_end": config.entry_end.isoformat(),
            "contract_status": config.contract_status,
            "min_dte": config.min_dte,
            "max_dte": config.max_dte,
            "forward_days": config.forward_days,
            "feature_set_version": "features_v001",
            "label_version": config.label_version,
        },
    )

    if args.jsonl_output:
        _write_jsonl(rows, Path(args.jsonl_output))

    print(f"Wrote {result.row_count} row(s) to {result.root_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _write_jsonl(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), default=str) + "\n")


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


def _provider_from_args(provider_name: str):
    if provider_name == "alpaca":
        return AlpacaProvider.from_env()
    if provider_name == "massive":
        return MassiveProvider.from_env()
    raise ValueError(f"Unsupported provider: {provider_name}")


if __name__ == "__main__":
    raise SystemExit(main())
