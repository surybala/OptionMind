"""CLI for ingesting Massive flat files into partitioned parquet datasets."""
from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from ml.providers.massive_flatfiles import MassiveFlatFilesClient
from ml.storage import ParquetDatasetWriter


FLATFILE_SCHEMA = [
    "source",
    "asset_class",
    "dataset",
    "ticker",
    "raw_ticker",
    "window_start",
    "window_date",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "transactions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Massive/Polygon flat files into parquet.")
    parser.add_argument("--asset-class", required=True, choices=["options", "stocks"])
    parser.add_argument("--dataset", default="minute_aggs_v1", choices=["minute_aggs_v1"])
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker filter. Optional but recommended.")
    parser.add_argument("--ticker-file", default=None, help="Optional file with one ticker per line.")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--min-output-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    client = MassiveFlatFilesClient.from_env()
    writer = ParquetDatasetWriter(
        root_dir=args.output_dir,
        partition_columns=["asset_class", "ticker", "window_date"],
    )
    tickers = _ticker_filter(args.asset_class, args.tickers, args.ticker_file)
    total_rows = 0
    result = None
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    current = start
    first_write = not args.append

    while current <= end:
        key = client.daily_file_key(asset_class=args.asset_class, dataset=args.dataset, day=current)
        day_rows: list[dict[str, Any]] = []
        for chunk in client.iter_csv_chunks(
            key,
            usecols=["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"],
            chunksize=args.chunksize,
        ):
            normalized = _normalize_chunk(chunk, args.asset_class)
            if tickers is not None:
                normalized = normalized[normalized["ticker"].isin(tickers)]
            if normalized.empty:
                continue
            day_rows.extend(normalized.to_dict(orient="records"))

        print(f"{current.isoformat()} -> {len(day_rows)} row(s)")
        if day_rows:
            result = writer.write(
                day_rows,
                dataset_version=args.dataset_version,
                dataset_type="massive_flatfiles",
                schema_columns=FLATFILE_SCHEMA,
                metadata={
                    "asset_class": args.asset_class,
                    "dataset": args.dataset,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "ticker_filter_count": 0 if tickers is None else len(tickers),
                },
                append=not first_write,
            )
            first_write = False
            total_rows = result.row_count
        current += timedelta(days=1)

    if result is None:
        result = writer.write(
            [],
            dataset_version=args.dataset_version,
            dataset_type="massive_flatfiles",
            schema_columns=FLATFILE_SCHEMA,
            metadata={
                "asset_class": args.asset_class,
                "dataset": args.dataset,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "ticker_filter_count": 0 if tickers is None else len(tickers),
            },
            append=False,
        )

    print(f"Wrote {total_rows} row(s) to {result.root_path}")
    print(f"Manifest: {result.manifest_path}")
    if args.min_output_rows and total_rows < args.min_output_rows:
        print(
            f"ERROR: dataset produced {total_rows} row(s), below --min-output-rows={args.min_output_rows}",
        )
        return 2
    return 0


def _ticker_filter(asset_class: str, tickers_arg: str, ticker_file: str | None) -> set[str] | None:
    values: list[str] = []
    if tickers_arg.strip():
        values.extend(item.strip() for item in tickers_arg.split(",") if item.strip())
    if ticker_file:
        values.extend(
            line.strip()
            for line in Path(ticker_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        return None
    return {_normalize_ticker(asset_class, value) for value in values}


def _normalize_chunk(frame: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    data = frame.copy()
    data["raw_ticker"] = data["ticker"].astype("string")
    data["ticker"] = data["ticker"].map(lambda value: _normalize_ticker(asset_class, value))
    timestamps = pd.to_datetime(data["window_start"], unit="ns", utc=True, errors="coerce")
    data["window_start"] = timestamps
    data["window_date"] = timestamps.dt.date
    data["source"] = "massive_flatfiles"
    data["asset_class"] = asset_class
    data["dataset"] = "minute_aggs_v1"
    ordered = data[
        [
            "source",
            "asset_class",
            "dataset",
            "ticker",
            "raw_ticker",
            "window_start",
            "window_date",
            "volume",
            "open",
            "close",
            "high",
            "low",
            "transactions",
        ]
    ]
    return ordered.dropna(subset=["ticker", "window_start"])


def _normalize_ticker(asset_class: str, value: Any) -> str:
    raw = str(value or "").strip().upper()
    if asset_class == "options":
        return raw[2:] if raw.startswith("O:") else raw
    return raw


if __name__ == "__main__":
    raise SystemExit(main())
