"""CLI for ingesting Massive flat files into partitioned parquet datasets."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
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
    "underlying",
    "window_start",
    "window_date",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "transactions",
]
_OPTION_TICKER_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class SymbolFilters:
    exact_tickers: set[str] | None = None
    underlyings: set[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Massive/Polygon flat files into parquet.")
    parser.add_argument("--asset-class", required=True, choices=["options", "stocks"])
    parser.add_argument("--dataset", default="minute_aggs_v1", choices=["minute_aggs_v1"])
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--tickers", default="", help="Comma-separated ticker filter. Optional but recommended.")
    parser.add_argument("--ticker-file", default=None, help="Optional file with one ticker per line.")
    parser.add_argument(
        "--underlyings",
        default="",
        help=(
            "Comma-separated underlying ticker filter. For stocks this matches the ticker; "
            "for options it matches all contracts whose OSI ticker belongs to those underlyings."
        ),
    )
    parser.add_argument("--underlying-file", default=None, help="Optional file with one underlying ticker per line.")
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
        partition_columns=_partition_columns(args.asset_class),
    )
    filters = _build_symbol_filters(
        args.asset_class,
        tickers_arg=args.tickers,
        ticker_file=args.ticker_file,
        underlyings_arg=args.underlyings,
        underlying_file=args.underlying_file,
    )
    total_rows = 0
    result = None
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    current = start
    first_write = not args.append

    while current <= end:
        key = client.daily_file_key(asset_class=args.asset_class, dataset=args.dataset, day=current)
        day_row_count = 0
        try:
            for chunk in client.iter_csv_chunks(
                key,
                usecols=["ticker", "volume", "open", "close", "high", "low", "window_start", "transactions"],
                chunksize=args.chunksize,
            ):
                normalized = _normalize_chunk(chunk, args.asset_class)
                normalized = _apply_symbol_filters(normalized, args.asset_class, filters)
                if normalized.empty:
                    continue
                day_row_count += len(normalized)
                result = writer.write(
                    normalized.to_dict(orient="records"),
                    dataset_version=args.dataset_version,
                    dataset_type="massive_flatfiles",
                    schema_columns=FLATFILE_SCHEMA,
                    metadata=_manifest_metadata(args, filters),
                    append=not first_write,
                )
                first_write = False
                total_rows = result.row_count
        except Exception as exc:
            if _is_missing_flatfile_error(exc):
                print(f"{current.isoformat()} -> missing flat file, skipped")
                current += timedelta(days=1)
                continue
            raise

        print(f"{current.isoformat()} -> {day_row_count} row(s)")
        current += timedelta(days=1)

    if result is None:
        result = writer.write(
            [],
            dataset_version=args.dataset_version,
            dataset_type="massive_flatfiles",
            schema_columns=FLATFILE_SCHEMA,
            metadata=_manifest_metadata(args, filters),
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


def _underlying_filter(underlyings_arg: str, underlying_file: str | None) -> set[str] | None:
    values: list[str] = []
    if underlyings_arg.strip():
        values.extend(item.strip().upper() for item in underlyings_arg.split(",") if item.strip())
    if underlying_file:
        values.extend(
            line.strip().upper()
            for line in Path(underlying_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        return None
    return set(values)


def _build_symbol_filters(
    asset_class: str,
    *,
    tickers_arg: str,
    ticker_file: str | None,
    underlyings_arg: str,
    underlying_file: str | None,
) -> SymbolFilters | None:
    exact_tickers = _ticker_filter(asset_class, tickers_arg, ticker_file)
    underlyings = _underlying_filter(underlyings_arg, underlying_file)
    if exact_tickers is None and underlyings is None:
        return None
    return SymbolFilters(exact_tickers=exact_tickers, underlyings=underlyings)


def _apply_symbol_filters(
    frame: pd.DataFrame,
    asset_class: str,
    filters: SymbolFilters | None,
) -> pd.DataFrame:
    if filters is None:
        return frame
    mask = pd.Series(False, index=frame.index)
    if filters.exact_tickers is not None:
        mask = mask | frame["ticker"].isin(filters.exact_tickers)
    if filters.underlyings is not None:
        if asset_class == "options":
            underlying_series = frame["ticker"].map(_option_underlying_from_ticker)
            mask = mask | underlying_series.isin(filters.underlyings)
        else:
            mask = mask | frame["ticker"].isin(filters.underlyings)
    return frame[mask]


def _manifest_metadata(args: argparse.Namespace, filters: SymbolFilters | None) -> dict[str, Any]:
    return {
        "asset_class": args.asset_class,
        "dataset": args.dataset,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "ticker_filter_count": 0 if filters is None or filters.exact_tickers is None else len(filters.exact_tickers),
        "underlying_filter_count": 0 if filters is None or filters.underlyings is None else len(filters.underlyings),
    }


def _partition_columns(asset_class: str) -> list[str]:
    if asset_class == "options":
        return ["asset_class", "underlying", "window_date"]
    return ["asset_class", "ticker", "window_date"]


def _normalize_chunk(frame: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    data = frame.copy()
    data["raw_ticker"] = data["ticker"].astype("string")
    data["ticker"] = data["ticker"].map(lambda value: _normalize_ticker(asset_class, value))
    if asset_class == "options":
        data["underlying"] = data["ticker"].map(_option_underlying_from_ticker)
    else:
        data["underlying"] = data["ticker"]
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
            "underlying",
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
    return ordered.dropna(subset=["ticker", "underlying", "window_start"])


def _normalize_ticker(asset_class: str, value: Any) -> str:
    raw = str(value or "").strip().upper()
    if asset_class == "options":
        return raw[2:] if raw.startswith("O:") else raw
    return raw


def _option_underlying_from_ticker(value: Any) -> str | None:
    raw = _normalize_ticker("options", value)
    match = _OPTION_TICKER_RE.match(raw)
    if not match:
        return None
    return match.group(1)


def _is_missing_flatfile_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = str(error.get("Code") or "").strip()
            if code in {"404", "NoSuchKey", "NotFound"}:
                return True
    return exc.__class__.__name__ == "NoSuchKey"


if __name__ == "__main__":
    raise SystemExit(main())
