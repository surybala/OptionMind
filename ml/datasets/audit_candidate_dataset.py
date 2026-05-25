"""Quality report for generated candidate datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_NUMERIC_COLUMNS = [
    "dte",
    "underlying_close",
    "underlying_return_1d",
    "underlying_return_5d",
    "underlying_realized_vol_5d",
    "underlying_realized_vol_20d",
    "strike_distance_pct",
    "moneyness",
    "option_entry_price",
    "option_entry_range_pct",
    "option_entry_volume",
    "option_entry_trade_count",
    "expected_pnl",
    "realized_pnl_per_contract",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "days_to_exit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated candidate dataset rows.")
    parser.add_argument("--input", required=True, help="JSONL file, parquet file, or dataset directory.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_dataset(Path(args.input))
    report = summarize_candidate_dataset(df)

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = render_markdown(report)
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_dataset(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.rglob("part-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No part-*.parquet files found under {path}")
        frames = [pd.read_parquet(file) for file in files]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if path.suffix == ".jsonl":
        if path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_json(path, lines=True)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset input: {path}")


def summarize_candidate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
    }
    if df.empty:
        return report

    if "entry_timestamp" in df:
        timestamps = pd.to_datetime(df["entry_timestamp"], errors="coerce")
        report["entry_start"] = _iso_or_none(timestamps.min())
        report["entry_end"] = _iso_or_none(timestamps.max())

    for column in ("source", "underlying", "option_type", "exit_reason", "label_version"):
        if column in df:
            report[f"{column}_counts"] = _value_counts(df[column])

    for column in ("profit_label", "stop_loss_hit", "large_loss_label"):
        if column in df:
            report[f"{column}_counts"] = _value_counts(df[column])

    report["numeric_summary"] = {
        column: _numeric_summary(df[column])
        for column in DEFAULT_NUMERIC_COLUMNS
        if column in df
    }
    report["missing_field_counts"] = _missing_field_counts(df)
    report["null_counts"] = {
        column: int(df[column].isna().sum())
        for column in DEFAULT_NUMERIC_COLUMNS
        if column in df and int(df[column].isna().sum()) > 0
    }
    report["pnl_extremes"] = _pnl_extremes(df)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Dataset Quality Report",
        "",
        f"- Rows: {report.get('row_count', 0)}",
        f"- Columns: {report.get('column_count', 0)}",
    ]
    if report.get("entry_start") or report.get("entry_end"):
        lines.append(f"- Entry range: {report.get('entry_start')} to {report.get('entry_end')}")

    for key in ("source_counts", "underlying_counts", "option_type_counts", "exit_reason_counts"):
        if key in report:
            lines.extend(["", f"## {key.replace('_', ' ').title()}", ""])
            for value, count in report[key].items():
                lines.append(f"- {value}: {count}")

    if report.get("numeric_summary"):
        lines.extend(["", "## Numeric Summary", "", "| Column | Missing | Min | P50 | Mean | Max |", "|---|---:|---:|---:|---:|---:|"])
        for column, summary in report["numeric_summary"].items():
            lines.append(
                f"| {column} | {summary['missing']} | {_fmt(summary['min'])} | "
                f"{_fmt(summary['p50'])} | {_fmt(summary['mean'])} | {_fmt(summary['max'])} |"
            )

    if report.get("missing_field_counts"):
        lines.extend(["", "## Missing Field Counts", ""])
        for field, count in report["missing_field_counts"].items():
            lines.append(f"- {field}: {count}")

    return "\n".join(lines) + "\n"


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).items()}


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    present = numeric.dropna()
    if present.empty:
        return {"missing": int(numeric.isna().sum()), "min": None, "p50": None, "mean": None, "max": None}
    return {
        "missing": int(numeric.isna().sum()),
        "min": round(float(present.min()), 6),
        "p50": round(float(present.median()), 6),
        "mean": round(float(present.mean()), 6),
        "max": round(float(present.max()), 6),
    }


def _missing_field_counts(df: pd.DataFrame) -> dict[str, int]:
    if "missing_fields" not in df:
        return {}
    counts: dict[str, int] = {}
    for value in df["missing_fields"]:
        fields = value if isinstance(value, list) else []
        for field in fields:
            counts[str(field)] = counts.get(str(field), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _pnl_extremes(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if "realized_pnl_per_contract" not in df:
        return {}
    columns = [
        column
        for column in ("entry_timestamp", "underlying", "option_symbol", "realized_pnl_per_contract", "exit_reason")
        if column in df
    ]
    sorted_df = df.sort_values("realized_pnl_per_contract")
    return {
        "worst": _records(sorted_df.head(5)[columns]),
        "best": _records(sorted_df.tail(5).sort_values("realized_pnl_per_contract", ascending=False)[columns]),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _iso_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()


def _fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
