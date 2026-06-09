"""Archive compact Massive reference datasets for future local model training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ml.datasets.etf_universe import broad_etf_underlyings, stable_etf_underlyings
from ml.providers.massive import MassiveProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive Massive reference data for local ML training.")
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD for historical reference pulls.")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD for historical reference pulls.")
    parser.add_argument(
        "--underlying-preset",
        default="broad-etfs",
        choices=["broad-etfs", "stable-etfs", "custom"],
        help="Universe preset to archive.",
    )
    parser.add_argument(
        "--underlyings",
        default="",
        help="Comma-separated underlying list when --underlying-preset=custom.",
    )
    parser.add_argument(
        "--dataset-version",
        default=None,
        help="Optional explicit dataset version. Defaults to a Massive reference tag derived from the date range.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/datasets/massive_reference",
        help="Root directory for archived reference artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    underlyings = _underlyings_from_args(args)
    dataset_version = args.dataset_version or (
        f"massive_reference_{_slug(args.underlying_preset)}_{start.isoformat().replace('-', '')}_{end.isoformat().replace('-', '')}_v001"
    )
    out_dir = Path(args.output_dir) / f"dataset_version={dataset_version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = MassiveProvider.from_env()
    session = requests.Session()

    print(f"Archiving Massive reference data for {len(underlyings)} underlyings -> {out_dir}")
    active_contracts = provider.get_option_contracts(underlyings, status="active")
    expired_contracts = provider.get_option_contracts(
        underlyings,
        expiration_gte=start,
        expiration_lte=end,
        status="expired",
    )
    contract_rows = _dedupe_rows(
        [_serialize_contract(item) for item in [*active_contracts, *expired_contracts]],
        key="symbol",
    )

    dividend_rows = []
    for symbol, events in provider.get_dividends(underlyings, start, end).items():
        for event in events:
            row = _serialize_dataclass(event)
            row["symbol"] = symbol
            dividend_rows.append(row)

    split_rows: list[dict[str, Any]] = []
    overview_rows: list[dict[str, Any]] = []
    for symbol in underlyings:
        split_rows.extend(_fetch_paginated_json(
            session,
            provider.api_key,
            provider.base_url,
            "/stocks/v1/splits",
            {
                "ticker": symbol,
                "execution_date.gte": start.isoformat(),
                "execution_date.lte": end.isoformat(),
                "limit": 5000,
                "sort": "execution_date.asc",
            },
        ))
        overview = _fetch_json(
            session,
            provider.api_key,
            provider.base_url,
            f"/v3/reference/tickers/{symbol}",
            {"date": end.isoformat()},
        ).get("results")
        if overview:
            overview_rows.append(overview)

    _write_jsonl(out_dir / "options_contracts.jsonl", contract_rows)
    _write_jsonl(out_dir / "dividends.jsonl", sorted(dividend_rows, key=lambda row: (row.get("symbol"), row.get("ex_date"))))
    _write_jsonl(out_dir / "splits.jsonl", sorted(split_rows, key=lambda row: (row.get("ticker"), row.get("execution_date"))))
    _write_jsonl(out_dir / "ticker_overviews.jsonl", sorted(overview_rows, key=lambda row: row.get("ticker")))

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset_version,
        "underlyings": underlyings,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "counts": {
            "options_contracts": len(contract_rows),
            "dividends": len(dividend_rows),
            "splits": len(split_rows),
            "ticker_overviews": len(overview_rows),
        },
        "files": [
            "options_contracts.jsonl",
            "dividends.jsonl",
            "splits.jsonl",
            "ticker_overviews.jsonl",
        ],
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _underlyings_from_args(args: argparse.Namespace) -> list[str]:
    if args.underlying_preset == "broad-etfs":
        return broad_etf_underlyings()
    if args.underlying_preset == "stable-etfs":
        return stable_etf_underlyings()
    return [item.strip().upper() for item in args.underlyings.split(",") if item.strip()]


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _serialize_contract(item: Any) -> dict[str, Any]:
    payload = _serialize_dataclass(item)
    raw = payload.get("raw")
    if isinstance(raw, dict):
        payload["shares_per_contract"] = raw.get("shares_per_contract")
        payload["additional_underlyings"] = raw.get("additional_underlyings")
    return payload


def _serialize_dataclass(item: Any) -> dict[str, Any]:
    payload = asdict(item) if is_dataclass(item) else dict(item)
    return _json_ready(payload)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _dedupe_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[str(row.get(key) or "")] = row
    return sorted(deduped.values(), key=lambda row: row.get(key) or "")


def _fetch_json(
    session: requests.Session,
    api_key: str,
    base_url: str,
    path: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = session.get(
        f"{base_url.rstrip('/')}{path}",
        params={**params, "apiKey": api_key},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _fetch_paginated_json(
    session: requests.Session,
    api_key: str,
    base_url: str,
    path: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_url: str | None = None
    current_params: dict[str, Any] | None = dict(params)
    while True:
        if next_url:
            response = session.get(next_url, params={"apiKey": api_key}, timeout=60)
        else:
            response = session.get(
                f"{base_url.rstrip('/')}{path}",
                params={**(current_params or {}), "apiKey": api_key},
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        rows.extend(payload.get("results") or [])
        next_url = payload.get("next_url")
        if not next_url:
            break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
