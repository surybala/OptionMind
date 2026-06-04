"""Partitioned Parquet dataset writer."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ml.storage.manifest import DatasetManifest
from ml.storage.partitions import partition_path


DEFAULT_PARTITIONS = ["source", "underlying", "entry_date"]


@dataclass(frozen=True)
class DatasetWriteResult:
    root_path: Path
    manifest_path: Path
    files: list[Path]
    row_count: int
    manifest: DatasetManifest


class ParquetDatasetWriter:
    """Write normalized dataset rows as partitioned Parquet plus manifest."""

    def __init__(
        self,
        root_dir: Path | str = "artifacts/datasets",
        partition_columns: list[str] | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.partition_columns = partition_columns or DEFAULT_PARTITIONS

    def write(
        self,
        rows: Iterable[Any],
        *,
        dataset_version: str,
        dataset_type: str,
        schema_columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        append: bool = False,
    ) -> DatasetWriteResult:
        normalized = [_normalize_row(row) for row in rows]
        dataset_root = self.root_dir / dataset_type / f"dataset_version={dataset_version}"
        dataset_root.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_root / "_manifest.json"
        existing_manifest = _read_manifest(manifest_path) if append else None

        files: list[Path] = []
        if normalized:
            for part_dir, part_rows in _group_by_partition(normalized, dataset_root, self.partition_columns).items():
                part_dir.mkdir(parents=True, exist_ok=True)
                part_path = _next_part_path(part_dir)
                _write_parquet(part_rows, part_path)
                files.append(part_path)
        else:
            part_path = dataset_root / "part-00000.parquet"
            _write_parquet([], part_path, schema_columns=schema_columns)
            files.append(part_path)

        all_files = [Path(path) for path in existing_manifest.files] + files if existing_manifest else files
        row_count = (existing_manifest.row_count if existing_manifest else 0) + len(normalized)
        manifest_metadata = _merge_manifest_metadata(
            existing_manifest.metadata if existing_manifest else None,
            metadata or {},
        )
        manifest = DatasetManifest.create(
            dataset_version=dataset_version,
            dataset_type=dataset_type,
            row_count=row_count,
            root_path=dataset_root,
            file_format="parquet",
            partition_columns=self.partition_columns,
            files=all_files,
            metadata=manifest_metadata,
        )
        manifest_path = manifest.write(manifest_path)
        return DatasetWriteResult(
            root_path=dataset_root,
            manifest_path=manifest_path,
            files=all_files,
            row_count=row_count,
            manifest=manifest,
        )


def _normalize_row(row: Any) -> dict[str, Any]:
    if is_dataclass(row):
        item = asdict(row)
    elif hasattr(row, "to_dict"):
        item = dict(row.to_dict())
    else:
        item = dict(row)
    entry_timestamp = item.get("entry_timestamp")
    if entry_timestamp is not None and "entry_date" not in item:
        item["entry_date"] = getattr(entry_timestamp, "date", lambda: str(entry_timestamp)[:10])()
    return item


def _group_by_partition(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    partition_columns: list[str],
) -> dict[Path, list[dict[str, Any]]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for row in rows:
        path = partition_path(dataset_root, row, partition_columns)
        grouped.setdefault(path, []).append(row)
    return grouped


def _next_part_path(part_dir: Path) -> Path:
    existing = sorted(part_dir.glob("part-*.parquet"))
    return part_dir / f"part-{len(existing):05d}.parquet"


def _write_parquet(rows: list[dict[str, Any]], path: Path, schema_columns: list[str] | None = None) -> None:
    df = pd.DataFrame(rows, columns=schema_columns)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet writing requires pyarrow or fastparquet. Install project dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc


def _read_manifest(path: Path) -> DatasetManifest | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetManifest(**payload)


def _merge_manifest_metadata(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing or {})
    merged.update(incoming)
    merged["start_date"] = _merge_iso_date_bound(
        (existing or {}).get("start_date"),
        incoming.get("start_date"),
        pick=min,
    )
    merged["end_date"] = _merge_iso_date_bound(
        (existing or {}).get("end_date"),
        incoming.get("end_date"),
        pick=max,
    )
    return {key: value for key, value in merged.items() if value is not None}


def _merge_iso_date_bound(
    existing_value: Any,
    incoming_value: Any,
    *,
    pick,
) -> str | Any | None:
    if existing_value is None:
        return incoming_value
    if incoming_value is None:
        return existing_value
    existing_date = _parse_iso_date(existing_value)
    incoming_date = _parse_iso_date(incoming_value)
    if existing_date is None or incoming_date is None:
        return incoming_value
    return pick(existing_date, incoming_date).isoformat()


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if len(text) != 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None
