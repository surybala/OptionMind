"""Partitioned Parquet dataset writer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
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
    ) -> DatasetWriteResult:
        normalized = [_normalize_row(row) for row in rows]
        dataset_root = self.root_dir / dataset_type / f"dataset_version={dataset_version}"
        dataset_root.mkdir(parents=True, exist_ok=True)

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

        manifest = DatasetManifest.create(
            dataset_version=dataset_version,
            dataset_type=dataset_type,
            row_count=len(normalized),
            root_path=dataset_root,
            file_format="parquet",
            partition_columns=self.partition_columns,
            files=files,
            metadata=metadata,
        )
        manifest_path = manifest.write(dataset_root / "_manifest.json")
        return DatasetWriteResult(
            root_path=dataset_root,
            manifest_path=manifest_path,
            files=files,
            row_count=len(normalized),
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
