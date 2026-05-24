"""Dataset manifest model."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    dataset_version: str
    dataset_type: str
    created_at: str
    row_count: int
    root_path: str
    file_format: str
    partition_columns: list[str]
    files: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        dataset_version: str,
        dataset_type: str,
        row_count: int,
        root_path: Path,
        file_format: str,
        partition_columns: list[str],
        files: list[Path],
        metadata: dict[str, Any] | None = None,
    ) -> "DatasetManifest":
        return cls(
            dataset_version=dataset_version,
            dataset_type=dataset_type,
            created_at=datetime.now(UTC).isoformat(),
            row_count=row_count,
            root_path=str(root_path),
            file_format=file_format,
            partition_columns=partition_columns,
            files=[str(path) for path in files],
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path
