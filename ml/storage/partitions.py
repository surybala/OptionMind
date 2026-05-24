"""Partition helpers for dataset artifact paths."""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any


_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def partition_value(value: Any) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        raw = value.isoformat()
    elif value is None:
        raw = "unknown"
    else:
        raw = str(value)
    return _SAFE_PART_RE.sub("_", raw)


def partition_path(root: Path, row: dict[str, Any], partition_columns: list[str]) -> Path:
    path = root
    for column in partition_columns:
        path = path / f"{column}={partition_value(row.get(column))}"
    return path
