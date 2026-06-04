"""Massive flat-file ingestion helpers for bulk historical datasets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterator
import time

import pandas as pd


_DEFAULT_BUCKET = "flatfiles"
_DEFAULT_ENDPOINT = "https://files.massive.com"


@dataclass
class MassiveFlatFilesClient:
    """Download and cache Massive/Polygon flat files through the S3 endpoint."""

    access_key: str
    secret_key: str
    endpoint_url: str = _DEFAULT_ENDPOINT
    bucket: str = _DEFAULT_BUCKET
    cache_dir: Path | str | None = None
    s3_client: Any | None = None
    max_download_retries: int = 3

    source: str = "massive_flatfiles"

    @classmethod
    def from_env(cls) -> "MassiveFlatFilesClient":
        access_key = os.getenv("MASSIVE_S3_ACCESS_KEY") or os.getenv("POLYGON_S3_ACCESS_KEY")
        secret_key = os.getenv("MASSIVE_S3_SECRET_KEY") or os.getenv("POLYGON_S3_SECRET_KEY")
        if not access_key or not secret_key:
            raise ValueError(
                "MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY are required "
                "(or POLYGON_S3_ACCESS_KEY / POLYGON_S3_SECRET_KEY)."
            )
        return cls(
            access_key=access_key,
            secret_key=secret_key,
            cache_dir=os.getenv("MASSIVE_FLATFILES_CACHE_DIR", "artifacts/cache/massive_flatfiles"),
        )

    def daily_file_key(
        self,
        *,
        asset_class: str,
        dataset: str,
        day: date,
    ) -> str:
        root = _dataset_root(asset_class, dataset)
        return f"{root}/{day.year:04d}/{day.month:02d}/{day.isoformat()}.csv.gz"

    def download_file(
        self,
        key: str,
        *,
        overwrite: bool = False,
    ) -> Path:
        cache_path = self._cache_path(key)
        if cache_path.exists() and not overwrite:
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
        attempts = max(1, int(self.max_download_retries))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                body = self._s3().get_object(Bucket=self.bucket, Key=key)["Body"]
                with tmp_path.open("wb") as fh:
                    while True:
                        chunk = body.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                tmp_path.replace(cache_path)
                return cache_path
            except Exception as exc:
                last_error = exc
                tmp_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                if attempt >= attempts or not _is_retryable_download_error(exc):
                    raise
                time.sleep(min(2 ** (attempt - 1), 5))
        raise last_error or RuntimeError(f"Failed to download flat file: {key}")

    def iter_csv_chunks(
        self,
        key: str,
        *,
        usecols: list[str] | None = None,
        chunksize: int = 250_000,
        dtype: dict[str, Any] | None = None,
    ) -> Iterator[pd.DataFrame]:
        path = self.download_file(key)
        yield from pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            chunksize=chunksize,
            dtype=dtype,
        )

    def _cache_path(self, key: str) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir must be configured for flat-file downloads")
        return Path(self.cache_dir) / key

    def _s3(self):
        if self.s3_client is not None:
            return self.s3_client
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # pragma: no cover - import-path only
            raise RuntimeError(
                "Flat-file ingestion requires boto3. Install it with "
                "`pip install boto3` or add it to the project environment."
            ) from exc
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        return self.s3_client


def _dataset_root(asset_class: str, dataset: str) -> str:
    normalized_asset = asset_class.strip().lower()
    normalized_dataset = dataset.strip().lower()
    roots = {
        ("options", "minute_aggs_v1"): "us_options_opra/minute_aggs_v1",
        ("stocks", "minute_aggs_v1"): "us_stocks_sip/minute_aggs_v1",
    }
    try:
        return roots[(normalized_asset, normalized_dataset)]
    except KeyError as exc:
        raise ValueError(f"Unsupported flat-file dataset: asset_class={asset_class}, dataset={dataset}") from exc


def _is_retryable_download_error(exc: Exception) -> bool:
    message = str(exc)
    retryable_fragments = (
        "Connection reset by peer",
        "Connection broken",
        "Read timeout",
        "Read timed out",
        "ResponseStreamingError",
        "EndpointConnectionError",
    )
    return any(fragment in message for fragment in retryable_fragments)
