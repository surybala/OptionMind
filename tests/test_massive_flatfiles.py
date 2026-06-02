import gzip
import io
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ml.providers.massive_flatfiles import MassiveFlatFilesClient
from ml.providers.parquet_minute import ParquetMinuteBarProvider
from ml.storage import ParquetDatasetWriter


class FakeS3Client:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.payloads[Key])}


def _gzip_csv(text: str) -> bytes:
    return gzip.compress(text.encode("utf-8"))


def test_massive_flatfiles_daily_file_key():
    client = MassiveFlatFilesClient(access_key="a", secret_key="b", cache_dir="/tmp")

    key = client.daily_file_key(
        asset_class="options",
        dataset="minute_aggs_v1",
        day=date(2025, 5, 12),
    )

    assert key == "us_options_opra/minute_aggs_v1/2025/05/2025-05-12.csv.gz"


def test_massive_flatfiles_iter_csv_chunks_downloads_and_caches(tmp_path):
    key = "us_options_opra/minute_aggs_v1/2025/05/2025-05-12.csv.gz"
    client = MassiveFlatFilesClient(
        access_key="a",
        secret_key="b",
        cache_dir=tmp_path,
        s3_client=FakeS3Client(
            {
                key: _gzip_csv(
                    "ticker,volume,open,close,high,low,window_start,transactions\n"
                    "O:SPY250519C00350000,10,1.0,1.1,1.2,0.9,1747056600000000000,3\n"
                )
            }
        ),
    )

    first = list(client.iter_csv_chunks(key, chunksize=1000))
    second = list(client.iter_csv_chunks(key, chunksize=1000))

    assert len(first) == 1
    assert first[0].iloc[0]["ticker"] == "O:SPY250519C00350000"
    assert len(second) == 1
    assert client.s3_client.calls == [("flatfiles", key)]


def test_parquet_minute_bar_provider_reads_target_partitions(tmp_path):
    pytest.importorskip("pyarrow")
    writer = ParquetDatasetWriter(
        root_dir=tmp_path,
        partition_columns=["asset_class", "ticker", "window_date"],
    )
    writer.write(
        [
            {
                "source": "massive_flatfiles",
                "asset_class": "options",
                "dataset": "minute_aggs_v1",
                "ticker": "SPY250519C00350000",
                "raw_ticker": "O:SPY250519C00350000",
                "window_start": datetime(2025, 5, 12, 14, 30, tzinfo=UTC),
                "window_date": date(2025, 5, 12),
                "volume": 10,
                "open": 1.0,
                "close": 1.1,
                "high": 1.2,
                "low": 0.9,
                "transactions": 3,
            },
            {
                "source": "massive_flatfiles",
                "asset_class": "stocks",
                "dataset": "minute_aggs_v1",
                "ticker": "SPY",
                "raw_ticker": "SPY",
                "window_start": datetime(2025, 5, 12, 14, 30, tzinfo=UTC),
                "window_date": date(2025, 5, 12),
                "volume": 1000,
                "open": 500.0,
                "close": 501.0,
                "high": 502.0,
                "low": 499.5,
                "transactions": 100,
            },
        ],
        dataset_version="flatfile_test",
        dataset_type="massive_flatfiles",
        schema_columns=[
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
        ],
    )

    provider = ParquetMinuteBarProvider(
        tmp_path / "massive_flatfiles" / "dataset_version=flatfile_test"
    )
    option_bars = provider.get_option_bars(
        ["O:SPY250519C00350000"],
        datetime(2025, 5, 12, 14, 0, tzinfo=UTC),
        datetime(2025, 5, 12, 15, 0, tzinfo=UTC),
        "1Min",
    )
    stock_bars = provider.get_stock_bars(
        ["SPY"],
        datetime(2025, 5, 12, 14, 0, tzinfo=UTC),
        datetime(2025, 5, 12, 15, 0, tzinfo=UTC),
        "1Min",
    )

    assert option_bars["SPY250519C00350000"][0].close == 1.1
    assert stock_bars["SPY"][0].close == 501.0
