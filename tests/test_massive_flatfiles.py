import gzip
import io
import sys
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ml.providers.massive_flatfiles import MassiveFlatFilesClient, _is_retryable_download_error
from ml.providers.parquet_minute import ParquetMinuteBarProvider
from ml.storage import ParquetDatasetWriter


class FakeS3Client:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get_object(self, Bucket, Key):
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(self.payloads[Key])}


class FlakyBody:
    def __init__(self, chunks, fail_once=False):
        self.chunks = list(chunks)
        self.fail_once = fail_once
        self.failed = False

    def read(self, _size):
        if self.fail_once and not self.failed:
            self.failed = True
            raise Exception("Connection reset by peer")
        if not self.chunks:
            return b""
        return self.chunks.pop(0)


class FlakyS3Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get_object(self, Bucket, Key):
        self.calls += 1
        if self.calls == 1:
            return {"Body": FlakyBody([self.payload[:10]], fail_once=True)}
        return {"Body": io.BytesIO(self.payload)}


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


def test_massive_flatfiles_client_uses_s3v4_signature(monkeypatch, tmp_path):
    captured = {}

    class FakeConfig:
        def __init__(self, *, signature_version):
            self.signature_version = signature_version

    class FakeBoto3Module:
        @staticmethod
        def client(service_name, **kwargs):
            captured["service_name"] = service_name
            captured["kwargs"] = kwargs
            return object()

    class FakeBotocoreConfigModule:
        Config = FakeConfig

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3Module)
    monkeypatch.setitem(sys.modules, "botocore.config", FakeBotocoreConfigModule)
    client = MassiveFlatFilesClient(access_key="a", secret_key="b", cache_dir=tmp_path)

    s3_client = client._s3()

    assert s3_client is client.s3_client
    assert captured["service_name"] == "s3"
    assert captured["kwargs"]["endpoint_url"] == "https://files.massive.com"
    assert captured["kwargs"]["aws_access_key_id"] == "a"
    assert captured["kwargs"]["aws_secret_access_key"] == "b"
    assert captured["kwargs"]["region_name"] == "us-east-1"
    assert captured["kwargs"]["config"].signature_version == "s3v4"


def test_massive_flatfiles_download_retries_stream_reset(tmp_path):
    key = "us_stocks_sip/minute_aggs_v1/2025/05/2025-05-12.csv.gz"
    payload = _gzip_csv(
        "ticker,volume,open,close,high,low,window_start,transactions\n"
        "SPY,10,1.0,1.1,1.2,0.9,1747056600000000000,3\n"
    )
    client = MassiveFlatFilesClient(
        access_key="a",
        secret_key="b",
        cache_dir=tmp_path,
        s3_client=FlakyS3Client(payload),
    )

    path = client.download_file(key)

    assert path.exists()
    assert path.read_bytes() == payload
    assert client.s3_client.calls == 2


def test_retryable_download_error_detects_stream_reset():
    assert _is_retryable_download_error(Exception("Connection reset by peer"))
    assert not _is_retryable_download_error(Exception("NoSuchKey"))


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
                "underlying": "SPY",
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
                "underlying": "SPY",
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
            "underlying",
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


def test_parquet_minute_bar_provider_reads_underlying_partitioned_option_dataset(tmp_path):
    pytest.importorskip("pyarrow")
    writer = ParquetDatasetWriter(
        root_dir=tmp_path,
        partition_columns=["asset_class", "underlying", "window_date"],
    )
    writer.write(
        [
            {
                "source": "massive_flatfiles",
                "asset_class": "options",
                "dataset": "minute_aggs_v1",
                "ticker": "SPY250519C00350000",
                "raw_ticker": "O:SPY250519C00350000",
                "underlying": "SPY",
                "window_start": datetime(2025, 5, 12, 14, 30, tzinfo=UTC),
                "window_date": date(2025, 5, 12),
                "volume": 10,
                "open": 1.0,
                "close": 1.1,
                "high": 1.2,
                "low": 0.9,
                "transactions": 3,
            },
        ],
        dataset_version="flatfile_underlying_test",
        dataset_type="massive_flatfiles",
        schema_columns=[
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
        ],
    )

    provider = ParquetMinuteBarProvider(
        tmp_path / "massive_flatfiles" / "dataset_version=flatfile_underlying_test"
    )
    option_bars = provider.get_option_bars(
        ["O:SPY250519C00350000"],
        datetime(2025, 5, 12, 14, 0, tzinfo=UTC),
        datetime(2025, 5, 12, 15, 0, tzinfo=UTC),
        "1Min",
    )

    assert option_bars["SPY250519C00350000"][0].close == 1.1


def test_parquet_minute_bar_provider_derives_option_contracts_from_local_symbols(tmp_path):
    pytest.importorskip("pyarrow")
    writer = ParquetDatasetWriter(
        root_dir=tmp_path,
        partition_columns=["asset_class", "underlying", "window_date"],
    )
    writer.write(
        [
            {
                "source": "massive_flatfiles",
                "asset_class": "options",
                "dataset": "minute_aggs_v1",
                "ticker": "SPY250519C00350000",
                "raw_ticker": "O:SPY250519C00350000",
                "underlying": "SPY",
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
                "asset_class": "options",
                "dataset": "minute_aggs_v1",
                "ticker": "SPY250516P00490000",
                "raw_ticker": "O:SPY250516P00490000",
                "underlying": "SPY",
                "window_start": datetime(2025, 5, 12, 14, 31, tzinfo=UTC),
                "window_date": date(2025, 5, 12),
                "volume": 12,
                "open": 2.0,
                "close": 2.1,
                "high": 2.2,
                "low": 1.9,
                "transactions": 4,
            },
        ],
        dataset_version="flatfile_contract_test",
        dataset_type="massive_flatfiles",
        schema_columns=[
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
        ],
    )

    provider = ParquetMinuteBarProvider(
        tmp_path / "massive_flatfiles" / "dataset_version=flatfile_contract_test"
    )
    contracts = provider.get_option_contracts(
        ["SPY"],
        expiration_gte=date(2025, 5, 16),
        expiration_lte=date(2025, 5, 19),
        status="all",
    )

    assert [(contract.symbol, contract.option_type, contract.strike) for contract in contracts] == [
        ("SPY250516P00490000", "put", 490.0),
        ("SPY250519C00350000", "call", 350.0),
    ]
