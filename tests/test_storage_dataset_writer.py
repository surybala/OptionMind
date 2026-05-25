import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from ml.datasets import CandidateDatasetRow
from ml.storage import ParquetDatasetWriter
from ml.storage.partitions import partition_path, partition_value


def _row(underlying="SPY", source="fake"):
    return CandidateDatasetRow(
        entry_timestamp=datetime(2026, 5, 14, tzinfo=UTC),
        underlying=underlying,
        option_symbol=f"{underlying}260626P00500000",
        option_type="put",
        strike=500.0,
        expiration=date(2026, 6, 26),
        dte=43,
        source=source,
        underlying_close=504.0,
        underlying_return_1d=0.008,
        underlying_return_5d=0.01,
        underlying_return_20d=0.05,
        underlying_range_pct=0.01,
        underlying_realized_vol_5d=0.2,
        underlying_realized_vol_20d=0.25,
        underlying_sma_20_distance_pct=0.02,
        underlying_above_sma_20=1,
        underlying_volatility_ratio_5d_20d=0.8,
        underlying_volume=1200,
        strike_distance_pct=-0.00793651,
        moneyness=1.008,
        market_regime_symbol="SPY",
        market_return_5d=0.01,
        market_return_20d=0.05,
        market_realized_vol_5d=0.2,
        market_realized_vol_20d=0.25,
        market_sma_20_distance_pct=0.02,
        market_above_sma_20=1,
        market_volatility_ratio_5d_20d=0.8,
        market_trend_regime="uptrend",
        market_volatility_regime="normal",
        option_entry_open=4.0,
        option_entry_high=4.2,
        option_entry_low=3.9,
        option_entry_price=4.0,
        option_entry_range_pct=0.075,
        option_entry_volume=100,
        option_entry_trade_count=12,
        option_entry_vwap=4.05,
        option_exit_price=2.0,
        exit_timestamp=datetime(2026, 5, 15, tzinfo=UTC),
        exit_reason="profit_take",
        expected_pnl=200.0,
        realized_pnl_per_contract=200.0,
        profit_label=1,
        stop_loss_hit=0,
        large_loss_label=0,
        max_adverse_excursion=0.0,
        max_favorable_excursion=200.0,
        days_to_exit=1.0,
        label_version="short_option_labels_v001",
    )


def test_partition_value_sanitizes_values():
    assert partition_value("SPY/../bad value") == "SPY_.._bad_value"
    assert partition_value(None) == "unknown"


def test_partition_path_uses_key_value_segments(tmp_path):
    path = partition_path(
        tmp_path,
        {"source": "fake", "underlying": "SPY", "entry_date": date(2026, 5, 14)},
        ["source", "underlying", "entry_date"],
    )

    assert path == tmp_path / "source=fake" / "underlying=SPY" / "entry_date=2026-05-14"


def test_parquet_writer_creates_partitions_and_manifest(tmp_path, monkeypatch):
    written = []

    def fake_to_parquet(self, path, index=False):
        written.append((path, len(self), index))
        path.write_text("fake parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    writer = ParquetDatasetWriter(root_dir=tmp_path)

    result = writer.write(
        [_row("SPY"), _row("QQQ")],
        dataset_version="candidate_rows_test",
        dataset_type="candidate_rows",
        metadata={"provider": "fake"},
    )

    assert result.row_count == 2
    assert len(result.files) == 2
    assert all(path.name == "part-00000.parquet" for path in result.files)
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["dataset_version"] == "candidate_rows_test"
    assert manifest["row_count"] == 2
    assert manifest["metadata"]["provider"] == "fake"
    assert len(written) == 2


def test_parquet_writer_writes_empty_schema_file(tmp_path, monkeypatch):
    written = []

    def fake_to_parquet(self, path, index=False):
        written.append((path, list(self.columns), len(self)))
        path.write_text("empty parquet")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    writer = ParquetDatasetWriter(root_dir=tmp_path)

    result = writer.write(
        [],
        dataset_version="empty",
        dataset_type="candidate_rows",
        schema_columns=["entry_timestamp", "underlying"],
    )

    assert result.row_count == 0
    assert len(result.files) == 1
    assert result.files[0].name == "part-00000.parquet"
    assert written == [(result.files[0], ["entry_timestamp", "underlying"], 0)]


def test_parquet_writer_explains_missing_engine(tmp_path, monkeypatch):
    def fake_to_parquet(self, path, index=False):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    writer = ParquetDatasetWriter(root_dir=tmp_path)

    with pytest.raises(RuntimeError, match="pyarrow or fastparquet"):
        writer.write([_row()], dataset_version="v", dataset_type="candidate_rows")
