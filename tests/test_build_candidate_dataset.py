from datetime import UTC, datetime
from pathlib import Path

from ml.datasets.build_candidate_dataset import (
    _dividend_provider_from_args,
    _option_price_provider_from_args,
    _parse_datetime,
    _provider_from_args,
    _stock_provider_from_args,
    _underlyings_from_args,
    parse_args,
)
from ml.providers import MassiveProvider, ParquetMinuteBarProvider


def test_parse_datetime_keeps_date_start_for_entry_start():
    assert _parse_datetime("2025-04-15") == datetime(2025, 4, 15, tzinfo=UTC)


def test_parse_datetime_expands_date_only_entry_end_to_end_of_day():
    assert _parse_datetime("2025-04-15", end_of_day=True) == datetime(
        2025, 4, 15, 23, 59, 59, 999999, tzinfo=UTC
    )


def test_parse_args_accepts_option_limit(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--option-limit",
            "1000",
            "--max-rows-per-underlying",
            "5000",
            "--sample-every-n-bars",
            "2",
            "--stock-lookback-days",
            "75",
            "--stock-timeframe",
            "1Day",
            "--option-timeframe",
            "1Min",
            "--market-regime-symbol",
            "qqq",
        ],
    )

    args = parse_args()
    assert args.option_limit == 1000
    assert args.max_rows_per_underlying == 5000
    assert args.sample_every_n_bars == 2
    assert args.stock_lookback_days == 75
    assert args.stock_timeframe == "1Day"
    assert args.option_timeframe == "1Min"
    assert args.market_regime_symbol == "qqq"


def test_parse_args_defaults_to_full_contract_metadata_pagination(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
        ],
    )

    args = parse_args()
    assert args.option_limit is None
    assert args.max_contracts == 300
    assert args.spread_widths == "5,10,15,20"
    assert args.dividend_provider == "massive"
    assert args.min_output_rows == 0
    assert args.append is False
    assert args.option_price_provider == "same"
    assert args.contract_status == "inactive"


def test_parse_args_accepts_parquet_provider(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--provider",
            "parquet",
            "--option-dataset-root",
            "/tmp/options",
        ],
    )

    args = parse_args()
    assert args.provider == "parquet"
    assert args.option_dataset_root == "/tmp/options"


def test_parse_args_accepts_all_contract_status(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--contract-status",
            "all",
        ],
    )

    args = parse_args()
    assert args.contract_status == "all"


def test_underlying_preset_expands_broad_etfs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--underlying-preset",
            "broad-etfs",
        ],
    )

    underlyings = _underlyings_from_args(parse_args())
    assert {"SPY", "QQQ", "IWM", "TLT", "GLD"}.issubset(set(underlyings))


def test_underlying_preset_expands_stable_etfs(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--underlying-preset",
            "stable-etfs",
        ],
    )

    underlyings = _underlyings_from_args(parse_args())
    assert {"SPY", "QQQ", "TLT", "GLD"}.issubset(set(underlyings))
    assert "IWM" not in underlyings
    assert "XBI" not in underlyings


def test_underlyings_accepts_broad_etfs_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--underlyings",
            "broad-etfs",
        ],
    )

    underlyings = _underlyings_from_args(parse_args())
    assert "SPY" in underlyings
    assert "XLF" in underlyings


def test_underlyings_accepts_stable_etfs_alias(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_candidate_dataset",
            "--entry-start",
            "2025-04-01",
            "--entry-end",
            "2025-04-30",
            "--underlyings",
            "stable-etfs",
        ],
    )

    underlyings = _underlyings_from_args(parse_args())
    assert "SPY" in underlyings
    assert "XLV" in underlyings
    assert "IWM" not in underlyings


def test_dividend_provider_reuses_massive_market_provider():
    provider = MassiveProvider(api_key="test")

    assert _dividend_provider_from_args("massive", provider) is provider


def test_provider_from_args_builds_parquet_provider(tmp_path):
    provider = _provider_from_args("parquet", option_dataset_root=str(tmp_path))

    assert isinstance(provider, ParquetMinuteBarProvider)
    assert Path(provider.dataset_root) == tmp_path


def test_provider_from_args_requires_option_dataset_root_for_parquet():
    try:
        _provider_from_args("parquet")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "--option-dataset-root" in str(exc)


def test_stock_provider_from_args_builds_parquet_provider(tmp_path):
    provider = _stock_provider_from_args("parquet", dataset_root=str(tmp_path))

    assert isinstance(provider, ParquetMinuteBarProvider)
    assert Path(provider.dataset_root) == tmp_path


def test_stock_provider_from_args_requires_dataset_root_for_parquet():
    try:
        _stock_provider_from_args("parquet")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "--stock-dataset-root" in str(exc)


def test_option_price_provider_from_args_builds_parquet_provider(tmp_path):
    provider = _option_price_provider_from_args("parquet", dataset_root=str(tmp_path))

    assert isinstance(provider, ParquetMinuteBarProvider)
    assert Path(provider.dataset_root) == tmp_path


def test_option_price_provider_from_args_requires_dataset_root_for_parquet():
    try:
        _option_price_provider_from_args("parquet")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "--option-dataset-root" in str(exc)
