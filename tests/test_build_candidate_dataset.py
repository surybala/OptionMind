from datetime import UTC, datetime

from ml.datasets.build_candidate_dataset import (
    _dividend_provider_from_args,
    _parse_datetime,
    _underlyings_from_args,
    parse_args,
)
from ml.providers import MassiveProvider


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
            "--market-regime-symbol",
            "qqq",
        ],
    )

    args = parse_args()
    assert args.option_limit == 1000
    assert args.max_rows_per_underlying == 5000
    assert args.sample_every_n_bars == 2
    assert args.stock_lookback_days == 75
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
    assert args.dividend_provider == "massive"
    assert args.min_output_rows == 0
    assert args.append is False


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


def test_dividend_provider_reuses_massive_market_provider():
    provider = MassiveProvider(api_key="test")

    assert _dividend_provider_from_args("massive", provider) is provider
