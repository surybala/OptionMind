from datetime import UTC, datetime

from ml.datasets.build_candidate_dataset import _parse_datetime, parse_args


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
