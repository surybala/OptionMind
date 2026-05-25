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
        ],
    )

    assert parse_args().option_limit == 1000
