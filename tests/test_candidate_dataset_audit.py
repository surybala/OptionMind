import json
from pathlib import Path

import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset, render_markdown, summarize_candidate_dataset


def _rows():
    return [
        {
            "entry_timestamp": "2026-05-14T00:00:00+00:00",
            "source": "fake",
            "underlying": "SPY",
            "option_symbol": "SPY260626P00500000",
            "option_type": "put",
            "dte": 43,
            "underlying_close": 504.0,
            "underlying_return_1d": 0.008,
            "option_entry_price": 4.0,
            "expected_pnl": 200.0,
            "realized_pnl_per_contract": 200.0,
            "profit_label": 1,
            "stop_loss_hit": 0,
            "large_loss_label": 0,
            "max_adverse_excursion": 0.0,
            "max_favorable_excursion": 200.0,
            "days_to_exit": 1.0,
            "exit_reason": "profit_take",
            "missing_fields": [],
        },
        {
            "entry_timestamp": "2026-05-15T00:00:00+00:00",
            "source": "fake",
            "underlying": "SPY",
            "option_symbol": "SPY260626C00550000",
            "option_type": "call",
            "dte": 42,
            "underlying_close": None,
            "option_entry_price": 4.0,
            "expected_pnl": -430.0,
            "realized_pnl_per_contract": -430.0,
            "profit_label": 0,
            "stop_loss_hit": 1,
            "large_loss_label": 1,
            "max_adverse_excursion": 430.0,
            "max_favorable_excursion": 0.0,
            "days_to_exit": 1.0,
            "exit_reason": "stop_loss",
            "missing_fields": ["underlying_close"],
        },
    ]


def test_audit_summarizes_jsonl_candidate_dataset(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")

    report = summarize_candidate_dataset(load_dataset(path))

    assert report["row_count"] == 2
    assert report["underlying_counts"] == {"SPY": 2}
    assert report["entry_date_counts"] == {"2026-05-14": 1, "2026-05-15": 1}
    assert report["exit_reason_counts"] == {"profit_take": 1, "stop_loss": 1}
    assert report["numeric_summary"]["realized_pnl_per_contract"]["min"] == -430.0
    assert report["missing_field_counts"] == {"underlying_close": 1}


def test_audit_loads_parquet_directory(tmp_path):
    dataset_dir = tmp_path / "dataset"
    part_dir = dataset_dir / "source=fake"
    part_dir.mkdir(parents=True)
    pd.DataFrame(_rows()).to_parquet(part_dir / "part-00000.parquet")

    df = load_dataset(dataset_dir)

    assert len(df) == 2


def test_audit_loads_empty_jsonl(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    report = summarize_candidate_dataset(load_dataset(path))

    assert report["row_count"] == 0


def test_audit_renders_markdown():
    report = summarize_candidate_dataset(pd.DataFrame(_rows()))
    markdown = render_markdown(report)

    assert "# Candidate Dataset Quality Report" in markdown
    assert "realized_pnl_per_contract" in markdown
