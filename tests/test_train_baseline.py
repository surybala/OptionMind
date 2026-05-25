import pandas as pd
import pytest

from ml.models.train_baseline import train_baseline


def _frame():
    return pd.DataFrame(
        [
            {"entry_timestamp": "2026-01-01", "dte": 30, "underlying_close": 500, "option_entry_price": 4.0, "expected_pnl": 120},
            {"entry_timestamp": "2026-01-02", "dte": 29, "underlying_close": 501, "option_entry_price": 4.1, "expected_pnl": 80},
            {"entry_timestamp": "2026-01-03", "dte": 28, "underlying_close": 502, "option_entry_price": 4.2, "expected_pnl": -50},
            {"entry_timestamp": "2026-01-04", "dte": 27, "underlying_close": 503, "option_entry_price": 4.3, "expected_pnl": -120},
            {"entry_timestamp": "2026-01-05", "dte": 26, "underlying_close": 504, "option_entry_price": 4.4, "expected_pnl": 200},
            {"entry_timestamp": "2026-01-06", "dte": 25, "underlying_close": 505, "option_entry_price": 4.5, "expected_pnl": 240},
        ]
    )


def test_train_baseline_returns_transparent_artifact():
    artifact = train_baseline(_frame(), test_fraction=0.33)

    assert artifact.model_type == "linear_least_squares_v001"
    assert artifact.target_column == "expected_pnl"
    assert "dte" in artifact.feature_columns
    assert "option_entry_price" in artifact.coefficients
    assert artifact.train_rows == 4
    assert artifact.test_rows == 2
    assert "test_mae" in artifact.metrics


def test_train_baseline_requires_labeled_rows():
    with pytest.raises(ValueError, match="Need at least"):
        train_baseline(pd.DataFrame([{"dte": 1, "expected_pnl": None}]))
