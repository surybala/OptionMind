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
    assert "test_profit_factor" in artifact.metrics
    assert "test_top_decile_max_drawdown" in artifact.metrics
    assert artifact.metrics["walk_forward_folds"] == 2
    assert artifact.walk_forward[0]["train_rows"] == 4
    assert artifact.walk_forward[0]["test_rows"] == 1
    assert "top_decile_actual_mean" in artifact.walk_forward[0]["metrics"]


def test_train_baseline_uses_train_only_fill_values():
    frame = _frame()
    frame.loc[:3, "option_entry_volume"] = [10, 20, 30, 40]
    frame.loc[4:, "option_entry_volume"] = [100000, 200000]
    frame.loc[0, "option_entry_volume"] = None

    artifact = train_baseline(frame, test_fraction=0.33, walk_forward_folds=0)

    assert artifact.fill_values["option_entry_volume"] == 30.0
    assert artifact.metrics["walk_forward_folds"] == 0


def test_train_baseline_requires_labeled_rows():
    with pytest.raises(ValueError, match="Need at least"):
        train_baseline(pd.DataFrame([{"dte": 1, "expected_pnl": None}]))


def test_train_baseline_embargo_excludes_rows_between_train_and_test():
    from ml.models.train_baseline import _walk_forward_splits

    # 10 rows; min_train_rows=6; 1 fold; embargo_rows=2
    # Fold: train=[0,6), embargo=[6,8), test=[8,10)
    splits = _walk_forward_splits(10, fold_count=1, min_train_rows=6, embargo_rows=2)
    assert len(splits) == 1
    fold_number, train_start, train_end, test_start, test_end = splits[0]
    assert train_end == 6
    assert test_start == 8
    assert test_end == 10


def test_train_baseline_embargo_skips_fold_when_embargo_consumes_all_test_rows():
    from ml.models.train_baseline import _walk_forward_splits

    # 8 rows; min_train_rows=6; fold chunk=[6,8) (2 rows); embargo=3 → test_start=9 >= test_end=8
    splits = _walk_forward_splits(8, fold_count=1, min_train_rows=6, embargo_rows=3)
    assert splits == []
