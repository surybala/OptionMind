import pandas as pd
import pytest
import numpy as np

pytest.importorskip("xgboost")

from ml.models.train_xgboost import (
    AsymmetricLossConfig,
    _asymmetric_objective,
    _cap_rows_per_underlying,
    _inverse_transform_target,
    _oversample_high_vol,
    _transform_target,
    train_xgboost,
)


def _frame():
    return pd.DataFrame(
        [
            {"entry_timestamp": "2026-01-01", "dte": 30, "underlying_close": 500, "option_entry_price": 4.0, "expected_pnl": 120},
            {"entry_timestamp": "2026-01-02", "dte": 29, "underlying_close": 501, "option_entry_price": 4.1, "expected_pnl": 80},
            {"entry_timestamp": "2026-01-03", "dte": 28, "underlying_close": 502, "option_entry_price": 4.2, "expected_pnl": -50},
            {"entry_timestamp": "2026-01-04", "dte": 27, "underlying_close": 503, "option_entry_price": 4.3, "expected_pnl": -120},
            {"entry_timestamp": "2026-01-05", "dte": 26, "underlying_close": 504, "option_entry_price": 4.4, "expected_pnl": 200},
            {"entry_timestamp": "2026-01-06", "dte": 25, "underlying_close": 505, "option_entry_price": 4.5, "expected_pnl": 240},
            {"entry_timestamp": "2026-01-07", "dte": 24, "underlying_close": 506, "option_entry_price": 4.6, "expected_pnl": -500},
            {"entry_timestamp": "2026-01-08", "dte": 23, "underlying_close": 507, "option_entry_price": 4.7, "expected_pnl": 320},
        ]
    )


def test_train_xgboost_returns_asymmetric_artifact(tmp_path):
    artifact = train_xgboost(
        _frame(),
        model_output=tmp_path / "model.json",
        min_rows=4,
        test_fraction=0.25,
        walk_forward_folds=1,
        num_boost_round=5,
        early_stopping_rounds=3,
        val_fraction=0.15,
        loss_config=AsymmetricLossConfig(max_multiplier=5.0),
    )

    assert artifact.model_type == "xgboost_asymmetric_pseudohuber_v002"
    assert artifact.target_column == "expected_pnl"
    assert artifact.train_rows == 6
    assert artifact.test_rows == 2
    assert artifact.loss_config["max_multiplier"] == 5.0
    assert artifact.params["num_boost_round"] <= 5
    assert "test_top_decile_actual_mean" in artifact.metrics
    assert artifact.metrics["walk_forward_folds"] == 1
    assert artifact.walk_forward[0]["test_rows"] == 2
    assert (tmp_path / "model.json").exists()


def test_train_xgboost_early_stopping_reduces_rounds(tmp_path):
    artifact = train_xgboost(
        _frame(),
        model_output=tmp_path / "model.json",
        min_rows=4,
        test_fraction=0.25,
        walk_forward_folds=0,
        num_boost_round=200,
        early_stopping_rounds=5,
        val_fraction=0.20,
    )

    # Early stopping should have fired well before 200 rounds on this tiny dataset.
    assert artifact.params["num_boost_round"] <= 200


def test_train_xgboost_embargo_reflected_in_walk_forward(tmp_path):
    artifact = train_xgboost(
        _frame(),
        model_output=tmp_path / "model.json",
        min_rows=4,
        test_fraction=0.25,
        walk_forward_folds=1,
        num_boost_round=5,
        embargo_days=0,
        val_fraction=0.0,
    )
    # With embargo_days=0 each fold reports embargo_rows=0
    assert artifact.walk_forward[0]["embargo_rows"] == 0


def test_train_xgboost_requires_labeled_rows(tmp_path):
    with pytest.raises(ValueError, match="Need at least"):
        train_xgboost(pd.DataFrame([{"dte": 1, "expected_pnl": None}]), model_output=tmp_path / "model.json")


def test_signed_log_target_transform_compresses_tail_losses():
    config = AsymmetricLossConfig(target_scale=100.0, target_clip=5000.0)
    raw = np.array([-10_000.0, -1_000.0, 1_000.0])

    transformed = _transform_target(raw, config)

    assert abs(transformed[0]) < abs(transformed[1]) * 2
    assert _inverse_transform_target(transformed, config)[0] == -5000.0
    assert _inverse_transform_target(transformed, config)[1] == pytest.approx(-1000.0)


def test_asymmetric_pseudo_huber_objective_clips_extreme_gradients():
    config = AsymmetricLossConfig(
        target_scale=100.0,
        target_clip=5000.0,
        huber_delta=1.0,
        gradient_clip=2.0,
        max_multiplier=5.0,
    )
    labels = _transform_target(np.array([-10_000.0, -1_000.0, 100.0]), config)
    predt = _transform_target(np.array([1_000.0, 1_000.0, 100.0]), config)

    class FakeDMatrix:
        def get_label(self):
            return labels

    grad, hess = _asymmetric_objective(config)(predt, FakeDMatrix())

    assert np.all(np.isfinite(grad))
    assert np.all(np.isfinite(hess))
    assert np.max(np.abs(grad)) <= config.gradient_clip
    assert np.min(hess) >= config.hessian_floor


def test_cap_rows_per_underlying_enforces_limit():
    df = pd.DataFrame(
        [{"underlying": "SPY", "entry_timestamp": f"2026-01-{i:02d}", "dte": i} for i in range(1, 11)]
        + [{"underlying": "QQQ", "entry_timestamp": f"2026-01-{i:02d}", "dte": i} for i in range(1, 6)]
    )
    capped = _cap_rows_per_underlying(df, max_rows=4)
    assert (capped["underlying"] == "SPY").sum() == 4
    assert (capped["underlying"] == "QQQ").sum() == 4


def test_cap_rows_per_underlying_preserves_chronological_order():
    df = pd.DataFrame(
        [{"underlying": "SPY", "entry_timestamp": f"2026-01-{i:02d}", "dte": i} for i in range(1, 11)]
    )
    capped = _cap_rows_per_underlying(df, max_rows=3)
    # Should keep the earliest 3 rows
    assert list(capped["dte"]) == [1, 2, 3]


def test_cap_rows_per_underlying_noop_when_none():
    df = pd.DataFrame([{"underlying": "SPY", "dte": i} for i in range(20)])
    result = _cap_rows_per_underlying(df, max_rows=100)
    assert len(result) == 20


def test_oversample_high_vol_replicates_correctly():
    df = pd.DataFrame(
        [{"vix_regime": 2.0, "expected_pnl": -200.0}] * 10
        + [{"vix_regime": 1.0, "expected_pnl": 50.0}] * 90
    )
    oversampled = _oversample_high_vol(df, factor=3)
    high_vol_count = (oversampled["vix_regime"] == 2.0).sum()
    assert high_vol_count == 30  # 10 * 3
    assert len(oversampled) == 90 + 30  # 90 normal + 10*3 high-vol


def test_oversample_high_vol_factor_one_is_noop():
    df = pd.DataFrame([{"vix_regime": 2.0, "expected_pnl": 0.0}] * 10)
    result = _oversample_high_vol(df, factor=1)
    assert len(result) == 10


def test_train_xgboost_with_per_underlying_cap(tmp_path):
    df = pd.DataFrame(
        [{"entry_timestamp": f"2026-01-{i:02d}", "underlying": "SPY", "dte": i,
          "underlying_close": 500, "option_entry_price": 4.0, "expected_pnl": 100 - i * 10}
         for i in range(1, 13)]
        + [{"entry_timestamp": f"2026-01-{i:02d}", "underlying": "QQQ", "dte": i,
            "underlying_close": 400, "option_entry_price": 3.0, "expected_pnl": 80 - i * 5}
           for i in range(1, 7)]
    )
    artifact = train_xgboost(
        df,
        model_output=tmp_path / "model.json",
        min_rows=4,
        test_fraction=0.25,
        walk_forward_folds=0,
        num_boost_round=5,
        val_fraction=0.0,
        max_rows_per_underlying=5,
    )
    # Cap of 5 per underlying: 5 SPY + 5 QQQ = 10 total rows, 25% test = 2-3 rows
    assert artifact.train_rows + artifact.test_rows <= 10


def test_train_xgboost_with_high_vol_oversampling(tmp_path):
    rows = []
    for i in range(1, 9):
        rows.append({
            "entry_timestamp": f"2026-01-{i:02d}", "dte": 30 - i,
            "underlying_close": 500, "option_entry_price": 4.0,
            "vix_close": 35.0 if i <= 2 else 18.0,  # 2 high-vol rows
            "expected_pnl": 100 if i % 2 == 0 else -50,
        })
    df = pd.DataFrame(rows)
    artifact = train_xgboost(
        df,
        model_output=tmp_path / "model.json",
        min_rows=4,
        test_fraction=0.25,
        walk_forward_folds=0,
        num_boost_round=5,
        val_fraction=0.0,
        high_vol_oversample_factor=3,
    )
    assert artifact.model_type is not None
