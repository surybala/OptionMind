"""Tests for the binary large-loss classifier training pipeline."""
import math

import numpy as np
import pandas as pd
import pytest

from ml.models.train_large_loss_classifier import (
    LargeLossClassifierArtifact,
    _clf_metrics,
    _compute_fill_values,
    train_large_loss_classifier,
)


def _frame(n_rows: int = 20, positive_rate: float = 0.15) -> pd.DataFrame:
    """Minimal dataset with large_loss_label and a handful of numeric features."""
    rng = np.random.default_rng(42)
    n_pos = max(1, int(n_rows * positive_rate))
    labels = np.zeros(n_rows, dtype=int)
    labels[:n_pos] = 1

    rows = []
    for i in range(n_rows):
        rows.append({
            "entry_timestamp": f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
            "dte": int(rng.integers(7, 45)),
            "underlying_close": float(rng.uniform(100, 600)),
            "option_entry_price": float(rng.uniform(0.5, 10)),
            "strike_distance_pct": float(rng.uniform(-0.15, 0.0)),
            "implied_volatility": float(rng.uniform(0.10, 0.60)),
            "vix_close": float(rng.uniform(12, 45)),
            "market_realized_vol_5d": float(rng.uniform(0.05, 0.50)),
            "large_loss_label": int(labels[i]),
        })
    return pd.DataFrame(rows)


def test_train_large_loss_classifier_returns_artifact(tmp_path):
    artifact = train_large_loss_classifier(
        _frame(40),
        model_output=tmp_path / "large_loss.json",
        test_fraction=0.25,
        walk_forward_folds=1,
        num_boost_round=10,
        val_fraction=0.15,
        early_stopping_rounds=5,
    )

    assert isinstance(artifact, LargeLossClassifierArtifact)
    assert artifact.model_type == "xgboost_binary_large_loss_v001"
    assert artifact.target_column == "large_loss_label"
    assert artifact.train_rows > 0
    assert artifact.test_rows > 0
    assert 0.0 <= artifact.train_positive_rate <= 1.0
    assert (tmp_path / "large_loss.json").exists()


def test_train_large_loss_classifier_metrics_populated(tmp_path):
    artifact = train_large_loss_classifier(
        _frame(40),
        model_output=tmp_path / "m.json",
        test_fraction=0.25,
        walk_forward_folds=1,
        embargo_days=0,  # keep tiny test frames from exhausting all folds
        num_boost_round=10,
        val_fraction=0.0,
        early_stopping_rounds=0,
    )

    assert "train_auc" in artifact.metrics
    assert "test_auc" in artifact.metrics
    assert "walk_forward_auc_mean" in artifact.metrics
    assert artifact.metrics["walk_forward_folds"] == 1


def test_train_large_loss_classifier_walk_forward_folds(tmp_path):
    artifact = train_large_loss_classifier(
        _frame(60),
        model_output=tmp_path / "m.json",
        test_fraction=0.25,
        walk_forward_folds=2,
        embargo_days=0,  # keep tiny test frames from exhausting all folds
        num_boost_round=5,
        val_fraction=0.0,
        early_stopping_rounds=0,
    )
    assert len(artifact.walk_forward) == 2
    for fold in artifact.walk_forward:
        assert "metrics" in fold
        assert "auc" in fold["metrics"]


def test_train_large_loss_classifier_scale_pos_weight_override(tmp_path):
    artifact = train_large_loss_classifier(
        _frame(40),
        model_output=tmp_path / "m.json",
        test_fraction=0.25,
        walk_forward_folds=0,
        num_boost_round=5,
        val_fraction=0.0,
        early_stopping_rounds=0,
        scale_pos_weight=3.0,
    )
    assert artifact.params["scale_pos_weight"] == 3.0


def test_train_large_loss_classifier_missing_target_raises():
    df = pd.DataFrame([{"dte": 30, "underlying_close": 500}])
    with pytest.raises(ValueError, match="large_loss_label"):
        train_large_loss_classifier(
            df,
            model_output=None,  # type: ignore[arg-type]
            num_boost_round=5,
        )


def test_clf_metrics_perfect_separation():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    m = _clf_metrics(y_true, y_prob, threshold=0.5)
    assert m["tp"] == 2
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


def test_clf_metrics_random_classifier():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = rng.uniform(0, 1, size=1000)
    m = _clf_metrics(y_true, y_prob, threshold=0.5)
    # AUC of a random classifier should be close to 0.5
    assert 0.4 < m["auc"] < 0.6


def test_compute_fill_values_uses_median():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, None], "b": [10.0, None, None, None]})
    fv = _compute_fill_values(df, ["a", "b"])
    assert fv["a"] == pytest.approx(2.0)
    assert fv["b"] == pytest.approx(10.0)
