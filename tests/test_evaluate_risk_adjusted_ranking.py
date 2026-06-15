"""
Tests for ml.models.evaluate_risk_adjusted_ranking.

Covers:
- apply_large_loss_gate: veto logic based on large-loss probability threshold
- compute_sortino_score: downside-vol-adjusted score computation
- RiskAdjustedConfig: dataclass defaults
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ml.models.evaluate_risk_adjusted_ranking import (
    RiskAdjustedConfig,
    apply_large_loss_gate,
    compute_sortino_score,
    evaluate_risk_adjusted_ranking,
)


# ══════════════════════════════════════════════════════════════════════════════
# RiskAdjustedConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskAdjustedConfig:

    def test_default_selection_fraction(self):
        cfg = RiskAdjustedConfig()
        assert cfg.selection_fraction == 0.10

    def test_default_max_large_loss_probability(self):
        cfg = RiskAdjustedConfig()
        assert cfg.max_large_loss_probability == 0.70

    def test_custom_values(self):
        cfg = RiskAdjustedConfig(selection_fraction=0.05, max_large_loss_probability=0.50)
        assert cfg.selection_fraction == 0.05
        assert cfg.max_large_loss_probability == 0.50

    def test_none_disables_gate(self):
        cfg = RiskAdjustedConfig(max_large_loss_probability=None)
        assert cfg.max_large_loss_probability is None


# ══════════════════════════════════════════════════════════════════════════════
# apply_large_loss_gate
# ══════════════════════════════════════════════════════════════════════════════

class TestApplyLargeLossGate:

    def test_rows_above_threshold_set_to_neg_inf(self):
        """Candidates with large-loss prob > threshold should be vetoed (-inf)."""
        scores = pd.Series([10.0, 20.0, 30.0])
        probs = pd.Series([0.20, 0.80, 0.10])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        assert gated.iloc[0] == 10.0
        assert gated.iloc[1] == float("-inf")
        assert gated.iloc[2] == 30.0

    def test_rows_at_threshold_not_vetoed(self):
        """Candidates exactly at the threshold should NOT be vetoed (> not >=)."""
        scores = pd.Series([50.0])
        probs = pd.Series([0.70])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        assert gated.iloc[0] == 50.0

    def test_none_threshold_disables_gate(self):
        """When max_large_loss_probability is None, no rows are vetoed."""
        scores = pd.Series([10.0, 20.0, 30.0])
        probs = pd.Series([0.90, 0.95, 0.99])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=None)

        assert gated.iloc[0] == 10.0
        assert gated.iloc[1] == 20.0
        assert gated.iloc[2] == 30.0

    def test_all_below_threshold_passes_all(self):
        """If all probs are below threshold, all scores remain unchanged."""
        scores = pd.Series([100.0, 200.0, 300.0])
        probs = pd.Series([0.10, 0.30, 0.50])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        pd.testing.assert_series_equal(gated, scores, check_names=False)

    def test_all_above_threshold_vetoes_all(self):
        """If all probs are above threshold, all scores set to -inf."""
        scores = pd.Series([100.0, 200.0])
        probs = pd.Series([0.80, 0.90])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        assert all(gated == float("-inf"))

    def test_nan_probability_treated_as_high_risk(self):
        """NaN in large_loss_probability should be filled with 1.0 (vetoed)."""
        scores = pd.Series([100.0, 200.0])
        probs = pd.Series([0.10, float("nan")])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        assert gated.iloc[0] == 100.0
        assert gated.iloc[1] == float("-inf")

    def test_nan_score_stays_neg_inf(self):
        """NaN in score should become -inf (fillna(-inf) in implementation)."""
        scores = pd.Series([float("nan"), 50.0])
        probs = pd.Series([0.10, 0.10])

        gated = apply_large_loss_gate(scores, probs, max_large_loss_probability=0.70)

        assert gated.iloc[0] == float("-inf")
        assert gated.iloc[1] == 50.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_sortino_score
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeSortinoScore:

    def test_basic_sortino_computation(self):
        """sortino_score = gated_score / (iv * sqrt(dte/252))."""
        gated = pd.Series([1.0])
        df = pd.DataFrame({"implied_volatility": [0.25], "dte": [30]})

        result = compute_sortino_score(gated, df)

        vol_factor = 0.25 * np.sqrt(30.0 / 252.0)
        expected = 1.0 / vol_factor
        assert abs(result.iloc[0] - expected) < 1e-6

    def test_neg_inf_score_stays_neg_inf(self):
        """Vetoed rows (-inf score) remain -inf in sortino computation."""
        gated = pd.Series([float("-inf"), 10.0])
        df = pd.DataFrame({"implied_volatility": [0.30, 0.30], "dte": [30, 30]})

        result = compute_sortino_score(gated, df)

        assert result.iloc[0] == float("-inf")
        assert np.isfinite(result.iloc[1])

    def test_nan_iv_defaults_to_025(self):
        """NaN implied_volatility values should be filled to 0.25."""
        gated = pd.Series([1.0])
        df = pd.DataFrame({"implied_volatility": [float("nan")], "dte": [30]})

        result = compute_sortino_score(gated, df)

        vol_factor = 0.25 * np.sqrt(30.0 / 252.0)
        expected = 1.0 / vol_factor
        assert abs(result.iloc[0] - expected) < 1e-6

    def test_nan_dte_defaults_to_30(self):
        """NaN dte values should be filled to 30."""
        gated = pd.Series([1.0])
        df = pd.DataFrame({"implied_volatility": [0.25], "dte": [float("nan")]})

        result = compute_sortino_score(gated, df)

        vol_factor = 0.25 * np.sqrt(30.0 / 252.0)
        expected = 1.0 / vol_factor
        assert abs(result.iloc[0] - expected) < 1e-6

    def test_higher_iv_produces_lower_sortino_score(self):
        """Higher IV means more downside risk, so sortino_score should be lower."""
        gated = pd.Series([10.0, 10.0])
        df = pd.DataFrame({"implied_volatility": [0.20, 0.50], "dte": [30, 30]})

        result = compute_sortino_score(gated, df)

        assert result.iloc[0] > result.iloc[1]

    def test_higher_dte_produces_lower_sortino_score(self):
        """Longer DTE means more exposure time, so sortino_score should be lower."""
        gated = pd.Series([10.0, 10.0])
        df = pd.DataFrame({"implied_volatility": [0.25, 0.25], "dte": [7, 45]})

        result = compute_sortino_score(gated, df)

        assert result.iloc[0] > result.iloc[1]

    def test_zero_iv_clipped_to_minimum(self):
        """Zero IV should be clipped to a small positive to prevent div-by-zero."""
        gated = pd.Series([10.0])
        df = pd.DataFrame({"implied_volatility": [0.0], "dte": [30]})

        result = compute_sortino_score(gated, df)

        assert np.isfinite(result.iloc[0])
        assert result.iloc[0] > 0


def test_evaluate_risk_adjusted_ranking_handles_dict_portfolio_diagnostics(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "prediction": [0.9, 0.5, 0.1],
            "realized_pnl_per_contract": [100.0, -50.0, 25.0],
            "return_on_risk": [0.2, -0.1, 0.05],
            "large_loss_label": [0, 1, 0],
            "stop_loss_hit": [0, 1, 0],
            "entry_timestamp": [
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                "2026-01-03T00:00:00+00:00",
            ],
            "strategy": ["PCS", "PCS", "CCS"],
            "dte": [14, 14, 14],
            "underlying_close": [500.0, 500.0, 500.0],
            "short_strike": [495.0, 495.0, 505.0],
            "long_strike": [490.0, 490.0, 510.0],
            "entry_credit": [1.0, 1.0, 1.0],
            "implied_volatility": [0.2, 0.2, 0.2],
        }
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking.load_dataset",
        lambda path: frame.copy(),
    )
    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking._score_holdout",
        lambda df, artifact: df.copy(),
    )
    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking._selection_metrics",
        lambda df, score_column, cfg: {
            "selected_rows": int(np.isfinite(pd.to_numeric(df[score_column], errors="coerce")).sum()),
            "mean_pnl": round(float(df["realized_pnl_per_contract"].mean()), 6),
            "profit_factor": 1.0,
            "win_rate": 0.5,
        },
    )
    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking._score_classifier",
        lambda df, path: np.array([0.1, 0.8, 0.2]) if "large_loss" in str(path) else np.array([0.1, 0.2, 0.9]),
    )
    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking.load_config",
        lambda path: {"ml_scanner": {"large_loss_veto_threshold": 0.6, "stop_loss_veto_threshold": 0.3}},
    )

    def _fake_controls(df, score_column, **kwargs):
        diagnostics = {
            "gate_stage": pd.Series(["selected", "portfolio_gamma", "selected"], index=df.index, dtype="string"),
            "gate_reason": pd.Series([pd.NA, "gamma", pd.NA], index=df.index, dtype="string"),
            "gate_stage_counts": {"selected": 2, "portfolio_gamma": 1},
        }
        return pd.to_numeric(df[score_column], errors="coerce").fillna(float("-inf")), diagnostics

    monkeypatch.setattr(
        "ml.models.evaluate_risk_adjusted_ranking.apply_portfolio_risk_controls",
        _fake_controls,
    )

    report = evaluate_risk_adjusted_ranking(
        Path("dummy"),
        artifact_path,
        large_loss_artifact=Path("large_loss.json"),
        stop_loss_artifact=Path("stop_loss.json"),
        runtime_config_path=Path("config.json"),
        config=RiskAdjustedConfig(apply_portfolio_risk_controls=True),
    )

    assert report["trade_pipeline"]["portfolio_controls_applied"] is True
    assert report["trade_pipeline"]["portfolio_diagnostics_summary"]["gate_stage_counts"]["selected"] == 2
