"""
Tests for ml.models.evaluate_risk_adjusted_ranking.

Covers:
- apply_large_loss_gate: veto logic based on large-loss probability threshold
- compute_sortino_score: downside-vol-adjusted score computation
- RiskAdjustedConfig: dataclass defaults
"""
import numpy as np
import pandas as pd

from ml.models.evaluate_risk_adjusted_ranking import (
    RiskAdjustedConfig,
    apply_large_loss_gate,
    compute_sortino_score,
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
