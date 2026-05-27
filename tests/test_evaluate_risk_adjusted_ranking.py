import pandas as pd

import ml.models.evaluate_risk_adjusted_ranking as evaluator
from ml.models.evaluate_risk_adjusted_ranking import (
    RiskAdjustedConfig,
    _cap_value,
    _evaluate_trade_pipeline,
    _resolve_risk_penalty_basis,
    apply_large_loss_gate,
    apply_portfolio_risk_controls,
    apply_probability_caps,
    risk_adjusted_score,
)


def test_risk_adjusted_score_penalizes_tail_probabilities_by_max_loss():
    score = risk_adjusted_score(
        pd.Series([100.0, 100.0]),
        pd.Series([500.0, 500.0]),
        pd.Series([0.10, 0.30]),
        pd.Series([0.20, 0.20]),
        large_loss_penalty_multiple=1.0,
        stop_loss_penalty_multiple=0.5,
    )

    assert score.tolist() == [0.0, -100.0]


def test_risk_adjusted_score_clips_invalid_probabilities():
    score = risk_adjusted_score(
        pd.Series([50.0]),
        pd.Series([100.0]),
        pd.Series([2.0]),
        pd.Series([-1.0]),
    )

    assert score.iloc[0] == -50.0


def test_risk_adjusted_score_can_penalize_in_return_on_risk_units():
    score = risk_adjusted_score(
        pd.Series([0.20]),
        pd.Series([500.0]),
        pd.Series([0.10]),
        pd.Series([0.20]),
        large_loss_penalty_multiple=1.0,
        stop_loss_penalty_multiple=0.5,
        risk_penalty_basis="return_on_risk",
    )

    assert round(score.iloc[0], 6) == 0.0


def test_resolve_risk_penalty_basis_uses_ranker_target_for_auto():
    assert _resolve_risk_penalty_basis("auto", "return_on_risk") == "return_on_risk"
    assert _resolve_risk_penalty_basis("auto", "expected_pnl") == "dollars"
    assert _resolve_risk_penalty_basis("dollars", "return_on_risk") == "dollars"


def test_cap_value_uses_default_unless_disabled_or_overridden():
    assert _cap_value(None, 0.70) == 0.70
    assert _cap_value(0.50, 0.70) == 0.50
    assert _cap_value(0.50, 0.70, disabled=True) is None


def test_apply_probability_caps_removes_rows_above_thresholds():
    score = apply_probability_caps(
        pd.Series([10.0, 20.0, 30.0]),
        pd.Series([0.20, 0.80, 0.10]),
        pd.Series([0.20, 0.20, 0.90]),
        max_large_loss_probability=0.70,
        max_stop_loss_probability=0.80,
    )

    assert score.iloc[0] == 10.0
    assert score.iloc[1] == float("-inf")
    assert score.iloc[2] == float("-inf")


def test_apply_large_loss_gate_eliminates_tail_risk_rows():
    score = apply_large_loss_gate(
        pd.Series([10.0, 20.0, 30.0]),
        pd.Series([0.20, 0.80, 0.70]),
        max_large_loss_probability=0.70,
    )

    assert score.iloc[0] == 10.0
    assert score.iloc[1] == float("-inf")
    assert score.iloc[2] == 30.0


def test_trade_pipeline_applies_large_loss_gate_before_portfolio_controls(monkeypatch):
    scored = pd.DataFrame(
        {
            "prediction": [100.0, 90.0, 80.0],
            "large_loss_probability": [0.10, 0.95, 0.20],
            "expected_pnl": [50.0, 500.0, 25.0],
            "max_profit": [100.0, 100.0, 100.0],
            "max_adverse_excursion": [10.0, 10.0, 10.0],
            "return_on_risk": [0.50, 5.0, 0.25],
            "large_loss_label": [0, 1, 0],
            "stop_loss_hit": [0, 1, 0],
            "strategy": ["PCS", "PCS", "PCS"],
            "underlying": ["SPY", "SPY", "QQQ"],
        }
    )
    seen = {}

    def fake_portfolio_controls(df, score_column, **kwargs):
        seen["score_column"] = score_column
        return pd.to_numeric(df[score_column], errors="coerce")

    monkeypatch.setattr(evaluator, "apply_portfolio_risk_controls", fake_portfolio_controls)

    report = _evaluate_trade_pipeline(
        scored,
        RiskAdjustedConfig(max_large_loss_probability=0.70),
        {},
    )

    assert seen["score_column"] == "large_loss_gate_score"
    assert report["large_loss_gate_eligible_rows"] == 2
    assert report["trade_pipeline_eligible_rows"] == 2
    assert report["trade_pipeline_selection"]["selected_rows"] == 2
    assert report["trade_pipeline_selection"]["large_loss_rate"] == 0.0


def test_apply_portfolio_risk_controls_uses_existing_service_defaults():
    df = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(["2026-01-05", "2026-01-05"], utc=True),
            "strategy": ["PCS", "PCS"],
            "underlying": ["SPY", "SPY"],
            "underlying_close": [500.0, 500.0],
            "dte": [30, 30],
            "short_strike": [499.0, 450.0],
            "long_strike": [494.0, 445.0],
            "entry_credit": [1.0, 1.0],
            "implied_volatility": [0.35, 0.35],
            "prediction": [100.0, 90.0],
        }
    )

    score = apply_portfolio_risk_controls(df, "prediction", account_capital=1_000.0)

    assert score.iloc[0] == float("-inf")
    assert score.iloc[1] == float("-inf")
