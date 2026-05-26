import pandas as pd

from ml.models.evaluate_risk_adjusted_ranking import (
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
