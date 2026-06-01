import math

import pandas as pd

from ml.models.portfolio_controls import apply_portfolio_risk_controls


def _row(index: int, score: float, strategy: str = "PCS") -> dict:
    return {
        "row_id": index,
        "entry_timestamp": "2025-01-03T15:30:00Z",
        "strategy": strategy,
        "underlying": "SPY",
        "underlying_close": 500.0,
        "short_strike": 500.0 if strategy == "PCS" else 450.0,
        "long_strike": 495.0 if strategy == "PCS" else 455.0,
        "dte": 7,
        "implied_volatility": 0.30,
        "entry_credit": 0.50,
        "gated_score": score,
    }


def _config() -> dict:
    return {
        "ml_scanner": {"top_n": 10},
        "pick_selection": {"mode": "model_ranked"},
        "risk_parameters": {
            "directional_exposure_caps": {"enabled": False},
            "portfolio_gamma_risk": {"enabled": False},
        },
    }


def test_portfolio_controls_marks_pick_selection_rejections():
    df = pd.DataFrame([
        _row(1, 0.90),
        _row(2, 0.80),
    ]).set_index("row_id")
    config = _config()
    config["ml_scanner"]["top_n"] = 1

    scores, diagnostics = apply_portfolio_risk_controls(
        df,
        "gated_score",
        scanner_controls=True,
        scanner_config=config,
        return_diagnostics=True,
    )

    assert math.isfinite(scores.loc[1])
    assert not math.isfinite(scores.loc[2])
    assert diagnostics["gate_stage"].loc[1] == "selected"
    assert diagnostics["gate_stage"].loc[2] == "pick_selection"
    assert diagnostics["gate_reason"].loc[2] == "scanner_controls"
    assert diagnostics["gate_stage_counts"] == {"selected": 1, "pick_selection": 1}


def test_portfolio_controls_marks_directional_exposure_rejections():
    df = pd.DataFrame([
        _row(1, 0.95, "PCS"),
        _row(2, 0.90, "PCS"),
        _row(3, 0.85, "PCS"),
    ]).set_index("row_id")
    config = _config()
    config["risk_parameters"]["directional_exposure_caps"] = {
        "enabled": True,
        "put": 0.02,
        "call": 0.02,
    }

    scores, diagnostics = apply_portfolio_risk_controls(
        df,
        "gated_score",
        account_capital=50_000,
        scanner_controls=True,
        scanner_config=config,
        return_diagnostics=True,
    )

    assert math.isfinite(scores.loc[1])
    assert math.isfinite(scores.loc[2])
    assert not math.isfinite(scores.loc[3])
    assert diagnostics["gate_stage"].loc[3] == "directional_exposure"
    assert diagnostics["gate_reason"].loc[3] == "side_exposure_cap"
    assert diagnostics["gate_stage_counts"] == {
        "selected": 2,
        "directional_exposure": 1,
    }


def test_portfolio_controls_uses_scanner_gamma_config_for_rejections():
    df = pd.DataFrame([_row(1, 0.90)]).set_index("row_id")
    config = _config()
    config["risk_parameters"]["portfolio_gamma_risk"] = {
        "enabled": True,
        "fail_closed": True,
        "max_stress_loss_pct": 0.00001,
        "max_near_expiry_stress_pct": 1.0,
        "symbol_stress_cap_enabled": True,
        "max_symbol_stress_pct": 1.0,
        "expiry_bucket_cap_enabled": False,
        "max_gamma_loss_to_daily_theta": 100.0,
        "shock_moves_pct": [1, 2, 3],
        "iv_shock_points": 10,
    }

    scores, diagnostics = apply_portfolio_risk_controls(
        df,
        "gated_score",
        account_capital=50_000,
        scanner_controls=True,
        scanner_config=config,
        return_diagnostics=True,
    )

    assert not math.isfinite(scores.loc[1])
    assert diagnostics["gate_stage"].loc[1] == "portfolio_gamma"
    assert "worst_stress" in str(diagnostics["gate_violation_codes"].loc[1]).split(",")
    assert diagnostics["portfolio_gamma_violation_counts"]["worst_stress"] == 1
