import math

import numpy as np
import pandas as pd
import pytest

from ml.models.train_baseline import (
    DEFAULT_FEATURE_COLUMNS,
    _bs_iv_simple,
    _compute_iv_skew_wing,
    _engineer_features,
    train_baseline,
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


# ---------------------------------------------------------------------------
# features_v004: new engineered features
# ---------------------------------------------------------------------------

def test_engineer_features_underlying_vol_vs_market():
    df = pd.DataFrame([{
        "underlying_realized_vol_5d": 0.30,
        "market_realized_vol_5d": 0.15,
    }])
    out = _engineer_features(df)
    assert "underlying_vol_vs_market" in out.columns
    assert out["underlying_vol_vs_market"].iloc[0] == pytest.approx(2.0)


def test_engineer_features_underlying_vol_vs_market_zero_market_vol():
    df = pd.DataFrame([{"underlying_realized_vol_5d": 0.30, "market_realized_vol_5d": 0.0}])
    out = _engineer_features(df)
    assert pd.isna(out["underlying_vol_vs_market"].iloc[0])


def test_engineer_features_option_activity_spike():
    df = pd.DataFrame([{"option_entry_trade_count": 50.0, "option_trade_count_5d_avg": 10.0}])
    out = _engineer_features(df)
    assert "option_activity_spike" in out.columns
    assert out["option_activity_spike"].iloc[0] == pytest.approx(5.0)


def test_engineer_features_option_activity_spike_zero_avg():
    df = pd.DataFrame([{"option_entry_trade_count": 50.0, "option_trade_count_5d_avg": 0.0}])
    out = _engineer_features(df)
    assert pd.isna(out["option_activity_spike"].iloc[0])


def test_engineer_features_iv_skew_wing_computed_for_credit_spread():
    """A shallow PCS: short put slightly OTM, long put further OTM — long IV > short IV."""
    S, K_short, K_long, dte = 500.0, 490.0, 480.0, 21
    T = dte / 365.0
    r = 0.045

    # Compute short IV from a known sigma (0.25) to get market_price for the short leg.
    sigma = 0.25
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K_short) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    from math import erf, exp, pi, sqrt
    nd2 = (1 + erf(-d2 / sqrt(2))) / 2
    nd1_neg = (1 + erf(-d1 / sqrt(2))) / 2
    short_price = K_short * exp(-r * T) * nd2 - S * nd1_neg

    # Compute long IV from sigma=0.30 (steeper put skew).
    sigma_long = 0.30
    d1l = (math.log(S / K_long) + (r + 0.5 * sigma_long**2) * T) / (sigma_long * sqrt_T)
    d2l = d1l - sigma_long * sqrt_T
    nd2l = (1 + erf(-d2l / sqrt(2))) / 2
    nd1l_neg = (1 + erf(-d1l / sqrt(2))) / 2
    long_price = K_long * exp(-r * T) * nd2l - S * nd1l_neg

    df = pd.DataFrame([{
        "long_option_entry_price": long_price,
        "underlying_close": S,
        "long_strike": K_long,
        "option_type": "put",
        "dte": dte,
        "implied_volatility": sigma,
    }])
    result = _compute_iv_skew_wing(df)
    assert result.iloc[0] == pytest.approx(0.30 - 0.25, abs=0.01)


def test_engineer_features_skips_iv_skew_wing_when_columns_missing():
    df = pd.DataFrame([{"dte": 21, "underlying_close": 500.0}])
    out = _engineer_features(df)
    # No credit-spread columns → no iv_skew_wing column
    assert "iv_skew_wing" not in out.columns


def test_bs_iv_simple_roundtrip():
    """Compute a price from a known sigma; recover the same sigma via IV solver."""
    S, K, T, r, sigma = 500.0, 490.0, 21 / 365.0, 0.045, 0.25
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    from math import erf, exp, pi, sqrt
    nd1 = (1 + erf(d1 / sqrt(2))) / 2
    nd2 = (1 + erf(d2 / sqrt(2))) / 2
    call_price = S * nd1 - K * exp(-r * T) * nd2

    recovered = _bs_iv_simple(call_price, S, K, T, r, "call")
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_bs_iv_simple_returns_none_on_bad_input():
    assert _bs_iv_simple(0.0, 500, 490, 21 / 365, 0.045, "call") is None  # zero price
    assert _bs_iv_simple(5.0, 0.0, 490, 21 / 365, 0.045, "call") is None  # zero spot


def test_feature_columns_exclude_dropped_binary_flags():
    dropped = {
        "has_earnings_in_forward_days",
        "has_dividend_in_forward_days",
        "has_fomc_in_forward_days",
        "has_macro_event_in_forward_days",
        "vix_above_20",
        "vix_above_30",
    }
    for col in dropped:
        assert col not in DEFAULT_FEATURE_COLUMNS, f"{col} should have been removed in features_v004"


def test_feature_columns_include_new_v004_features():
    new_features = {"underlying_vol_vs_market", "option_activity_spike", "iv_skew_wing"}
    for col in new_features:
        assert col in DEFAULT_FEATURE_COLUMNS, f"{col} missing from features_v004"
