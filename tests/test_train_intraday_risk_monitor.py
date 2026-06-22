import pandas as pd
import pytest

pytest.importorskip("xgboost")

from ml.models.train_intraday_risk_monitor import _engineer_intraday_features, train_intraday_risk_monitor


def test_loss_pct_of_max_loss_uses_contract_dollar_units():
    frame = _frame(entry_count=1, states_per_entry=1)
    frame.loc[0, "max_loss"] = 350.0
    frame.loc[0, "pnl_per_contract"] = -175.0

    engineered = _engineer_intraday_features(frame)

    assert engineered["loss_pct_of_max_loss"].iloc[0] == pytest.approx(0.5)


def _frame(entry_count: int = 12, states_per_entry: int = 3) -> pd.DataFrame:
    rows: list[dict] = []
    for entry_idx in range(entry_count):
        label = 1 if entry_idx % 3 == 0 else 0
        strategy = "PCS" if entry_idx % 2 == 0 else "CCS"
        entry_ts = pd.Timestamp(f"2026-01-{entry_idx + 1:02d}T14:30:00Z")
        for state_idx in range(states_per_entry):
            current_debit = 2.8 + state_idx * 0.15 if label else 0.7 + state_idx * 0.05
            pnl = (1.5 - current_debit) * 100.0
            rows.append(
                {
                    "entry_timestamp": entry_ts.isoformat(),
                    "state_timestamp": (entry_ts + pd.Timedelta(minutes=5 * state_idx)).isoformat(),
                    "underlying": "SPY" if entry_idx % 2 == 0 else "QQQ",
                    "strategy": strategy,
                    "short_option_symbol": f"OPT{entry_idx:03d}S",
                    "long_option_symbol": f"OPT{entry_idx:03d}L",
                    "dte": 14 - (entry_idx % 5),
                    "spread_width": 5.0,
                    "entry_credit": 1.5,
                    "max_loss": 3.5,
                    "stop_debit": 3.2,
                    "profit_take_debit": 0.4,
                    "current_debit": current_debit,
                    "pnl_per_contract": pnl,
                    "profit_captured_pct": ((1.5 - current_debit) / 1.5) * 100.0,
                    "stop_distance_pct": ((3.2 - current_debit) / 3.2) * 100.0,
                    "minutes_since_entry": float(state_idx * 5),
                    "minutes_to_expiry": float((14 - (entry_idx % 5)) * 390),
                    "underlying_close": 500.0 + entry_idx,
                    "underlying_return_5m": 0.001 * (state_idx + 1),
                    "underlying_return_15m": 0.002 * (state_idx + 1),
                    "underlying_return_30m": 0.003 * (state_idx + 1),
                    "underlying_realized_vol_15m": 0.15 + 0.01 * state_idx,
                    "underlying_realized_vol_30m": 0.18 + 0.01 * state_idx,
                    "short_leg_close": current_debit * 0.7,
                    "long_leg_close": current_debit * 0.3,
                    "short_leg_volume": 10 + state_idx,
                    "long_leg_volume": 6 + state_idx,
                    "short_leg_trade_count": 5 + state_idx,
                    "long_leg_trade_count": 4 + state_idx,
                    "market_trend_regime": "uptrend" if entry_idx % 2 == 0 else "downtrend",
                    "market_volatility_regime": "low" if entry_idx % 2 == 0 else "high",
                    "minutes_to_exit": float(30 - state_idx * 5),
                    "stop_loss_hit_5m": int(label and state_idx >= 2),
                    "stop_loss_hit_15m": int(label),
                    "stop_loss_hit_30m": int(label),
                    "profit_take_hit_5m": int((not label) and state_idx >= 2),
                    "profit_take_hit_15m": int(not label),
                    "profit_take_hit_30m": int(not label),
                }
            )
    return pd.DataFrame(rows)


def test_train_intraday_risk_monitor_returns_artifact(tmp_path):
    artifact = train_intraday_risk_monitor(
        _frame(),
        model_output=tmp_path / "intraday_risk.json",
        test_fraction=0.25,
        walk_forward_folds=2,
        min_walk_forward_train_groups=4,
        embargo_days=0,
        num_boost_round=10,
        val_fraction=0.0,
        early_stopping_rounds=0,
    )

    assert artifact.model_type == "xgboost_intraday_risk_monitor_v001"
    assert artifact.target_column == "stop_loss_hit_30m"
    assert artifact.train_rows > 0
    assert artifact.test_rows > 0
    assert artifact.train_entries + artifact.test_entries == 12
    assert 0.05 <= artifact.recommended_close_threshold <= 0.95
    assert artifact.metrics["walk_forward_folds"] == 2
    assert len(artifact.walk_forward) == 2
    assert (tmp_path / "intraday_risk.json").exists()


def test_train_intraday_risk_monitor_respects_feature_filters(tmp_path):
    artifact = train_intraday_risk_monitor(
        _frame(),
        model_output=tmp_path / "intraday_risk.json",
        test_fraction=0.25,
        walk_forward_folds=2,
        min_walk_forward_train_groups=4,
        embargo_days=0,
        num_boost_round=10,
        val_fraction=0.0,
        early_stopping_rounds=0,
        include_features=["is_pcs", "is_ccs", "dte", "underlying_return_5m", "market_trend_uptrend"],
        exclude_features=["market_trend_uptrend"],
    )

    assert artifact.feature_columns == ["is_pcs", "is_ccs", "dte", "underlying_return_5m"]


def test_train_intraday_risk_monitor_requires_target(tmp_path):
    frame = _frame()
    frame = frame.drop(columns=["stop_loss_hit_30m"])

    with pytest.raises(ValueError, match="stop_loss_hit_30m"):
        train_intraday_risk_monitor(
            frame,
            model_output=tmp_path / "intraday_risk.json",
            num_boost_round=5,
            val_fraction=0.0,
            early_stopping_rounds=0,
        )


def test_train_intraday_risk_monitor_holdout_uses_timestamp_embargo(tmp_path):
    artifact = train_intraday_risk_monitor(
        _frame(entry_count=16),
        model_output=tmp_path / "intraday_risk.json",
        test_fraction=0.25,
        walk_forward_folds=2,
        min_walk_forward_train_groups=4,
        embargo_days=1,
        num_boost_round=10,
        val_fraction=0.0,
        early_stopping_rounds=0,
    )

    assert artifact.split_summary["actual_gap_days"] is not None
    assert artifact.split_summary["actual_gap_days"] >= 1.0
