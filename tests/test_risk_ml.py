from __future__ import annotations

import datetime
import json
from unittest.mock import MagicMock

from ml.models.registry import ModelRegistry, register_model_artifact, save_registry
from src.position_monitor import PositionMonitor
from src.risk_ml import MlExitRiskService


def _ml_exit_artifact(tmp_path, filename: str = "ml_exit_linear.json"):
    artifact_path = tmp_path / filename
    artifact_path.write_text(
        json.dumps(
            {
                "model_type": "linear_least_squares_v001",
                "created_at": "2026-06-01T00:00:00+00:00",
                "target_column": "p_stop_loss_15m",
                "feature_version": "ml_exit_features_v001",
                "label_version": "ml_exit_labels_v001",
                "feature_columns": ["risk_score", "stop_proximity", "short_leg_bid_ask_spread_pct"],
                "fill_values": {
                    "risk_score": 0.0,
                    "stop_proximity": 0.0,
                    "short_leg_bid_ask_spread_pct": 0.0,
                },
                "intercept": 0.0,
                "coefficients": {
                    "risk_score": 1.0,
                    "stop_proximity": 0.0,
                    "short_leg_bid_ask_spread_pct": 0.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _monitor_config(tmp_path, **overrides):
    cfg = {
        "risk_parameters": {
            "stop_loss_multiplier": 99.0,
            "profit_take_enabled": False,
            "gamma_risk": {"enabled": False},
            "ml_exit_risk": {
                "enabled": True,
                "artifact_path": str(_ml_exit_artifact(tmp_path)),
                "threshold": 0.70,
                "confirmations_required": 1,
                "min_age_minutes": 0.0,
            },
        }
    }
    cfg["risk_parameters"]["ml_exit_risk"].update(overrides)
    return cfg


def _base_position() -> dict:
    entry_time = datetime.datetime.now() - datetime.timedelta(minutes=45)
    return {
        "id": 101,
        "symbol": "AAPL",
        "type": "PCS",
        "expiry": "2099-12-31",
        "timestamp": entry_time.isoformat(),
        "premium": 1.00,
        "contracts": 1,
        "legs": {"short_strike": 100.0, "long_strike": 95.0},
    }


def test_ml_exit_risk_service_scores_trigger(tmp_path):
    service = MlExitRiskService(_monitor_config(tmp_path))
    pos = _base_position()

    payload = service.score_position(
        pos,
        current_mark=1.20,
        spot=99.0,
        risk={"risk_score": 0.92, "gamma_theta_ratio": 2.4, "net_short_delta": -0.31},
    )

    assert payload is not None
    assert payload["ml_exit_risk_score"] == 0.92
    assert payload["ml_exit_risk_should_trigger"] is True
    assert payload["ml_exit_risk_guard_reason"] is None


def test_ml_exit_risk_service_loads_champion_from_registry(tmp_path):
    artifact_path = _ml_exit_artifact(tmp_path, "registry_model.json")
    registry = register_model_artifact(
        ModelRegistry(),
        artifact_path,
        model_id="risk_v003",
        promotion_status="champion",
    )
    registry_path = tmp_path / "risk_model_registry.json"
    save_registry(registry, registry_path)
    service = MlExitRiskService(
        {
            "risk_parameters": {
                "profit_take_enabled": False,
                "gamma_risk": {"enabled": False},
                "ml_exit_risk": {
                    "enabled": True,
                    "registry_path": str(registry_path),
                    "threshold": 0.70,
                    "confirmations_required": 1,
                    "min_age_minutes": 0.0,
                },
            }
        }
    )

    payload = service.score_position(
        _base_position(),
        current_mark=1.20,
        spot=99.0,
        risk={"risk_score": 0.92},
    )

    assert payload is not None
    assert payload["ml_exit_risk_should_trigger"] is True
    assert payload["ml_exit_risk_model_id"] == "risk_v003"


def test_ml_exit_risk_service_passes_optional_intraday_features(tmp_path):
    service = MlExitRiskService(_monitor_config(tmp_path))
    pos = _base_position()

    row = service._build_feature_row(
        pos,
        current_mark=1.20,
        spot=99.0,
        risk={
            "underlying_return_5m": 0.01,
            "underlying_realized_vol_15m": 0.22,
            "underlying_vol_ratio_15m_30m": 1.15,
            "short_leg_close": 0.9,
            "long_leg_close": 0.3,
            "short_leg_volume": 12,
            "long_leg_volume": 8,
            "short_leg_trade_count": 6,
            "long_leg_trade_count": 4,
            "leg_volume_imbalance": 0.2,
            "leg_trade_count_imbalance": 0.2,
            "market_trend_regime": "uptrend",
            "market_volatility_regime": "high",
        },
        chain=None,
    )

    assert row["underlying_return_5m"] == 0.01
    assert row["underlying_realized_vol_15m"] == 0.22
    assert row["market_trend_uptrend"] == 1.0
    assert row["market_volatility_high"] == 1.0


def test_ml_exit_risk_service_applies_min_age_guard(tmp_path):
    service = MlExitRiskService(_monitor_config(tmp_path, min_age_minutes=120.0))
    pos = _base_position()

    payload = service.score_position(
        pos,
        current_mark=1.10,
        spot=99.5,
        risk={"risk_score": 0.95},
    )

    assert payload is not None
    assert payload["ml_exit_risk_should_trigger"] is False
    assert payload["ml_exit_risk_guard_reason"] == "below_min_age_minutes"


def test_ml_exit_risk_service_still_scores_without_greek_risk_payload(tmp_path):
    service = MlExitRiskService(_monitor_config(tmp_path))
    pos = _base_position()

    payload = service.score_position(
        pos,
        current_mark=1.20,
        spot=99.0,
        risk=None,
        chain=None,
    )

    assert payload is not None
    assert "ml_exit_risk_score" in payload
    assert payload["ml_exit_risk_model_id"] is not None


def test_position_monitor_ml_exit_requires_confirmations(tmp_path):
    monitor = PositionMonitor(MagicMock(), MagicMock(), _monitor_config(tmp_path, confirmations_required=2))
    monitor._execute_close = MagicMock(return_value={"closed": True})
    pos = _base_position()

    metrics_fn = lambda: (
        9,
        {
            "risk_score": 0.91,
            "gamma_theta_ratio": 2.1,
            "net_short_delta": -0.28,
            "net_delta": 0.0,
            "net_gamma": -0.01,
            "net_theta": 0.005,
            "net_vega": -0.02,
        },
    )

    result1 = monitor._apply_triggers(
        pos,
        dry_run=True,
        entry_premium=1.00,
        current_mark=1.10,
        spot=99.0,
        gamma_risk_fn=lambda *_args: None,
        metrics_fn=metrics_fn,
    )
    result2 = monitor._apply_triggers(
        pos,
        dry_run=True,
        entry_premium=1.00,
        current_mark=1.10,
        spot=99.0,
        gamma_risk_fn=lambda *_args: None,
        metrics_fn=metrics_fn,
    )

    assert result1 is None
    assert result2 == {"closed": True}
    assert monitor._execute_close.call_count == 1
    assert monitor._execute_close.call_args.args[3] == "ML_RISK_EXIT"
    assert pos["ml_exit_risk_confirmation_count"] == 2


def test_position_monitor_uses_ml_before_stop_loss_fallback(tmp_path):
    cfg = {
        "risk_parameters": {
            "stop_loss_multiplier": 0.05,
            "profit_take_enabled": False,
            "gamma_risk": {"enabled": False},
            "ml_exit_risk": {
                "enabled": True,
                "artifact_path": str(_ml_exit_artifact(tmp_path)),
                "threshold": 0.70,
                "confirmations_required": 1,
                "min_age_minutes": 0.0,
            },
        }
    }
    monitor = PositionMonitor(MagicMock(), MagicMock(), cfg)
    monitor._execute_close = MagicMock(return_value={"closed": True})
    pos = _base_position()

    result = monitor._apply_triggers(
        pos,
        dry_run=True,
        entry_premium=1.00,
        current_mark=1.10,
        spot=99.0,
        gamma_risk_fn=lambda *_args: ("reason", "extras", {}),
        metrics_fn=lambda: (9, {"risk_score": 0.91, "gamma_theta_ratio": 2.1, "net_short_delta": -0.28}),
    )

    assert result == {"closed": True}
    assert monitor._execute_close.call_args.args[3] == "ML_RISK_EXIT"


def test_position_monitor_uses_ml_before_profit_take(tmp_path):
    cfg = {
        "risk_parameters": {
            "stop_loss_multiplier": 99.0,
            "profit_take_enabled": True,
            "profit_take_pct": 0.50,
            "gamma_risk": {"enabled": False},
            "ml_exit_risk": {
                "enabled": True,
                "artifact_path": str(_ml_exit_artifact(tmp_path)),
                "threshold": 0.70,
                "confirmations_required": 1,
                "min_age_minutes": 0.0,
            },
        }
    }
    monitor = PositionMonitor(MagicMock(), MagicMock(), cfg)
    monitor._execute_close = MagicMock(return_value={"closed": True})
    pos = _base_position()

    result = monitor._apply_triggers(
        pos,
        dry_run=True,
        entry_premium=1.00,
        current_mark=0.40,
        spot=99.0,
        gamma_risk_fn=lambda *_args: None,
        metrics_fn=lambda: (9, {"risk_score": 0.91, "gamma_theta_ratio": 2.1, "net_short_delta": -0.28}),
    )

    assert result == {"closed": True}
    assert monitor._execute_close.call_args.args[3] == "ML_RISK_EXIT"


def test_get_risk_snapshot_includes_ml_exit_score(tmp_path):
    db = MagicMock()
    executor = MagicMock()
    monitor = PositionMonitor(db, executor, _monitor_config(tmp_path))
    pos = _base_position()
    db.get_open_positions.return_value = [pos]
    monitor._risk_service.enrich_position = MagicMock(
        return_value={
            **pos,
            "current_mark": 1.15,
            "pnl_per_share": -0.15,
            "spot": 99.0,
            "risk_score": 0.88,
            "gamma_theta_ratio": 1.9,
            "net_short_delta": 0.22,
            "risk_level": "CAUTION",
        }
    )

    snapshot = monitor.get_risk_snapshot()

    assert len(snapshot) == 1
    assert snapshot[0]["ml_exit_risk_score"] == 0.88
    assert snapshot[0]["ml_exit_risk_should_trigger"] is True
