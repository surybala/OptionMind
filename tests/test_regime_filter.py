from __future__ import annotations

from src.regime import RegimeService


def _cfg(**overrides):
    cfg = {
        "risk_parameters": {
            "regime_filter": {
                "enabled": True,
                "fail_closed": False,
                "yellow_quantity_multiplier": 0.65,
                "orange_quantity_multiplier": 0.30,
                "vix": {
                    "green_below": 18,
                    "yellow_below": 25,
                    "orange_below": 32,
                },
                "vix_spike": {
                    "one_day_pct": 0.15,
                    "red_one_day_pct": 0.25,
                    "three_day_pct": 0.25,
                },
                "trend": {
                    "sma_fast": 20,
                    "sma_slow": 50,
                },
                "realized_vol": {
                    "fast_days": 5,
                    "slow_days": 20,
                    "expansion_multiple": 1.5,
                },
            }
        }
    }
    cfg["risk_parameters"]["regime_filter"].update(overrides)
    return cfg


def test_disabled_filter_returns_green_without_throttle():
    svc = RegimeService({"risk_parameters": {"regime_filter": {"enabled": False}}})
    result = svc.evaluate(vix_current=40)
    assert result.label == "GREEN"
    assert result.quantity_multiplier == 1.0
    assert result.top_n_multiplier == 1.0
    assert result.pause_new_trades is False
    assert result.enabled is False


def test_vix_thresholds_drive_base_regime():
    svc = RegimeService(_cfg())
    assert svc.evaluate(vix_current=17.9).label == "GREEN"
    assert svc.evaluate(vix_current=18.0).label == "YELLOW"
    assert svc.evaluate(vix_current=25.0).label == "ORANGE"
    assert svc.evaluate(vix_current=32.0).label == "RED"


def test_yellow_and_orange_return_configured_throttles():
    svc = RegimeService(_cfg())
    yellow = svc.evaluate(vix_current=20)
    orange = svc.evaluate(vix_current=28)

    assert yellow.label == "YELLOW"
    assert yellow.quantity_multiplier == 0.65
    assert yellow.top_n_multiplier == 1.0
    assert yellow.pause_new_trades is False

    assert orange.label == "ORANGE"
    assert orange.quantity_multiplier == 0.30
    assert orange.top_n_multiplier == 1.0
    assert orange.pause_new_trades is False


def test_red_is_informational_only_for_new_trade_selection():
    svc = RegimeService(_cfg())
    result = svc.evaluate(vix_current=35)
    assert result.label == "RED"
    assert result.quantity_multiplier == 1.0
    assert result.top_n_multiplier == 1.0
    assert result.pause_new_trades is False


def test_red_vix_one_day_spike_overrides_low_vix_level():
    svc = RegimeService(_cfg())
    result = svc.evaluate(vix_current=16, vix_history=[12, 16])
    assert result.label == "RED"
    assert result.pause_new_trades is False


def test_three_day_vix_spike_escalates_to_orange():
    svc = RegimeService(_cfg())
    result = svc.evaluate(vix_current=17, vix_history=[12, 13, 14, 16])
    assert result.label == "ORANGE"
    assert result.pause_new_trades is False


def test_spy_below_averages_with_rising_vix_escalates_to_orange():
    svc = RegimeService(_cfg())
    spy = [100.0] * 45 + [95.0] * 5
    vix = [15.0, 15.2, 15.5, 16.0, 17.0]
    result = svc.evaluate(vix_current=17, vix_history=vix, spy_history=spy)
    assert result.label == "ORANGE"
    assert result.metrics["spy_below_fast"] is True
    assert result.metrics["spy_below_slow"] is True


def test_missing_data_fail_open_or_fail_closed():
    fail_open = RegimeService(_cfg()).evaluate()
    assert fail_open.label == "GREEN"
    assert fail_open.pause_new_trades is False

    fail_closed = RegimeService(_cfg(fail_closed=True)).evaluate()
    assert fail_closed.label == "RED"
    assert fail_closed.pause_new_trades is False
