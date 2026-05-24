import datetime as dt
from unittest.mock import MagicMock

from src.greeks import bs_greeks, position_risk_score
from src.portfolio_risk import PortfolioRiskService


def _expiry(days=7):
    return (dt.date.today() + dt.timedelta(days=days)).isoformat()


def _cfg(**overrides):
    cfg = {
        'enabled': True,
        'fail_closed': True,
        'max_stress_loss_pct': 0.01,
        'shock_moves_pct': [1, 2, 3],
        'iv_shock_points': 10,
        'max_gamma_loss_to_daily_theta': 10.0,
        'near_expiry_dte': 2,
        'max_near_expiry_stress_pct': 0.01,
        'symbol_stress_cap_enabled': True,
        'max_symbol_stress_pct': 1.0,
        'max_expiry_bucket_pct': 1.0,
        'default_iv': 0.25,
    }
    cfg.update(overrides)
    return {'risk_parameters': {'portfolio_gamma_risk': cfg}}


def _svc(config=None, enrich_side_effect=None):
    risk = MagicMock()
    if enrich_side_effect is not None:
        risk.enrich_position.side_effect = enrich_side_effect
    return PortfolioRiskService(config or _cfg(), position_risk_service=risk)


def _pcs_pick(quantity=10, expiry=None):
    return {
        'strategy': 'PCS',
        'symbol': 'SPY',
        'expiry': expiry or _expiry(7),
        'current_price': 500.0,
        'short_strike': 480.0,
        'long_strike': 475.0,
        'premium': 0.80,
        'quantity': quantity,
        'score': 0.9,
        'short_iv': 0.35,
        'long_iv': 0.36,
    }


def _ic_pick(quantity=3, expiry=None):
    return {
        'strategy': 'IC',
        'symbol': 'SPY',
        'expiry': expiry or _expiry(7),
        'current_price': 500.0,
        'short_put': 480.0,
        'long_put': 475.0,
        'short_call': 520.0,
        'long_call': 525.0,
        'premium': 1.60,
        'quantity': quantity,
        'score': 0.95,
        'short_put_iv': 0.35,
        'long_put_iv': 0.36,
        'short_call_iv': 0.35,
        'long_call_iv': 0.36,
    }


def test_bs_greeks_and_position_risk_include_delta_and_vega():
    one_leg = bs_greeks(500, 480, 0.35, 7, 'put')

    assert 'vega' in one_leg

    risk = position_risk_score(
        500,
        [
            {'strike': 480, 'iv': 0.35, 'option_type': 'put', 'position': 'short'},
            {'strike': 475, 'iv': 0.36, 'option_type': 'put', 'position': 'long'},
        ],
        7,
    )

    assert 'net_delta' in risk
    assert 'net_vega' in risk


def test_portfolio_gamma_gate_reduces_quantity_to_stress_budget():
    svc = _svc(_cfg(max_stress_loss_pct=0.003, max_gamma_loss_to_daily_theta=100.0))
    pick = _pcs_pick(quantity=20)

    filtered = svc.filter_picks([pick], [], account_capital=50_000)

    assert len(filtered) == 1
    assert 1 <= filtered[0]['quantity'] < 20
    assert filtered[0]['portfolio_worst_stress_loss'] <= 151.0


def test_portfolio_gamma_gate_rejects_when_one_contract_breaks_limit():
    svc = _svc(_cfg(max_stress_loss_pct=0.00001, max_gamma_loss_to_daily_theta=100.0))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [], account_capital=50_000)

    assert filtered == []


def test_near_expiry_cap_blocks_short_gamma_concentration():
    svc = _svc(_cfg(max_near_expiry_stress_pct=0.00001, max_gamma_loss_to_daily_theta=100.0))

    filtered = svc.filter_picks([_pcs_pick(quantity=1, expiry=_expiry(1))], [], 50_000)

    assert filtered == []


def test_symbol_stress_cap_blocks_single_name_concentration():
    svc = _svc(_cfg(
        max_stress_loss_pct=1.0,
        max_symbol_stress_pct=0.00001,
        max_gamma_loss_to_daily_theta=100.0,
    ))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [], 50_000)

    assert filtered == []


def test_symbol_stress_cap_floor_keeps_small_budget_usable():
    svc = _svc(_cfg(
        max_stress_loss_pct=1.0,
        max_symbol_stress_pct=0.00001,
        min_symbol_stress_dollars=250.0,
        max_gamma_loss_to_daily_theta=100.0,
    ))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [], 5_000)

    assert len(filtered) == 1


def test_global_stress_cap_floor_is_reflected_in_limits():
    svc = _svc(_cfg(
        max_stress_loss_pct=0.05,
        min_stress_loss_dollars=500.0,
    ))

    summary = svc.summarize_positions([], 5_000)

    assert summary['limits']['max_stress_loss'] == 500.0


def test_complex_spread_quantity_cap_limits_ic_when_symbol_stress_is_high():
    svc = _svc(_cfg(
        max_stress_loss_pct=1.0,
        max_symbol_stress_pct=0.004,
        min_symbol_stress_dollars=0.0,
        complex_spread_quantity_cap={
            'enabled': True,
            'strategies': ['IC', 'IFLY'],
            'symbol_stress_threshold_pct': 0.50,
            'max_quantity': 1,
        },
        max_gamma_loss_to_daily_theta=100.0,
    ))

    filtered = svc.filter_picks([_ic_pick(quantity=3)], [], 50_000)

    assert len(filtered) == 1
    assert filtered[0]['quantity'] == 1


def test_complex_spread_quantity_cap_can_be_disabled():
    svc = _svc(_cfg(
        max_stress_loss_pct=1.0,
        max_symbol_stress_pct=1.0,
        complex_spread_quantity_cap={'enabled': False},
        max_gamma_loss_to_daily_theta=100.0,
    ))

    filtered = svc.filter_picks([_ic_pick(quantity=3)], [], 50_000)

    assert len(filtered) == 1
    assert filtered[0]['quantity'] == 3


def test_existing_open_positions_are_included_in_running_risk():
    enriched = {
        'id': 7,
        'symbol': 'SPY',
        'expiry': _expiry(7),
        'spot': 500.0,
        'contracts': 2,
        'net_delta': 0.05,
        'net_gamma': -0.001,
        'net_theta': 0.03,
        'net_vega': -5.0,
    }
    svc = _svc(_cfg(max_stress_loss_pct=0.002, max_gamma_loss_to_daily_theta=100.0),
               enrich_side_effect=[enriched])

    filtered = svc.filter_picks([_pcs_pick(quantity=20)], [{'id': 7}], 50_000)

    assert len(filtered) <= 1
    if filtered:
        assert filtered[0]['quantity'] < 20


def test_fail_closed_rejects_new_picks_when_open_position_greeks_unavailable():
    svc = _svc(_cfg(fail_closed=True), enrich_side_effect=RuntimeError("data down"))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [{'id': 1}], 50_000)

    assert filtered == []


def test_gamma_loss_to_theta_can_be_warning_only():
    svc = _svc(_cfg(
        max_stress_loss_pct=0.10,
        max_gamma_loss_to_daily_theta=0.01,
        gamma_loss_to_theta_warning_only=True,
        expiry_bucket_cap_enabled=False,
    ))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [], 50_000)

    assert len(filtered) == 1
    assert 'portfolio_max_symbol_stress_loss' in filtered[0]
    assert filtered[0]['portfolio_gamma_loss_to_daily_theta'] > 0.01


def test_expiry_bucket_cap_can_be_disabled():
    svc = _svc(_cfg(
        max_stress_loss_pct=0.10,
        max_gamma_loss_to_daily_theta=100.0,
        expiry_bucket_cap_enabled=False,
        max_expiry_bucket_pct=0.00001,
    ))

    filtered = svc.filter_picks([_pcs_pick(quantity=1)], [], 50_000)

    assert len(filtered) == 1
