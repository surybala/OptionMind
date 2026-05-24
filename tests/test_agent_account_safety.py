import sys
from unittest.mock import MagicMock


_external_mock = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.trading',
    'alpaca.trading.client',
    'alpaca.trading.enums',
    'alpaca.trading.requests',
    'numpy',
    'pandas',
    'scipy',
    'scipy.stats',
    'yfinance',
]:
    sys.modules.setdefault(_mod, _external_mock)

from agent import (  # noqa: E402
    _apply_directional_exposure_caps,
    _directional_exposure,
    _filter_max_loss_multiple,
    _max_loss_multiple_for_pick,
    _max_loss_per_contract_for_pick,
)


def _pcs_pick(symbol='SPY', premium=0.50, quantity=1):
    return {
        'symbol': symbol,
        'strategy': 'PCS',
        'short_strike': 500,
        'long_strike': 495,
        'premium': premium,
        'quantity': quantity,
        'score': 0.9,
    }


def _ccs_pick(symbol='QQQ', premium=0.50, quantity=1):
    return {
        'symbol': symbol,
        'strategy': 'CCS',
        'short_strike': 450,
        'long_strike': 455,
        'premium': premium,
        'quantity': quantity,
        'score': 0.8,
    }


def test_max_loss_multiple_uses_spread_width_minus_credit():
    pick = _pcs_pick(premium=0.50)

    assert _max_loss_per_contract_for_pick(pick) == 450.0
    assert _max_loss_multiple_for_pick(pick) == 9.0


def test_max_loss_multiple_filter_rejects_underpaid_width():
    rich_credit = _pcs_pick(symbol='SPY', premium=0.90)
    thin_credit = _pcs_pick(symbol='QQQ', premium=0.50)

    kept = _filter_max_loss_multiple(
        [rich_credit, thin_credit],
        {'risk_parameters': {'max_loss_multiple': {'enabled': True, 'default': 6.0}}},
    )

    assert kept == [rich_credit]
    assert rich_credit['max_loss_multiple'] < 6.0
    assert thin_credit['max_loss_multiple'] == 9.0


def test_directional_exposure_counts_pending_closes_as_still_deployed():
    positions = [
        {
            'symbol': 'SPY',
            'type': 'PCS',
            'premium': 0.50,
            'contracts': 2,
            'status': 'PENDING_CLOSE',
            'legs': {'short_strike': 500, 'long_strike': 495},
        },
        {
            'symbol': 'QQQ',
            'type': 'CCS',
            'premium': 1.00,
            'contracts': 1,
            'status': 'EXECUTED',
            'legs': {'short_strike': 450, 'long_strike': 455},
        },
    ]

    assert _directional_exposure(positions) == {'put': 900.0, 'call': 400.0}


def test_directional_exposure_cap_reduces_quantity_to_remaining_budget():
    positions = [
        {
            'symbol': 'SPY',
            'type': 'PCS',
            'premium': 0.50,
            'contracts': 2,
            'status': 'EXECUTED',
            'legs': {'short_strike': 500, 'long_strike': 495},
        }
    ]
    picks = [_pcs_pick(symbol='IWM', premium=0.50, quantity=5)]
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.04,
                'call': 0.04,
            }
        }
    }

    sized = _apply_directional_exposure_caps(picks, positions, config, account_capital=50_000)

    assert len(sized) == 1
    assert sized[0]['quantity'] == 2


def test_directional_exposure_cap_rejects_when_side_is_full_but_keeps_other_side():
    positions = [
        {
            'symbol': 'SPY',
            'type': 'PCS',
            'premium': 0.50,
            'contracts': 5,
            'status': 'EXECUTED',
            'legs': {'short_strike': 500, 'long_strike': 495},
        }
    ]
    picks = [
        _pcs_pick(symbol='IWM', premium=0.50, quantity=1),
        _ccs_pick(symbol='QQQ', premium=0.50, quantity=1),
    ]
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.04,
                'call': 0.04,
            }
        }
    }

    sized = _apply_directional_exposure_caps(picks, positions, config, account_capital=50_000)

    assert [p['symbol'] for p in sized] == ['QQQ']
    assert sized[0]['quantity'] == 1


def test_directional_exposure_cap_floor_keeps_small_budget_usable():
    picks = [_pcs_pick(symbol='IWM', premium=0.50, quantity=1)]
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.05,
                'call': 0.05,
                'min_side_cap_dollars': 1500,
            }
        }
    }

    sized = _apply_directional_exposure_caps(picks, [], config, account_capital=5_000)

    assert len(sized) == 1
    assert sized[0]['quantity'] == 1


def test_directional_exposure_cap_floor_allows_one_fifteen_wide_spread():
    pick = {
        'symbol': 'SPY',
        'strategy': 'PCS',
        'short_strike': 500,
        'long_strike': 485,
        'premium': 0.50,
        'quantity': 1,
        'score': 0.9,
    }
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.05,
                'call': 0.05,
                'min_side_cap_dollars': 1500,
            }
        }
    }

    sized = _apply_directional_exposure_caps([pick], [], config, account_capital=5_000)

    assert len(sized) == 1


def test_default_directional_cap_is_gross_concentration_not_tail_loss_budget():
    positions = [
        {
            'symbol': 'SPY',
            'type': 'PCS',
            'premium': 0.50,
            'contracts': 45,
            'status': 'EXECUTED',
            'legs': {'short_strike': 500, 'long_strike': 495},
        },
        {
            'symbol': 'TLT',
            'type': 'CCS',
            'premium': 1.00,
            'contracts': 25,
            'status': 'EXECUTED',
            'legs': {'short_strike': 90, 'long_strike': 95},
        },
    ]
    picks = [
        _pcs_pick(symbol='QQQ', premium=0.50, quantity=10),
        _ccs_pick(symbol='USO', premium=0.50, quantity=10),
    ]
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.50,
                'call': 0.50,
                'min_side_cap_dollars': 1500,
            }
        }
    }

    sized = _apply_directional_exposure_caps(picks, positions, config, account_capital=50_000)

    by_symbol = {p['symbol']: p for p in sized}
    assert by_symbol['QQQ']['quantity'] == 10
    assert by_symbol['USO']['quantity'] == 10


def test_iron_condor_risk_consumes_put_and_call_capacity_evenly():
    pick = {
        'symbol': 'SPY',
        'strategy': 'IC',
        'short_put': 490,
        'long_put': 485,
        'short_call': 510,
        'long_call': 515,
        'premium': 1.00,
        'quantity': 10,
        'score': 0.7,
    }
    config = {
        'risk_parameters': {
            'directional_exposure_caps': {
                'enabled': True,
                'put': 0.04,
                'call': 0.04,
            }
        }
    }

    sized = _apply_directional_exposure_caps([pick], [], config, account_capital=10_000)

    assert len(sized) == 1
    assert sized[0]['quantity'] == 2
