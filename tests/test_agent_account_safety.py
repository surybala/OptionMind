import sys
from types import SimpleNamespace
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
    _filter_max_loss_multiple,
    _max_loss_multiple_for_pick,
    _max_loss_per_contract_for_pick,
)
from src.agent_risk import apply_ml_position_sizing, apply_ml_quantity_overlays  # noqa: E402


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


def test_directional_exposure_cap_shrinks_quantity():
    pick = _pcs_pick(quantity=3)
    accepted, rejected = apply_ml_quantity_overlays(
        [pick],
        [],
        {
            'risk_parameters': {
                'directional_exposure_caps': {
                    'enabled': True,
                    'put': 0.05,
                    'call': 0.05,
                    'min_side_cap_dollars': 0.0,
                }
            }
        },
        account_capital=10_000,
        available_capital=10_000,
        max_contracts=3,
    )

    assert rejected == []
    assert accepted[0]['quantity'] == 1


def test_ml_position_sizing_uses_rank_tiers_before_overlays():
    picks = [
        _pcs_pick(symbol='SPY', quantity=1),
        _pcs_pick(symbol='QQQ', quantity=1),
        _pcs_pick(symbol='IWM', quantity=1),
        _pcs_pick(symbol='DIA', quantity=1),
    ]
    picks[0]['score'] = 0.90
    picks[1]['score'] = 0.80
    picks[2]['score'] = 0.70
    picks[3]['score'] = 0.60

    sized = apply_ml_position_sizing(
        picks,
        {
            'max_contracts_per_pick': 4,
            'risk_parameters': {
                'ml_position_sizing': {
                    'enabled': True,
                    'rank_tiers': [
                        {'max_rank': 1, 'quantity': 4},
                        {'max_rank': 3, 'quantity': 3},
                        {'max_rank': 6, 'quantity': 2},
                    ],
                }
            },
        },
        max_contracts=4,
    )

    qty_by_symbol = {pick['symbol']: pick['quantity'] for pick in sized}
    assert qty_by_symbol == {'SPY': 4, 'QQQ': 3, 'IWM': 3, 'DIA': 2}
    assert sized[0]['requested_quantity_basis'] == 'ml_rank_tiers'


def test_correlated_cluster_cap_shrinks_quantity():
    pick = _pcs_pick(symbol='SMH', quantity=3)
    accepted, rejected = apply_ml_quantity_overlays(
        [pick],
        [],
        {
            'risk_parameters': {
                'correlated_cluster_caps': {
                    'enabled': True,
                    'max_cluster_pct': 0.10,
                    'min_cluster_cap_dollars': 0.0,
                    'clusters': {'SEMIS': ['SMH', 'SOXX']},
                }
            }
        },
        account_capital=10_000,
        available_capital=10_000,
        max_contracts=3,
    )

    assert rejected == []
    assert accepted[0]['quantity'] == 2
    assert accepted[0]['correlated_clusters'] == ['SEMIS']


def test_regime_quantity_throttle_shrinks_quantity():
    pick = _pcs_pick(quantity=4)
    accepted, rejected = apply_ml_quantity_overlays(
        [pick],
        [],
        {'risk_parameters': {}},
        account_capital=10_000,
        available_capital=10_000,
        regime=SimpleNamespace(label='ORANGE', quantity_multiplier=0.50),
        max_contracts=4,
    )

    assert rejected == []
    assert accepted[0]['quantity'] == 2
    assert accepted[0]['regime_quantity_multiplier'] == 0.50
