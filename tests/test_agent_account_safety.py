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
