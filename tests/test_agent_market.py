from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd

_external_mock = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.trading',
    'alpaca.trading.client',
    'alpaca.trading.enums',
    'alpaca.trading.requests',
]:
    sys.modules.setdefault(_mod, _external_mock)

from src.agent_market import _fetch_regime_history, fetch_vix


class _ExplodingYFinance:
    def __getattr__(self, name):
        raise AssertionError(f"yfinance should not be used in HFT mode (attr={name})")


def test_fetch_vix_hft_uses_alpaca_only(tmp_path):
    cache_path = tmp_path / "vix_cache.json"
    client = MagicMock()
    client.get_spot_price.side_effect = lambda symbol: 18.75 if symbol == "VIX" else None

    with patch("src.agent_market._VIX_CACHE_PATH", str(cache_path)):
        with patch("src.alpaca_data.make_alpaca_data_client", return_value=client):
            with patch.dict(sys.modules, {"yfinance": _ExplodingYFinance()}):
                result = fetch_vix({"hft_mode": True})

    assert result == 18.75


def test_fetch_regime_history_hft_uses_alpaca_only():
    client = MagicMock()
    client.get_bulk_history.return_value = {
        "SPY": pd.Series([100.0, 101.5, 103.0])
    }

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=client):
        with patch.dict(sys.modules, {"yfinance": _ExplodingYFinance()}):
            result = _fetch_regime_history("SPY", config={"hft_mode": True})

    assert result == [100.0, 101.5, 103.0]


def test_fetch_regime_history_hft_maps_vix_symbol_to_alpaca():
    client = MagicMock()
    client.get_bulk_history.return_value = {
        "VIX": pd.Series([16.0, 16.5, 17.0])
    }

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=client):
        result = _fetch_regime_history("^VIX", config={"hft_mode": True})

    assert result == [16.0, 16.5, 17.0]
    client.get_bulk_history.assert_called_once()
