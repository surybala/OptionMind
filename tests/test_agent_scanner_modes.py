"""Tests for agent.py scanner factory."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

from agent import _build_scanner, _load_tickers  # noqa: E402


_BASE_CONFIG = {
    'ml_scanner': {},
    'pick_selection': {},
}


class TestBuildScannerML:

    def test_returns_live_paper_inference_provider_when_init_succeeds(self):
        with patch('src.model_scanner.LivePaperInferenceProvider') as MockLive:
            instance = MockLive.return_value
            scanner = _build_scanner(_BASE_CONFIG)
        assert scanner is instance

    def test_defaults_pick_selection_mode_to_model_ranked(self):
        captured = {}
        def _capture(cfg):
            captured['config'] = cfg
            return MagicMock()
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=_capture):
            _build_scanner(_BASE_CONFIG)
        assert captured['config']['pick_selection']['mode'] == 'model_ranked'

    def test_does_not_override_existing_pick_selection_mode(self):
        config = dict(_BASE_CONFIG, pick_selection={'mode': 'custom_mode'})
        captured = {}
        def _capture(cfg):
            captured['config'] = cfg
            return MagicMock()
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=_capture):
            _build_scanner(config)
        assert captured['config']['pick_selection']['mode'] == 'custom_mode'

    def test_falls_back_to_model_scanner_when_live_provider_init_fails(self):
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=RuntimeError("no model")):
            with patch('src.model_scanner.ModelScanner') as MockModel:
                instance = MockModel.return_value
                scanner = _build_scanner(_BASE_CONFIG)
        assert scanner is instance


class TestBuildScannerMLExplicitProvider:

    def test_uses_model_scanner_when_explicit_provider_set(self):
        config = dict(_BASE_CONFIG, ml_scanner={'provider': 'custom'})
        with patch('src.model_scanner.ModelScanner') as MockModel:
            instance = MockModel.return_value
            scanner = _build_scanner(config)
        assert scanner is instance

    def test_does_not_call_live_paper_inference_when_explicit_provider_set(self):
        config = dict(_BASE_CONFIG, ml_scanner={'provider': 'custom'})
        with patch('src.model_scanner.ModelScanner'):
            with patch('src.model_scanner.LivePaperInferenceProvider') as MockLive:
                _build_scanner(config)
        MockLive.assert_not_called()


class TestLoadTickers:

    def _args(self, *, universe=None, tickers=None, refresh_universe=False):
        return SimpleNamespace(
            universe=universe,
            tickers=tickers,
            refresh_universe=refresh_universe,
        )

    def test_etf_universe_uses_stable_preset(self):
        with patch('src.universe.get_stable_etf_universe', return_value=['SPY', 'QQQ']) as mock_stable, \
             patch('src.universe.get_etf_universe') as mock_all:
            result = _load_tickers(self._args(universe='etf'), {'universe': 'etf'})

        assert result == ['SPY', 'QQQ']
        mock_stable.assert_called_once()
        mock_all.assert_not_called()

    def test_etf_all_universe_uses_full_listing(self):
        with patch('src.universe.get_etf_universe', return_value=['AAA', 'BBB']) as mock_all:
            result = _load_tickers(
                self._args(universe='etf-all', refresh_universe=True),
                {'universe': 'etf'},
            )

        assert result == ['AAA', 'BBB']
        mock_all.assert_called_once()
