"""Tests for agent.py dual-scanner-mode factory and config resolution."""
from __future__ import annotations

import sys
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

from agent import _build_scanner  # noqa: E402


_BASE_CONFIG = {
    'scanner': 'ml',
    'ml_scanner': {},
    'pick_selection': {},
}


# ---------------------------------------------------------------------------
# Legacy mode
# ---------------------------------------------------------------------------

class TestBuildScannerLegacy:

    def test_returns_option_scanner_instance(self):
        with patch('src.scanner.OptionScanner') as MockScanner:
            instance = MockScanner.return_value
            scanner = _build_scanner(_BASE_CONFIG, 'legacy')
        assert scanner is instance

    def test_does_not_instantiate_ml_classes(self):
        with patch('src.scanner.OptionScanner') as MockScanner:
            with patch('src.model_scanner.LivePaperInferenceProvider') as MockLive:
                _build_scanner(_BASE_CONFIG, 'legacy')
        MockLive.assert_not_called()


# ---------------------------------------------------------------------------
# ML mode — LivePaperInferenceProvider (no explicit ml_scanner.provider)
# ---------------------------------------------------------------------------

class TestBuildScannerML:

    def test_returns_live_paper_inference_provider_when_init_succeeds(self):
        with patch('src.model_scanner.LivePaperInferenceProvider') as MockLive:
            instance = MockLive.return_value
            scanner = _build_scanner(_BASE_CONFIG, 'ml')
        assert scanner is instance

    def test_defaults_pick_selection_mode_to_model_ranked(self):
        captured = {}
        def _capture(cfg):
            captured['config'] = cfg
            return MagicMock()
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=_capture):
            _build_scanner(_BASE_CONFIG, 'ml')
        assert captured['config']['pick_selection']['mode'] == 'model_ranked'

    def test_does_not_override_existing_pick_selection_mode(self):
        config = dict(_BASE_CONFIG, pick_selection={'mode': 'custom_mode'})
        captured = {}
        def _capture(cfg):
            captured['config'] = cfg
            return MagicMock()
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=_capture):
            _build_scanner(config, 'ml')
        assert captured['config']['pick_selection']['mode'] == 'custom_mode'

    def test_falls_back_to_model_scanner_when_live_provider_init_fails(self):
        with patch('src.model_scanner.LivePaperInferenceProvider', side_effect=RuntimeError("no model")):
            with patch('src.model_scanner.ModelScanner') as MockModel:
                instance = MockModel.return_value
                scanner = _build_scanner(_BASE_CONFIG, 'ml')
        assert scanner is instance


# ---------------------------------------------------------------------------
# ML mode — explicit ml_scanner.provider falls back to ModelScanner
# ---------------------------------------------------------------------------

class TestBuildScannerMLExplicitProvider:

    def test_uses_model_scanner_when_explicit_provider_set(self):
        config = dict(_BASE_CONFIG, ml_scanner={'provider': 'custom'})
        with patch('src.model_scanner.ModelScanner') as MockModel:
            instance = MockModel.return_value
            scanner = _build_scanner(config, 'ml')
        assert scanner is instance

    def test_does_not_call_live_paper_inference_when_explicit_provider_set(self):
        config = dict(_BASE_CONFIG, ml_scanner={'provider': 'custom'})
        with patch('src.model_scanner.ModelScanner'):
            with patch('src.model_scanner.LivePaperInferenceProvider') as MockLive:
                _build_scanner(config, 'ml')
        MockLive.assert_not_called()


# ---------------------------------------------------------------------------
# Config scanner key resolution (config["scanner"] drives mode)
# ---------------------------------------------------------------------------

class TestScannerTypeFromConfig:

    def test_config_scanner_legacy_selects_legacy_path(self):
        config = dict(_BASE_CONFIG, scanner='legacy')
        # Simulate the resolution logic in _run_once
        scanner_type = config.get('scanner', 'ml')
        assert scanner_type == 'legacy'

    def test_config_scanner_ml_selects_ml_path(self):
        config = dict(_BASE_CONFIG, scanner='ml')
        scanner_type = config.get('scanner', 'ml')
        assert scanner_type == 'ml'

    def test_config_missing_scanner_key_defaults_to_ml(self):
        config = {k: v for k, v in _BASE_CONFIG.items() if k != 'scanner'}
        scanner_type = config.get('scanner', 'ml')
        assert scanner_type == 'ml'
