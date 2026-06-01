from __future__ import annotations

import sys
from pathlib import Path
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

from agent import _validate_ml_hft_runtime  # noqa: E402


def _write_artifact(path: Path) -> None:
    path.write_text("{}", encoding="utf-8")


def _base_config(tmp_path: Path) -> dict:
    registry = tmp_path / "model_registry.json"
    classifier = tmp_path / "large_loss_classifier.json"
    _write_artifact(registry)
    _write_artifact(classifier)
    return {
        "ml_scanner": {
            "enabled": True,
            "registry_path": str(registry),
            "large_loss_classifier_path": str(classifier),
            "provider": "",
            "data_provider": "",
        },
        "pick_selection": {"mode": "model_ranked"},
        "hft_mode": True,
    }


def test_validate_ml_hft_runtime_accepts_strict_ml_hft_config(tmp_path):
    config = _base_config(tmp_path)

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        _validate_ml_hft_runtime(config)


def test_validate_ml_hft_runtime_rejects_non_hft_mode(tmp_path):
    config = _base_config(tmp_path)
    config["hft_mode"] = False

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`hft_mode` must be true." in str(exc)


def test_validate_ml_hft_runtime_rejects_custom_scanner_provider(tmp_path):
    config = _base_config(tmp_path)
    config["ml_scanner"]["provider"] = "custom.module:Provider"

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`ml_scanner.provider` must be empty" in str(exc)


def test_validate_ml_hft_runtime_rejects_non_alpaca_data_provider(tmp_path):
    config = _base_config(tmp_path)
    config["ml_scanner"]["data_provider"] = "custom.module:Provider"

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`ml_scanner.data_provider` must be empty or point to `AlpacaProvider`" in str(exc)


def test_validate_ml_hft_runtime_requires_large_loss_classifier(tmp_path):
    config = _base_config(tmp_path)
    config["ml_scanner"]["large_loss_classifier_path"] = ""

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`ml_scanner.large_loss_classifier_path` must be set." in str(exc)
