from __future__ import annotations

from datetime import datetime
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo


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

from agent import _next_daemon_run_dt, _validate_ml_hft_runtime  # noqa: E402


_EASTERN = ZoneInfo("US/Eastern")


def _write_artifact(path: Path) -> None:
    path.write_text("{}", encoding="utf-8")


def _write_registry(path: Path) -> None:
    artifact = path.parent / "champion.json"
    model = path.parent / "champion.xgboost.json"
    model.write_text("{}", encoding="utf-8")
    artifact.write_text(
        json.dumps(
            {
                "model_type": "xgboost_asymmetric_pseudohuber_v002",
                "model_path": str(model),
                "target_column": "return_on_risk",
            }
        ),
        encoding="utf-8",
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": "model_registry_v001",
                "champion_model_id": "champion",
                "models": [
                    {
                        "model_id": "champion",
                        "artifact_manifest": {
                            "artifact_path": str(artifact),
                            "artifact_sha256": "unused",
                            "artifact_created_at": None,
                            "model_type": "xgboost_asymmetric_pseudohuber_v002",
                            "target_column": "return_on_risk",
                            "model_path": str(model),
                            "model_sha256": "unused",
                        },
                        "feature_version": "features_test",
                        "label_version": "labels_test",
                        "data_range": {"start": None, "end": None},
                        "metrics": {},
                        "promotion_status": "champion",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _base_config(tmp_path: Path) -> dict:
    registry = tmp_path / "model_registry.json"
    classifier = tmp_path / "large_loss_classifier.json"
    _write_registry(registry)
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


def test_next_daemon_run_dt_runs_immediately_when_starting_inside_trading_window():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=_EASTERN)

    next_dt = _next_daemon_run_dt(
        "09:35",
        "US/Eastern",
        True,
        "09:30",
        "16:00",
        "US/Eastern",
        True,
        immediate_if_in_trading_window=True,
        now=now,
    )

    assert next_dt == now


def test_next_daemon_run_dt_uses_next_scan_time_after_trading_window():
    now = datetime(2026, 6, 1, 16, 1, tzinfo=_EASTERN)

    next_dt = _next_daemon_run_dt(
        "09:35",
        "US/Eastern",
        True,
        "09:30",
        "16:00",
        "US/Eastern",
        True,
        immediate_if_in_trading_window=True,
        now=now,
    )

    assert next_dt == datetime(2026, 6, 2, 9, 35, tzinfo=_EASTERN)


def test_next_daemon_run_dt_can_disable_startup_immediate_scan():
    now = datetime(2026, 6, 1, 12, 0, tzinfo=_EASTERN)

    next_dt = _next_daemon_run_dt(
        "09:35",
        "US/Eastern",
        True,
        "09:30",
        "16:00",
        "US/Eastern",
        True,
        immediate_if_in_trading_window=False,
        now=now,
    )

    assert next_dt == datetime(2026, 6, 2, 9, 35, tzinfo=_EASTERN)


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


def test_validate_ml_hft_runtime_rejects_strategy_specific_rankers(tmp_path):
    config = _base_config(tmp_path)
    config["ml_scanner"]["strategy_rankers"] = {
        "PCS": {"artifact_path": "artifacts/models/pcs.json"},
        "CCS": {"artifact_path": "artifacts/models/ccs.json"},
    }

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`ml_scanner.strategy_rankers` must not be configured" in str(exc)


def test_validate_ml_hft_runtime_requires_champion_model_file(tmp_path):
    config = _base_config(tmp_path)
    registry = json.loads(Path(config["ml_scanner"]["registry_path"]).read_text(encoding="utf-8"))
    model_path = Path(registry["models"][0]["artifact_manifest"]["model_path"])
    model_path.unlink()

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "Champion model file not found" in str(exc)


def test_validate_ml_hft_runtime_requires_large_loss_classifier(tmp_path):
    config = _base_config(tmp_path)
    config["ml_scanner"]["large_loss_classifier_path"] = ""

    with patch("src.alpaca_data.make_alpaca_data_client", return_value=object()):
        try:
            _validate_ml_hft_runtime(config)
            assert False, "expected ValueError"
        except ValueError as exc:
            assert "`ml_scanner.large_loss_classifier_path` must be set." in str(exc)
