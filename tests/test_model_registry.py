import json

from ml.models.registry import (
    load_champion_artifact,
    load_registry,
    promote_model,
    register_model_artifact,
    rollback_champion,
    save_registry,
)


def _artifact(path, model_type="linear_least_squares_v001", test_mae=1.0):
    path.write_text(
        json.dumps(
            {
                "model_type": model_type,
                "created_at": "2026-05-24T00:00:00+00:00",
                "target_column": "expected_pnl",
                "feature_version": "features_v002",
                "label_version": "short_option_labels_v001",
                "data_range": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-02-01T00:00:00+00:00"},
                "feature_columns": ["dte"],
                "fill_values": {"dte": 30.0},
                "intercept": 0.0,
                "coefficients": {"dte": 1.0},
                "metrics": {"test_mae": test_mae},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_model_registry_promotes_champion_and_sets_rollback(tmp_path):
    first = _artifact(tmp_path / "first.json", test_mae=10.0)
    second = _artifact(tmp_path / "second.json", test_mae=5.0)

    registry = load_registry(tmp_path / "missing.json")
    registry = register_model_artifact(registry, first, model_id="first")
    registry = promote_model(registry, "first")
    registry = register_model_artifact(registry, second, model_id="second")
    registry = promote_model(registry, "second")

    assert registry.champion_model_id == "second"
    assert registry.rollback_model_id == "first"
    assert registry.champion.metrics["test_mae"] == 5.0
    assert registry.rollback.promotion_status == "rollback"


def test_model_registry_save_load_and_load_champion_artifact(tmp_path):
    artifact = _artifact(tmp_path / "artifact.json")
    registry = register_model_artifact(load_registry(tmp_path / "registry.json"), artifact, model_id="m1")
    registry = promote_model(registry, "m1")
    registry_path = save_registry(registry, tmp_path / "registry.json")

    loaded = load_registry(registry_path)
    champion, payload = load_champion_artifact(registry_path)

    assert loaded.champion_model_id == "m1"
    assert champion.artifact_manifest.artifact_sha256
    assert payload["feature_version"] == "features_v002"


def test_model_registry_rollback_promotes_previous_champion(tmp_path):
    first = _artifact(tmp_path / "first.json")
    second = _artifact(tmp_path / "second.json")
    registry = register_model_artifact(load_registry(tmp_path / "registry.json"), first, model_id="first")
    registry = promote_model(registry, "first")
    registry = register_model_artifact(registry, second, model_id="second")
    registry = promote_model(registry, "second")

    rolled_back = rollback_champion(registry)

    assert rolled_back.champion_model_id == "first"
    assert rolled_back.rollback_model_id == "second"
