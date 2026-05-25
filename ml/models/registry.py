"""Lightweight model registry for trained OptionMind artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


PromotionStatus = Literal["candidate", "champion", "rollback", "rejected", "archived"]
REGISTRY_SCHEMA_VERSION = "model_registry_v001"


@dataclass(frozen=True)
class ModelArtifactManifest:
    artifact_path: str
    artifact_sha256: str
    artifact_created_at: str | None
    model_type: str
    target_column: str | None
    model_path: str | None = None
    model_sha256: str | None = None

    @classmethod
    def from_artifact(cls, artifact_path: Path, artifact: dict[str, Any]) -> "ModelArtifactManifest":
        model_path = artifact.get("model_path")
        resolved_model_path = _resolve_related_path(artifact_path, model_path) if model_path else None
        return cls(
            artifact_path=str(artifact_path),
            artifact_sha256=_sha256_file(artifact_path),
            artifact_created_at=artifact.get("created_at"),
            model_type=str(artifact.get("model_type", "unknown")),
            target_column=artifact.get("target_column"),
            model_path=str(resolved_model_path) if resolved_model_path else None,
            model_sha256=_sha256_file(resolved_model_path) if resolved_model_path and resolved_model_path.exists() else None,
        )


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    artifact_manifest: ModelArtifactManifest
    feature_version: str
    label_version: str
    data_range: dict[str, str | None]
    metrics: dict[str, Any]
    promotion_status: PromotionStatus = "candidate"
    registered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    promoted_at: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRegistry:
    schema_version: str = REGISTRY_SCHEMA_VERSION
    champion_model_id: str | None = None
    rollback_model_id: str | None = None
    models: list[ModelRegistryEntry] = field(default_factory=list)

    def get(self, model_id: str) -> ModelRegistryEntry | None:
        return next((entry for entry in self.models if entry.model_id == model_id), None)

    @property
    def champion(self) -> ModelRegistryEntry | None:
        return self.get(self.champion_model_id) if self.champion_model_id else None

    @property
    def rollback(self) -> ModelRegistryEntry | None:
        return self.get(self.rollback_model_id) if self.rollback_model_id else None


def load_registry(path: Path | str = "artifacts/model_registry.json") -> ModelRegistry:
    registry_path = Path(path)
    if not registry_path.exists():
        return ModelRegistry()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return _registry_from_dict(payload)


def save_registry(registry: ModelRegistry, path: Path | str = "artifacts/model_registry.json") -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(_registry_to_dict(registry), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry_path


def register_model_artifact(
    registry: ModelRegistry,
    artifact_path: Path | str,
    *,
    model_id: str | None = None,
    feature_version: str | None = None,
    label_version: str | None = None,
    data_range: dict[str, str | None] | None = None,
    promotion_status: PromotionStatus = "candidate",
    notes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelRegistry:
    path = Path(artifact_path)
    artifact = _load_artifact(path)
    entry = ModelRegistryEntry(
        model_id=model_id or _default_model_id(path, artifact),
        artifact_manifest=ModelArtifactManifest.from_artifact(path, artifact),
        feature_version=feature_version or str(artifact.get("feature_version") or "unknown"),
        label_version=label_version or str(artifact.get("label_version") or "unknown"),
        data_range=data_range or dict(artifact.get("data_range") or {"start": None, "end": None}),
        metrics=dict(artifact.get("metrics") or {}),
        promotion_status=promotion_status,
        notes=notes,
        metadata=metadata or {},
    )
    models = [existing for existing in registry.models if existing.model_id != entry.model_id]
    models.append(entry)
    updated = ModelRegistry(
        schema_version=registry.schema_version,
        champion_model_id=registry.champion_model_id,
        rollback_model_id=registry.rollback_model_id,
        models=models,
    )
    if promotion_status == "champion":
        return promote_model(updated, entry.model_id)
    return updated


def promote_model(
    registry: ModelRegistry,
    model_id: str,
    *,
    rollback_model_id: str | None = None,
    notes: str | None = None,
) -> ModelRegistry:
    if registry.get(model_id) is None:
        raise KeyError(f"Model not found in registry: {model_id}")
    previous_champion = registry.champion_model_id
    rollback_id = rollback_model_id if rollback_model_id is not None else previous_champion
    promoted_at = datetime.now(UTC).isoformat()
    models: list[ModelRegistryEntry] = []
    for entry in registry.models:
        status: PromotionStatus = entry.promotion_status
        entry_promoted_at = entry.promoted_at
        entry_notes = entry.notes
        if entry.model_id == model_id:
            status = "champion"
            entry_promoted_at = promoted_at
            entry_notes = notes if notes is not None else entry.notes
        elif rollback_id and entry.model_id == rollback_id:
            status = "rollback"
        elif entry.promotion_status in {"champion", "rollback"}:
            status = "candidate"
        models.append(
            ModelRegistryEntry(
                model_id=entry.model_id,
                artifact_manifest=entry.artifact_manifest,
                feature_version=entry.feature_version,
                label_version=entry.label_version,
                data_range=entry.data_range,
                metrics=entry.metrics,
                promotion_status=status,
                registered_at=entry.registered_at,
                promoted_at=entry_promoted_at,
                notes=entry_notes,
                metadata=entry.metadata,
            )
        )
    return ModelRegistry(
        schema_version=registry.schema_version,
        champion_model_id=model_id,
        rollback_model_id=rollback_id if rollback_id != model_id else None,
        models=models,
    )


def rollback_champion(registry: ModelRegistry) -> ModelRegistry:
    if not registry.rollback_model_id:
        raise ValueError("No rollback pointer is configured")
    return promote_model(registry, registry.rollback_model_id, rollback_model_id=registry.champion_model_id)


def load_champion_artifact(path: Path | str = "artifacts/model_registry.json") -> tuple[ModelRegistryEntry, dict[str, Any]]:
    registry = load_registry(path)
    champion = registry.champion
    if champion is None:
        raise ValueError(f"No champion model is configured in {path}")
    artifact_path = Path(champion.artifact_manifest.artifact_path)
    return champion, _load_artifact(artifact_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the OptionMind model registry.")
    parser.add_argument("--registry", default="artifacts/model_registry.json")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="Register a trained artifact.")
    register.add_argument("--artifact", required=True)
    register.add_argument("--model-id", default=None)
    register.add_argument("--promote", action="store_true")
    register.add_argument("--notes", default=None)

    promote = sub.add_parser("promote", help="Promote a registered model to champion.")
    promote.add_argument("--model-id", required=True)
    promote.add_argument("--rollback-model-id", default=None)
    promote.add_argument("--notes", default=None)

    sub.add_parser("rollback", help="Promote the rollback pointer back to champion.")
    sub.add_parser("show", help="Print the current registry JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    if args.command == "register":
        registry = register_model_artifact(
            registry,
            args.artifact,
            model_id=args.model_id,
            promotion_status="champion" if args.promote else "candidate",
            notes=args.notes,
        )
        save_registry(registry, args.registry)
    elif args.command == "promote":
        registry = promote_model(
            registry,
            args.model_id,
            rollback_model_id=args.rollback_model_id,
            notes=args.notes,
        )
        save_registry(registry, args.registry)
    elif args.command == "rollback":
        registry = rollback_champion(registry)
        save_registry(registry, args.registry)

    print(json.dumps(_registry_to_dict(registry), indent=2, sort_keys=True))
    return 0


def _load_artifact(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry_from_dict(payload: dict[str, Any]) -> ModelRegistry:
    return ModelRegistry(
        schema_version=payload.get("schema_version", REGISTRY_SCHEMA_VERSION),
        champion_model_id=payload.get("champion_model_id"),
        rollback_model_id=payload.get("rollback_model_id"),
        models=[
            ModelRegistryEntry(
                model_id=item["model_id"],
                artifact_manifest=ModelArtifactManifest(**item["artifact_manifest"]),
                feature_version=item.get("feature_version", "unknown"),
                label_version=item.get("label_version", "unknown"),
                data_range=dict(item.get("data_range") or {"start": None, "end": None}),
                metrics=dict(item.get("metrics") or {}),
                promotion_status=item.get("promotion_status", "candidate"),
                registered_at=item.get("registered_at") or datetime.now(UTC).isoformat(),
                promoted_at=item.get("promoted_at"),
                notes=item.get("notes"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload.get("models", [])
        ],
    )


def _registry_to_dict(registry: ModelRegistry) -> dict[str, Any]:
    return asdict(registry)


def _default_model_id(path: Path, artifact: dict[str, Any]) -> str:
    digest = _sha256_file(path)[:10]
    return f"{artifact.get('model_type', path.stem)}_{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_related_path(artifact_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return artifact_path.parent / path


if __name__ == "__main__":
    raise SystemExit(main())
