"""Run an agent-like ML backtest over a date range and summarize results."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.evaluate_exit_criteria import _prediction_to_target_space
from ml.models.evaluate_risk_adjusted_ranking import _score_classifier, apply_large_loss_gate
from ml.models.portfolio_controls import apply_portfolio_risk_controls
from ml.models.train_baseline import _max_drawdown, _profit_factor
from ml.models.train_xgboost import _engineer_features, _transform_xgb_frame
from src.utils import load_config

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent-like ML backtest over a calendar range.")
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--year", type=int, default=2025, help="Calendar year to analyze.")
    parser.add_argument("--config", default="config.json", help="Runtime config path.")
    parser.add_argument("--registry", default="artifacts/model_registry.json", help="Model registry path.")
    parser.add_argument("--ranker-artifact", default=None, help="Optional explicit ranker artifact path.")
    parser.add_argument("--pcs-ranker-artifact", default=None, help="Optional explicit PCS ranker artifact path.")
    parser.add_argument("--ccs-ranker-artifact", default=None, help="Optional explicit CCS ranker artifact path.")
    parser.add_argument("--large-loss-artifact", default=None, help="Optional explicit large-loss artifact path.")
    parser.add_argument("--json-output", default=None, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_backtest(
        dataset_path=Path(args.input),
        year=args.year,
        config_path=Path(args.config),
        registry_path=Path(args.registry),
        ranker_artifact_path=Path(args.ranker_artifact) if args.ranker_artifact else None,
        pcs_ranker_artifact_path=Path(args.pcs_ranker_artifact) if args.pcs_ranker_artifact else None,
        ccs_ranker_artifact_path=Path(args.ccs_ranker_artifact) if args.ccs_ranker_artifact else None,
        large_loss_artifact_path=Path(args.large_loss_artifact) if args.large_loss_artifact else None,
    )
    payload = json.loads(json.dumps(report, default=str))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_backtest(
    *,
    dataset_path: Path,
    year: int,
    config_path: Path,
    registry_path: Path,
    ranker_artifact_path: Path | None = None,
    pcs_ranker_artifact_path: Path | None = None,
    ccs_ranker_artifact_path: Path | None = None,
    large_loss_artifact_path: Path | None = None,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required for ML backtesting.")

    repo_root = config_path.parent.resolve()
    config = load_config(str(config_path))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ranker_artifact_path = ranker_artifact_path or _resolve_ranker_artifact(repo_root, registry)
    strategy_ranker_paths = _resolve_strategy_ranker_paths(
        repo_root,
        config,
        default_ranker=ranker_artifact_path,
        pcs_ranker_artifact_path=pcs_ranker_artifact_path,
        ccs_ranker_artifact_path=ccs_ranker_artifact_path,
    )
    large_loss_artifact_path = large_loss_artifact_path or Path(
        str(config.get("ml_scanner", {}).get("large_loss_classifier_path") or "")
    )
    if not large_loss_artifact_path.is_absolute():
        large_loss_artifact_path = repo_root / large_loss_artifact_path
    if not ranker_artifact_path.is_absolute():
        ranker_artifact_path = repo_root / ranker_artifact_path

    df = load_dataset(dataset_path)
    df = _filter_year(df, year)
    scored = _score_rankers(df, strategy_ranker_paths)
    scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact_path)
    scored["gated_score"] = apply_large_loss_gate(
        scored["prediction"],
        scored["large_loss_probability"],
        max_large_loss_probability=float(config.get("ml_scanner", {}).get("large_loss_veto_threshold", 0.70)),
    )
    scored["gate_stage"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["gate_reason"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["gate_violation_codes"] = pd.Series(pd.NA, index=scored.index, dtype="string")
    scored["directional_reduced"] = False
    scored["portfolio_gamma_reduced"] = False
    large_loss_threshold = float(config.get("ml_scanner", {}).get("large_loss_veto_threshold", 0.70))
    large_loss_veto = ~np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce"))
    scored.loc[large_loss_veto, "gate_stage"] = "large_loss_veto"
    scored.loc[large_loss_veto, "gate_reason"] = (
        "p_large_loss>" + pd.to_numeric(scored.loc[large_loss_veto, "large_loss_probability"], errors="coerce").round(4).astype(str)
    )
    scored["allocator_regime_label"] = _allocator_regime_labels(scored)
    portfolio_result = apply_portfolio_risk_controls(
        scored,
        "gated_score",
        account_capital=float(config.get("max_capital_per_period", 50_000.0)),
        scanner_controls=True,
        scanner_config=config,
        regime_label_column="allocator_regime_label",
        return_diagnostics=True,
    )
    scored["portfolio_score"], portfolio_diagnostics = portfolio_result
    for column in (
        "gate_stage",
        "gate_reason",
        "gate_violation_codes",
        "directional_reduced",
        "portfolio_gamma_reduced",
    ):
        scored.loc[:, column] = portfolio_diagnostics[column]
    scored.loc[large_loss_veto, "gate_stage"] = "large_loss_veto"
    scored.loc[large_loss_veto, "gate_reason"] = f"p_large_loss>{large_loss_threshold:.2f}"
    scored.loc[large_loss_veto, "gate_violation_codes"] = pd.NA
    scored.loc[large_loss_veto, "directional_reduced"] = False
    scored.loc[large_loss_veto, "portfolio_gamma_reduced"] = False
    scored["quarter"] = pd.PeriodIndex(pd.to_datetime(scored["entry_timestamp"]), freq="Q").astype(str)
    selected = scored[np.isfinite(pd.to_numeric(scored["portfolio_score"], errors="coerce"))].copy()
    selected = selected.sort_values(["entry_timestamp", "portfolio_score"], ascending=[True, False])

    quarters = [f"{year}Q1", f"{year}Q2", f"{year}Q3", f"{year}Q4"]
    quarterly = {
        quarter: _quarter_report(scored, selected, quarter)
        for quarter in quarters
    }
    overall = _overall_report(scored, selected, year)
    return {
        "dataset_path": str(dataset_path),
        "year": year,
        "config_path": str(config_path),
        "registry_path": str(registry_path),
        "ranker_artifact": str(ranker_artifact_path),
        "strategy_rankers": {key: str(value.path) for key, value in strategy_ranker_paths.items()},
        "large_loss_artifact": str(large_loss_artifact_path),
        "runtime": {
            "pick_selection_mode": ((config.get("pick_selection") or {}).get("mode")),
            "scanner_top_n": ((config.get("ml_scanner") or {}).get("top_n")),
            "large_loss_veto_threshold": float(config.get("ml_scanner", {}).get("large_loss_veto_threshold", 0.70)),
            "max_capital_per_period": float(config.get("max_capital_per_period", 50_000.0)),
            "regime_allocator_enabled": bool(((config.get("pick_selection") or {}).get("regime_allocation") or {}).get("enabled", False)),
            "allocator_regime_source": "dataset_proxy_v1",
        },
        "gate_diagnostics": {
            "gate_stage_counts": _gate_stage_counts(scored),
            "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(scored),
            "quantity_reduction_counts": _quantity_reduction_counts(scored),
        },
        "overall": overall,
        "quarters": quarterly,
    }


def _resolve_ranker_artifact(repo_root: Path, registry: dict[str, Any]) -> Path:
    champion_id = registry.get("champion_model_id")
    for model in registry.get("models", []):
        if model.get("model_id") == champion_id:
            artifact_path = model.get("artifact_manifest", {}).get("artifact_path")
            if not artifact_path:
                break
            return repo_root / artifact_path
    raise ValueError(f"Unable to resolve champion artifact for {champion_id!r}")


@dataclass(frozen=True)
class _ScoringArtifact:
    strategy: str
    path: Path
    artifact: dict[str, Any]
    feature_columns: list[str]
    fill_values: dict[str, float]
    booster: Any


def _resolve_strategy_ranker_paths(
    repo_root: Path,
    config: dict[str, Any],
    *,
    default_ranker: Path,
    pcs_ranker_artifact_path: Path | None,
    ccs_ranker_artifact_path: Path | None,
) -> dict[str, _ScoringArtifact]:
    configured = (
        config.get("ml_scanner", {}).get("strategy_rankers", {})
        if isinstance(config.get("ml_scanner", {}).get("strategy_rankers", {}), dict)
        else {}
    )
    raw_paths: dict[str, Path] = {
        "DEFAULT": default_ranker,
    }
    if pcs_ranker_artifact_path is not None:
        raw_paths["PCS"] = pcs_ranker_artifact_path
    elif "PCS" in configured:
        raw_paths["PCS"] = _resolve_source_path(repo_root, configured["PCS"])
    if ccs_ranker_artifact_path is not None:
        raw_paths["CCS"] = ccs_ranker_artifact_path
    elif "CCS" in configured:
        raw_paths["CCS"] = _resolve_source_path(repo_root, configured["CCS"])
    return {
        strategy: _load_scoring_artifact(strategy, repo_root / path if not path.is_absolute() else path)
        for strategy, path in raw_paths.items()
    }


def _resolve_source_path(repo_root: Path, source: Any) -> Path:
    if isinstance(source, str):
        return Path(source)
    if not isinstance(source, dict):
        raise TypeError("strategy_rankers entries must be a path string or config object")
    artifact_path = str(source.get("artifact_path") or "").strip()
    if artifact_path:
        return Path(artifact_path)
    registry_path = str(source.get("registry_path") or "").strip()
    if not registry_path:
        raise ValueError("strategy_rankers entry must set artifact_path or registry_path")
    registry_payload = json.loads((repo_root / registry_path).read_text(encoding="utf-8"))
    model_id = str(source.get("model_id") or "").strip()
    target_id = model_id or registry_payload.get("champion_model_id")
    for model in registry_payload.get("models", []):
        if model.get("model_id") == target_id:
            return Path(str(model.get("artifact_manifest", {}).get("artifact_path")))
    raise ValueError(f"Unable to resolve strategy ranker {target_id!r} from {registry_path}")


def _filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if "entry_timestamp" not in df:
        raise ValueError("Dataset is missing entry_timestamp; cannot run calendar backtest.")
    working = df.copy()
    working["entry_timestamp"] = pd.to_datetime(working["entry_timestamp"], errors="coerce", utc=True)
    working = working.dropna(subset=["entry_timestamp"])
    return working.loc[working["entry_timestamp"].dt.year == year].copy()


def _load_scoring_artifact(strategy: str, path: Path) -> _ScoringArtifact:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    feature_columns = list(artifact["feature_columns"])
    fill_values = dict(artifact.get("fill_values") or {})
    booster = xgb.Booster()
    booster.load_model(str(artifact["model_path"]))
    return _ScoringArtifact(
        strategy=strategy,
        path=path,
        artifact=artifact,
        feature_columns=feature_columns,
        fill_values=fill_values,
        booster=booster,
    )


def _score_rankers(df: pd.DataFrame, artifacts: dict[str, _ScoringArtifact]) -> pd.DataFrame:
    engineered = _engineer_features(df.copy())
    engineered["prediction"] = np.nan
    engineered["ranker_strategy"] = "DEFAULT"

    for strategy, artifact in artifacts.items():
        if strategy == "DEFAULT":
            index = engineered.index
        else:
            index = engineered.index[
                engineered.get("strategy", pd.Series(index=engineered.index, dtype=object)).astype(str).str.upper() == strategy
            ]
        if len(index) == 0:
            continue
        frame = _transform_xgb_frame(
            engineered.loc[index],
            artifact.feature_columns,
            dict(artifact.fill_values),
        )
        raw_prediction = artifact.booster.predict(xgb.DMatrix(frame, feature_names=list(frame.columns)))
        engineered.loc[index, "prediction"] = _prediction_to_target_space(raw_prediction, artifact.artifact)
        engineered.loc[index, "ranker_strategy"] = strategy

    if engineered["prediction"].isna().any():
        default_artifact = artifacts["DEFAULT"]
        missing = engineered["prediction"].isna()
        frame = _transform_xgb_frame(
            engineered.loc[missing],
            default_artifact.feature_columns,
            dict(default_artifact.fill_values),
        )
        raw_prediction = default_artifact.booster.predict(xgb.DMatrix(frame, feature_names=list(frame.columns)))
        engineered.loc[missing, "prediction"] = _prediction_to_target_space(raw_prediction, default_artifact.artifact)
        engineered.loc[missing, "ranker_strategy"] = "DEFAULT"
    return engineered


def _overall_report(scored: pd.DataFrame, selected: pd.DataFrame, year: int) -> dict[str, Any]:
    universe = _selection_stats(scored)
    gated = _selection_stats(scored[np.isfinite(pd.to_numeric(scored["gated_score"], errors="coerce"))])
    chosen = _selection_stats(selected)
    return {
        "label": str(year),
        "universe": universe,
        "post_large_loss_gate": gated,
        "selected": chosen,
        "gated_row_rejection_rate": _rate_from_counts(universe["rows"] - gated["rows"], universe["rows"]),
        "portfolio_selection_rate": _rate_from_counts(chosen["rows"], gated["rows"]),
        "baseline_mean_pnl_delta": _delta(chosen.get("mean_pnl"), universe.get("mean_pnl")),
        "baseline_profit_factor_delta": _delta(chosen.get("profit_factor"), universe.get("profit_factor")),
        "gate_stage_counts": _gate_stage_counts(scored),
        "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(scored),
        "quantity_reduction_counts": _quantity_reduction_counts(scored),
    }


def _quarter_report(scored: pd.DataFrame, selected: pd.DataFrame, quarter: str) -> dict[str, Any]:
    quarter_scored = scored.loc[scored["quarter"] == quarter].copy()
    quarter_selected = selected.loc[selected["quarter"] == quarter].copy()
    universe = _selection_stats(quarter_scored)
    gated = _selection_stats(
        quarter_scored[np.isfinite(pd.to_numeric(quarter_scored["gated_score"], errors="coerce"))].copy()
    )
    chosen = _selection_stats(quarter_selected)
    return {
        "label": quarter,
        "universe": universe,
        "post_large_loss_gate": gated,
        "selected": chosen,
        "gated_row_rejection_rate": _rate_from_counts(universe["rows"] - gated["rows"], universe["rows"]),
        "portfolio_selection_rate": _rate_from_counts(chosen["rows"], gated["rows"]),
        "baseline_mean_pnl_delta": _delta(chosen.get("mean_pnl"), universe.get("mean_pnl")),
        "baseline_profit_factor_delta": _delta(chosen.get("profit_factor"), universe.get("profit_factor")),
        "gate_stage_counts": _gate_stage_counts(quarter_scored),
        "portfolio_gamma_violation_counts": _portfolio_gamma_violation_counts(quarter_scored),
        "quantity_reduction_counts": _quantity_reduction_counts(quarter_scored),
    }


def _selection_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "entry_dates": 0,
            "avg_trades_per_entry_date": None,
            "total_pnl": None,
            "mean_pnl": None,
            "median_pnl": None,
            "profit_factor": None,
            "win_rate": None,
            "mean_return_on_risk": None,
            "large_loss_rate": None,
            "stop_loss_rate": None,
            "p05_pnl": None,
            "worst_pnl": None,
            "max_drawdown": None,
            "strategy_counts": {},
            "top_underlyings_by_pnl": {},
            "worst_underlyings_by_pnl": {},
            "strategy_pnl": {},
            "vol_regime_counts": {},
            "allocator_regime_counts": {},
            "ranker_strategy_counts": {},
        }

    ordered = df.sort_values("entry_timestamp")
    pnl = pd.to_numeric(ordered.get("expected_pnl"), errors="coerce").dropna().to_numpy(dtype=float)
    entry_dates = pd.to_datetime(ordered["entry_timestamp"], errors="coerce").dt.date.nunique()
    strategy_group = ordered.groupby("strategy", dropna=False) if "strategy" in ordered else None
    vol_regime_column = _first_present_column(ordered, ["market_volatility_regime", "vix_regime"])

    return {
        "rows": int(len(ordered)),
        "entry_dates": int(entry_dates),
        "avg_trades_per_entry_date": _safe_divide(len(ordered), entry_dates),
        "total_pnl": _sum_or_none(pnl),
        "mean_pnl": _mean(pnl),
        "median_pnl": _percentile(pnl, 50),
        "profit_factor": _profit_factor(pnl),
        "win_rate": _rate_series(ordered.get("expected_pnl"), lambda s: s > 0),
        "mean_return_on_risk": _series_mean(ordered.get("return_on_risk")),
        "large_loss_rate": _series_mean(ordered.get("large_loss_label")),
        "stop_loss_rate": _series_mean(ordered.get("stop_loss_hit")),
        "p05_pnl": _percentile(pnl, 5),
        "worst_pnl": _min_or_none(pnl),
        "max_drawdown": _max_drawdown(pnl) if len(pnl) else None,
        "strategy_counts": _value_counts(ordered.get("strategy")),
        "top_underlyings_by_pnl": _grouped_pnl(ordered, "underlying", ascending=False, limit=5),
        "worst_underlyings_by_pnl": _grouped_pnl(ordered, "underlying", ascending=True, limit=5),
        "strategy_pnl": _strategy_pnl(strategy_group),
        "vol_regime_counts": _value_counts(ordered.get(vol_regime_column) if vol_regime_column else None),
        "allocator_regime_counts": _value_counts(ordered.get("allocator_regime_label")),
        "ranker_strategy_counts": _value_counts(ordered.get("ranker_strategy")),
    }


def _strategy_pnl(grouped: pd.core.groupby.DataFrameGroupBy | None) -> dict[str, Any]:
    if grouped is None:
        return {}
    out: dict[str, Any] = {}
    for key, frame in grouped:
        pnl = pd.to_numeric(frame.get("expected_pnl"), errors="coerce").dropna().to_numpy(dtype=float)
        out[str(key)] = {
            "rows": int(len(frame)),
            "total_pnl": _sum_or_none(pnl),
            "mean_pnl": _mean(pnl),
            "win_rate": _rate_series(frame.get("expected_pnl"), lambda s: s > 0),
        }
    return out


def _grouped_pnl(df: pd.DataFrame, column: str, *, ascending: bool, limit: int) -> dict[str, float]:
    if column not in df:
        return {}
    grouped = (
        df.assign(_pnl=pd.to_numeric(df.get("expected_pnl"), errors="coerce"))
        .dropna(subset=["_pnl"])
        .groupby(column, dropna=False)["_pnl"]
        .sum()
        .sort_values(ascending=ascending)
        .head(limit)
    )
    return {str(index): round(float(value), 6) for index, value in grouped.items()}


def _series_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _rate_series(series: pd.Series | None, predicate) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(predicate(numeric).mean()), 6)


def _value_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.astype("string").value_counts(dropna=False)
    return {str(index): int(value) for index, value in counts.items()}


def _gate_stage_counts(df: pd.DataFrame) -> dict[str, int]:
    if "gate_stage" not in df:
        return {}
    stage = df["gate_stage"].dropna()
    if stage.empty:
        return {}
    return {str(index): int(value) for index, value in stage.astype("string").value_counts().items()}


def _portfolio_gamma_violation_counts(df: pd.DataFrame) -> dict[str, int]:
    if "gate_violation_codes" not in df:
        return {}
    counts: dict[str, int] = {}
    for raw_codes in df["gate_violation_codes"].dropna().astype("string"):
        for code in str(raw_codes).split(","):
            if not code:
                continue
            counts[code] = counts.get(code, 0) + 1
    return counts


def _quantity_reduction_counts(df: pd.DataFrame) -> dict[str, int]:
    directional = 0
    gamma = 0
    directional_series = pd.Series(False, index=df.index, dtype=bool)
    gamma_series = pd.Series(False, index=df.index, dtype=bool)
    if "directional_reduced" in df:
        directional_series = pd.Series(df["directional_reduced"], index=df.index).fillna(False).astype(bool)
        directional = int(directional_series.sum())
    if "portfolio_gamma_reduced" in df:
        gamma_series = pd.Series(df["portfolio_gamma_reduced"], index=df.index).fillna(False).astype(bool)
        gamma = int(gamma_series.sum())
    return {
        "directional_exposure": directional,
        "portfolio_gamma": gamma,
        "any": int((directional_series | gamma_series).sum()),
    }


def _rate_from_counts(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _sum_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.sum(values)), 6)


def _mean(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.mean(values)), 6)


def _percentile(values: np.ndarray, q: float) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.percentile(values, q)), 6)


def _min_or_none(values: np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    return round(float(np.min(values)), 6)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(float(a - b), 6)


def _first_present_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _allocator_regime_labels(df: pd.DataFrame) -> pd.Series:
    trend = df.get("market_trend_regime", pd.Series(index=df.index, dtype=object)).astype("string").str.lower()
    vol = df.get("market_volatility_regime", pd.Series(index=df.index, dtype=object)).astype("string").str.lower()
    vix = pd.to_numeric(df.get("vix_regime", pd.Series(index=df.index, dtype=float)), errors="coerce")

    labels = pd.Series("GREEN", index=df.index, dtype="string")
    yellow = (
        trend.isin(["downtrend", "sideways"])
        | vol.isin(["high", "normal"])
        | (vix >= 1.0)
    )
    orange = (
        ((trend == "downtrend") & (vol.isin(["high", "normal"])))
        | (vix >= 2.0)
    )
    labels.loc[yellow.fillna(False)] = "YELLOW"
    labels.loc[orange.fillna(False)] = "ORANGE"
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
