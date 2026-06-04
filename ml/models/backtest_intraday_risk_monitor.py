"""Backtest intraday risk-monitor models over trade paths."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.train_intraday_risk_monitor import _engineer_intraday_features

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class _ModelArtifact:
    path: Path
    artifact: dict[str, Any]
    booster: Any
    feature_columns: list[str]
    fill_values: dict[str, float]
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest intraday risk-monitor models over trade paths.")
    parser.add_argument("--input", required=True, help="intraday_risk_rows dataset directory/file")
    parser.add_argument("--model", action="append", required=True, help="Path to model artifact JSON; repeatable")
    parser.add_argument("--year", action="append", type=int, required=True, help="Calendar year to analyze; repeatable")
    parser.add_argument("--json-output", default=None, help="Optional JSON report output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_backtest(
        dataset_path=Path(args.input),
        model_paths=[Path(value) for value in args.model],
        years=args.year,
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
    model_paths: list[Path],
    years: list[int],
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required for intraday risk backtesting.")
    df = load_dataset(dataset_path)
    working = _prepare_frame(df)
    artifacts = [_load_artifact(path) for path in model_paths]

    years_sorted = sorted({int(year) for year in years})
    year_reports = {
        str(year): {
            artifact.path.name: _backtest_year(working, artifact, year)
            for artifact in artifacts
        }
        for year in years_sorted
    }
    return {
        "dataset_path": str(dataset_path),
        "years": years_sorted,
        "models": [str(artifact.path) for artifact in artifacts],
        "reports": year_reports,
    }


def _load_artifact(path: Path) -> _ModelArtifact:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    booster = xgb.Booster()
    model_path = Path(str(artifact["model_path"]))
    booster.load_model(str(model_path))
    return _ModelArtifact(
        path=path,
        artifact=artifact,
        booster=booster,
        feature_columns=list(artifact["feature_columns"]),
        fill_values={str(key): float(value) for key, value in (artifact.get("fill_values") or {}).items()},
        threshold=float(artifact["recommended_close_threshold"]),
    )


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    working = _engineer_intraday_features(df.copy())
    working["entry_timestamp"] = pd.to_datetime(working["entry_timestamp"], utc=True, errors="coerce")
    working["state_timestamp"] = pd.to_datetime(working["state_timestamp"], utc=True, errors="coerce")
    working = working.dropna(subset=["entry_timestamp", "state_timestamp", "underlying", "short_option_symbol", "long_option_symbol"])
    working["_group_key"] = working.apply(_trade_group_key, axis=1)
    working.sort_values(["_group_key", "state_timestamp"], inplace=True)
    working.reset_index(drop=True, inplace=True)
    return working


def _trade_group_key(row: pd.Series) -> str:
    return "||".join(
        [
            str(row.get("underlying") or ""),
            pd.Timestamp(row["entry_timestamp"]).isoformat(),
            str(row.get("short_option_symbol") or ""),
            str(row.get("long_option_symbol") or ""),
        ]
    )


def _backtest_year(df: pd.DataFrame, artifact: _ModelArtifact, year: int) -> dict[str, Any]:
    year_df = df.loc[df["entry_timestamp"].dt.year == year].copy()
    if year_df.empty:
        return {"year": year, "trades": 0}

    scores = _score_rows(year_df, artifact)
    year_df["_score"] = scores
    trade_reports: list[dict[str, Any]] = []
    for _, group in year_df.groupby("_group_key", sort=False):
        trade_reports.append(_evaluate_trade(group.copy(), artifact.threshold))
    trades = pd.DataFrame(trade_reports)
    return _summarize_year(year, artifact, trades)


def _score_rows(df: pd.DataFrame, artifact: _ModelArtifact) -> np.ndarray:
    frame = (
        df[artifact.feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(artifact.fill_values)
    )
    matrix = xgb.DMatrix(frame, feature_names=list(frame.columns))
    return artifact.booster.predict(matrix)


def _evaluate_trade(group: pd.DataFrame, threshold: float) -> dict[str, Any]:
    ordered = group.sort_values("state_timestamp").reset_index(drop=True)
    trigger_mask = pd.to_numeric(ordered["_score"], errors="coerce").fillna(-np.inf) >= threshold
    triggered = bool(trigger_mask.any())
    trigger_row = ordered.loc[trigger_mask.idxmax()] if triggered else None
    final_row = ordered.iloc[-1]
    trigger_pnl = float(trigger_row["pnl_per_contract"]) if triggered else float(final_row["pnl_per_contract"])
    final_pnl = float(final_row["pnl_per_contract"])
    pnl_delta = trigger_pnl - final_pnl
    actual_exit_reason = str(final_row.get("intraday_exit_reason") or "")
    return {
        "entry_year": int(pd.Timestamp(final_row["entry_timestamp"]).year),
        "underlying": str(final_row.get("underlying") or ""),
        "strategy": str(final_row.get("strategy") or ""),
        "actual_exit_reason": actual_exit_reason,
        "triggered": triggered,
        "trigger_score": float(trigger_row["_score"]) if triggered else None,
        "trigger_minutes_to_exit": float(trigger_row["minutes_to_exit"]) if triggered else 0.0,
        "trigger_minutes_since_entry": float(trigger_row["minutes_since_entry"]) if triggered else float(final_row["minutes_since_entry"]),
        "trigger_pnl_per_contract": trigger_pnl,
        "final_pnl_per_contract": final_pnl,
        "pnl_delta_per_contract": pnl_delta,
        "trigger_stop_loss_hit_30m": int(trigger_row["stop_loss_hit_30m"]) if triggered else 0,
        "trigger_profit_take_hit_30m": int(trigger_row["profit_take_hit_30m"]) if triggered else 0,
        "max_adverse_pnl_per_contract": float(pd.to_numeric(ordered["pnl_per_contract"], errors="coerce").min()),
        "max_favorable_pnl_per_contract": float(pd.to_numeric(ordered["pnl_per_contract"], errors="coerce").max()),
    }


def _summarize_year(year: int, artifact: _ModelArtifact, trades: pd.DataFrame) -> dict[str, Any]:
    triggered = trades.loc[trades["triggered"]].copy()
    actual_stop = trades.loc[trades["actual_exit_reason"] == "stop_loss"].copy()
    actual_profit = trades.loc[trades["actual_exit_reason"] == "profit_take"].copy()
    actual_horizon = trades.loc[trades["actual_exit_reason"] == "horizon"].copy()

    return {
        "year": year,
        "model_artifact": str(artifact.path),
        "threshold": artifact.threshold,
        "trades": int(len(trades)),
        "triggered_trades": int(len(triggered)),
        "trigger_rate": _mean_bool(trades["triggered"]),
        "actual_stop_loss_trades": int(len(actual_stop)),
        "actual_profit_take_trades": int(len(actual_profit)),
        "actual_horizon_trades": int(len(actual_horizon)),
        "stop_loss_recall_trade": _safe_divide(
            int(((trades["triggered"]) & (trades["actual_exit_reason"] == "stop_loss")).sum()),
            int((trades["actual_exit_reason"] == "stop_loss").sum()),
        ),
        "false_close_rate_trade": _safe_divide(
            int(((trades["triggered"]) & (trades["actual_exit_reason"] != "stop_loss")).sum()),
            int(len(trades)),
        ),
        "precision_on_triggered_trades": _safe_divide(
            int(((trades["triggered"]) & (trades["actual_exit_reason"] == "stop_loss")).sum()),
            int(trades["triggered"].sum()),
        ),
        "avg_trigger_minutes_to_exit": _series_mean(triggered.get("trigger_minutes_to_exit")),
        "avg_trigger_pnl_per_contract": _series_mean(triggered.get("trigger_pnl_per_contract")),
        "avg_final_pnl_per_contract": _series_mean(trades.get("final_pnl_per_contract")),
        "avg_pnl_delta_per_contract": _series_mean(trades.get("pnl_delta_per_contract")),
        "median_pnl_delta_per_contract": _series_percentile(trades.get("pnl_delta_per_contract"), 50),
        "positive_pnl_delta_rate": _rate_series(trades.get("pnl_delta_per_contract"), lambda s: s > 0),
        "worse_pnl_delta_rate": _rate_series(trades.get("pnl_delta_per_contract"), lambda s: s < 0),
        "avg_pnl_delta_stop_loss_trades": _series_mean(actual_stop.get("pnl_delta_per_contract")),
        "avg_pnl_delta_profit_take_trades": _series_mean(actual_profit.get("pnl_delta_per_contract")),
        "avg_pnl_delta_horizon_trades": _series_mean(actual_horizon.get("pnl_delta_per_contract")),
        "trigger_reason_mix": _value_counts(triggered.get("actual_exit_reason")),
        "strategy_trigger_rates": _group_rate(trades, "strategy", "triggered"),
        "underlying_pnl_delta_top": _grouped_mean(trades, "underlying", "pnl_delta_per_contract", ascending=False, limit=8),
        "underlying_pnl_delta_bottom": _grouped_mean(trades, "underlying", "pnl_delta_per_contract", ascending=True, limit=8),
    }


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _mean_bool(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    return round(float(series.astype(bool).mean()), 6)


def _series_mean(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(numeric.mean()), 6)


def _series_percentile(series: pd.Series | None, q: float) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return round(float(np.percentile(numeric, q)), 6)


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


def _group_rate(df: pd.DataFrame, group_col: str, bool_col: str) -> dict[str, float]:
    if group_col not in df or bool_col not in df:
        return {}
    out: dict[str, float] = {}
    for key, frame in df.groupby(group_col, dropna=False):
        out[str(key)] = round(float(frame[bool_col].astype(bool).mean()), 6)
    return dict(sorted(out.items(), key=lambda item: item[1], reverse=True))


def _grouped_mean(df: pd.DataFrame, group_col: str, value_col: str, *, ascending: bool, limit: int) -> dict[str, float]:
    if group_col not in df or value_col not in df:
        return {}
    grouped = (
        df.assign(_value=pd.to_numeric(df[value_col], errors="coerce"))
        .dropna(subset=["_value"])
        .groupby(group_col, dropna=False)["_value"]
        .mean()
        .sort_values(ascending=ascending)
        .head(limit)
    )
    return {str(index): round(float(value), 6) for index, value in grouped.items()}


if __name__ == "__main__":
    raise SystemExit(main())
