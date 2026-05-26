"""Evaluate a ranker after penalizing large-loss and stop-loss risk."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.evaluate_exit_criteria import ExitCriteriaConfig, _selection_metrics, _score_holdout
from ml.models.train_large_loss_classifier import _predict_prob, _transform_frame
from ml.models.train_xgboost import _engineer_features
from src.pick_selection import select_top_picks_with_scanner_controls
from src.portfolio_risk import PortfolioRiskService

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class RiskAdjustedConfig:
    selection_fraction: float = 0.10
    large_loss_penalty_multiple: float = 1.0
    stop_loss_penalty_multiple: float = 0.50
    max_large_loss_probability: float | None = None
    max_stop_loss_probability: float | None = None
    portfolio_risk_controls: bool = False
    portfolio_account_capital: float = 50_000.0
    portfolio_max_candidates_per_entry_date: int = 50
    scanner_controls: bool = True
    scanner_config_path: str | None = "config.json"
    scanner_top_n_per_entry_date: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate risk-adjusted ranking from ranker and risk classifiers.")
    parser.add_argument("--input", required=True, help="Candidate dataset directory, parquet, or JSONL.")
    parser.add_argument("--ranker-artifact", required=True, help="Expected-P&L XGBoost artifact JSON.")
    parser.add_argument("--large-loss-artifact", default=None, help="Binary classifier artifact for large_loss_label.")
    parser.add_argument("--stop-loss-artifact", default=None, help="Binary classifier artifact for stop_loss_hit.")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--selection-fraction", type=float, default=RiskAdjustedConfig.selection_fraction)
    parser.add_argument("--large-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.large_loss_penalty_multiple)
    parser.add_argument("--stop-loss-penalty-multiple", type=float, default=RiskAdjustedConfig.stop_loss_penalty_multiple)
    parser.add_argument("--max-large-loss-probability", type=float, default=None)
    parser.add_argument("--max-stop-loss-probability", type=float, default=None)
    parser.add_argument(
        "--portfolio-risk-controls",
        action="store_true",
        help="Apply the existing PortfolioRiskService default portfolio-level controls to selected spreads.",
    )
    parser.add_argument("--portfolio-account-capital", type=float, default=RiskAdjustedConfig.portfolio_account_capital)
    parser.add_argument(
        "--portfolio-max-candidates-per-entry-date",
        type=int,
        default=RiskAdjustedConfig.portfolio_max_candidates_per_entry_date,
    )
    parser.add_argument("--scanner-config-path", default=RiskAdjustedConfig.scanner_config_path)
    parser.add_argument("--scanner-top-n-per-entry-date", type=int, default=None)
    parser.add_argument("--no-scanner-controls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate_risk_adjusted_ranking(
        Path(args.input),
        Path(args.ranker_artifact),
        large_loss_artifact=Path(args.large_loss_artifact) if args.large_loss_artifact else None,
        stop_loss_artifact=Path(args.stop_loss_artifact) if args.stop_loss_artifact else None,
        config=RiskAdjustedConfig(
            selection_fraction=args.selection_fraction,
            large_loss_penalty_multiple=args.large_loss_penalty_multiple,
            stop_loss_penalty_multiple=args.stop_loss_penalty_multiple,
            max_large_loss_probability=args.max_large_loss_probability,
            max_stop_loss_probability=args.max_stop_loss_probability,
            portfolio_risk_controls=args.portfolio_risk_controls,
            portfolio_account_capital=args.portfolio_account_capital,
            portfolio_max_candidates_per_entry_date=args.portfolio_max_candidates_per_entry_date,
            scanner_controls=not args.no_scanner_controls,
            scanner_config_path=args.scanner_config_path,
            scanner_top_n_per_entry_date=args.scanner_top_n_per_entry_date,
        ),
    )
    payload = _jsonable(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def evaluate_risk_adjusted_ranking(
    dataset_path: Path,
    ranker_artifact_path: Path,
    *,
    large_loss_artifact: Path | None = None,
    stop_loss_artifact: Path | None = None,
    config: RiskAdjustedConfig | None = None,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required to evaluate risk-adjusted ranking.")
    cfg = config or RiskAdjustedConfig()
    df = load_dataset(dataset_path)
    ranker_artifact = json.loads(ranker_artifact_path.read_text(encoding="utf-8"))
    scored = _score_holdout(df, ranker_artifact)
    if large_loss_artifact:
        scored["large_loss_probability"] = _score_classifier(scored, large_loss_artifact)
    else:
        scored["large_loss_probability"] = 0.0
    if stop_loss_artifact:
        scored["stop_loss_probability"] = _score_classifier(scored, stop_loss_artifact)
    else:
        scored["stop_loss_probability"] = 0.0

    scored["risk_adjusted_score"] = risk_adjusted_score(
        scored["prediction"],
        scored.get("max_loss", 0.0),
        scored["large_loss_probability"],
        scored["stop_loss_probability"],
        large_loss_penalty_multiple=cfg.large_loss_penalty_multiple,
        stop_loss_penalty_multiple=cfg.stop_loss_penalty_multiple,
    )
    scored["risk_adjusted_score"] = apply_probability_caps(
        scored["risk_adjusted_score"],
        scored["large_loss_probability"],
        scored["stop_loss_probability"],
        max_large_loss_probability=cfg.max_large_loss_probability,
        max_stop_loss_probability=cfg.max_stop_loss_probability,
    )
    gate_cfg = ExitCriteriaConfig(selection_fraction=cfg.selection_fraction)
    raw = _selection_metrics(scored, "prediction", gate_cfg)
    adjusted = _selection_metrics(scored, "risk_adjusted_score", gate_cfg)
    portfolio_selection = None
    portfolio_deltas = None
    portfolio_eligible_rows = None
    scanner_config = _load_scanner_config(cfg.scanner_config_path)
    if cfg.portfolio_risk_controls:
        scored["portfolio_risk_score"] = apply_portfolio_risk_controls(
            scored,
            "risk_adjusted_score",
            account_capital=cfg.portfolio_account_capital,
            max_candidates_per_entry_date=cfg.portfolio_max_candidates_per_entry_date,
            scanner_controls=cfg.scanner_controls,
            scanner_config=scanner_config,
            scanner_top_n_per_entry_date=cfg.scanner_top_n_per_entry_date,
        )
        portfolio_eligible = scored[np.isfinite(pd.to_numeric(scored["portfolio_risk_score"], errors="coerce"))]
        portfolio_eligible_rows = int(len(portfolio_eligible))
        portfolio_selection = (
            _selection_metrics(portfolio_eligible, "portfolio_risk_score", ExitCriteriaConfig(selection_fraction=1.0))
            if portfolio_eligible_rows
            else {"rows": int(len(scored)), "selected_rows": 0}
        )
        portfolio_deltas = _metric_deltas(raw, portfolio_selection)
    return {
        "dataset_path": str(dataset_path),
        "ranker_artifact": str(ranker_artifact_path),
        "large_loss_artifact": str(large_loss_artifact) if large_loss_artifact else None,
        "stop_loss_artifact": str(stop_loss_artifact) if stop_loss_artifact else None,
        "config": asdict(cfg),
        "holdout_rows": int(len(scored)),
        "risk_eligible_rows": int(np.isfinite(pd.to_numeric(scored["risk_adjusted_score"], errors="coerce")).sum()),
        "raw_selection": raw,
        "risk_adjusted_selection": adjusted,
        "portfolio_risk_selection": portfolio_selection,
        "deltas": _metric_deltas(raw, adjusted),
        "portfolio_risk_deltas": portfolio_deltas,
        "risk_probability_summary": {
            "large_loss_probability": _prob_summary(scored["large_loss_probability"]),
            "stop_loss_probability": _prob_summary(scored["stop_loss_probability"]),
        },
        "portfolio_risk_eligible_rows": portfolio_eligible_rows,
    }


def risk_adjusted_score(
    prediction: Any,
    max_loss: Any,
    large_loss_probability: Any,
    stop_loss_probability: Any,
    *,
    large_loss_penalty_multiple: float = 1.0,
    stop_loss_penalty_multiple: float = 0.50,
) -> pd.Series:
    pred = pd.to_numeric(pd.Series(prediction), errors="coerce").fillna(-np.inf)
    loss = pd.to_numeric(pd.Series(max_loss), errors="coerce").fillna(0.0).clip(lower=0.0)
    large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    stop = pd.to_numeric(pd.Series(stop_loss_probability), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    penalty = loss * (large_loss_penalty_multiple * large + stop_loss_penalty_multiple * stop)
    return pred - penalty


def apply_probability_caps(
    score: Any,
    large_loss_probability: Any,
    stop_loss_probability: Any,
    *,
    max_large_loss_probability: float | None = None,
    max_stop_loss_probability: float | None = None,
) -> pd.Series:
    capped = pd.to_numeric(pd.Series(score), errors="coerce").fillna(-np.inf).copy()
    if max_large_loss_probability is not None:
        large = pd.to_numeric(pd.Series(large_loss_probability), errors="coerce").fillna(1.0)
        capped.loc[large > max_large_loss_probability] = -np.inf
    if max_stop_loss_probability is not None:
        stop = pd.to_numeric(pd.Series(stop_loss_probability), errors="coerce").fillna(1.0)
        capped.loc[stop > max_stop_loss_probability] = -np.inf
    return capped


def apply_portfolio_risk_controls(
    df: pd.DataFrame,
    score_column: str,
    *,
    account_capital: float = 50_000.0,
    max_candidates_per_entry_date: int | None = 50,
    scanner_controls: bool = True,
    scanner_config: dict[str, Any] | None = None,
    scanner_top_n_per_entry_date: int | None = None,
) -> pd.Series:
    """Keep only rows accepted by the existing live portfolio risk service.

    Historical rows are evaluated one entry date at a time. Expiry is normalized
    to today + row.dte because PortfolioRiskService is intentionally live-time
    code and computes DTE from calendar dates.
    """
    out = pd.Series(-np.inf, index=df.index, dtype=float)
    if df.empty:
        return out
    svc = PortfolioRiskService({"risk_parameters": {"portfolio_gamma_risk": {}}})
    working = df.copy()
    working["_portfolio_entry_date"] = _entry_dates(working)
    scores = pd.to_numeric(working[score_column], errors="coerce")
    for _, group in working[np.isfinite(scores)].groupby("_portfolio_entry_date", sort=True):
        ranked = group.assign(_portfolio_score=scores.loc[group.index]).sort_values(
            "_portfolio_score", ascending=False
        )
        candidate_limit = (
            scanner_top_n_per_entry_date
            or _scanner_top_n(scanner_config)
            or max_candidates_per_entry_date
        )
        if scanner_controls:
            picks = [_pick_from_scored_row(index, row) for index, row in ranked.iterrows()]
            picks = [pick for pick in picks if pick is not None]
            picks = select_top_picks_with_scanner_controls(
                picks,
                n=int(candidate_limit or len(picks)),
                config=scanner_config or {},
            )
        else:
            if max_candidates_per_entry_date is not None and max_candidates_per_entry_date > 0:
                ranked = ranked.head(max_candidates_per_entry_date)
            picks = [_pick_from_scored_row(index, row) for index, row in ranked.iterrows()]
            picks = [pick for pick in picks if pick is not None]
        accepted = svc.filter_picks(picks, [], account_capital=account_capital)
        for pick in accepted:
            out.loc[pick["_row_index"]] = float(pick["_portfolio_score"])
    return out


def _load_scanner_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _scanner_top_n(config: dict[str, Any] | None) -> int | None:
    cfg = config or {}
    value = (
        cfg.get("ml_scanner", {}).get("top_n")
        if isinstance(cfg.get("ml_scanner"), dict)
        else None
    )
    if value is None:
        value = cfg.get("top_n_picks", cfg.get("top_n_per_strategy"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _entry_dates(df: pd.DataFrame) -> pd.Series:
    if "entry_timestamp" in df:
        return pd.to_datetime(df["entry_timestamp"], errors="coerce").dt.date.astype(str)
    if "entry_date" in df:
        return df["entry_date"].astype(str)
    return pd.Series(["all"] * len(df), index=df.index)


def _pick_from_scored_row(index: Any, row: pd.Series) -> dict[str, Any] | None:
    strategy = str(row.get("strategy") or "").upper()
    if strategy not in {"PCS", "CCS"}:
        return None
    dte = _positive_int(row.get("dte"))
    spot = _positive_float(row.get("underlying_close"))
    short_strike = _positive_float(row.get("short_strike") or row.get("strike"))
    long_strike = _positive_float(row.get("long_strike"))
    if dte is None or spot is None or short_strike is None or long_strike is None:
        return None
    short_iv = _positive_float(row.get("implied_volatility")) or 0.25
    iv_skew = _float_value(row.get("iv_skew_wing")) or 0.0
    long_iv = max(0.01, short_iv + iv_skew)
    score = _float_value(row.get("_portfolio_score"))
    if score is None:
        return None
    pick = {
        "_row_index": index,
        "_portfolio_score": score,
        "strategy": strategy,
        "symbol": str(row.get("underlying") or row.get("symbol") or "?"),
        "expiry": (date.today() + timedelta(days=dte)).isoformat(),
        "current_price": spot,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "premium": _positive_float(row.get("entry_credit")) or _positive_float(row.get("option_entry_price")) or 0.01,
        "quantity": 1,
        "score": score,
        "short_iv": short_iv,
        "long_iv": long_iv,
    }
    if strategy == "PCS":
        pick["short_put"] = short_strike
        pick["long_put"] = long_strike
    else:
        pick["short_call"] = short_strike
        pick["long_call"] = long_strike
    return pick


def _score_classifier(df: pd.DataFrame, artifact_path: Path) -> np.ndarray:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    engineered = _engineer_features(df.copy())
    feature_columns = list(artifact["feature_columns"])
    frame = _transform_frame(engineered, feature_columns, dict(artifact.get("fill_values") or {}))
    booster = xgb.Booster()
    booster.load_model(str(artifact["model_path"]))
    return _predict_prob(booster, frame)


def _metric_deltas(raw: dict[str, Any], adjusted: dict[str, Any]) -> dict[str, Any]:
    metrics = (
        "mean_pnl",
        "slippage_adjusted_mean_pnl",
        "profit_factor",
        "slippage_adjusted_profit_factor",
        "win_rate",
        "large_loss_rate",
        "stop_loss_rate",
        "p05_pnl",
        "worst_pnl",
        "max_drawdown",
        "single_underlying_share",
    )
    out: dict[str, Any] = {}
    for metric in metrics:
        raw_value = _float_or_none(raw.get(metric))
        adjusted_value = _float_or_none(adjusted.get(metric))
        out[metric] = None if raw_value is None or adjusted_value is None else round(adjusted_value - raw_value, 6)
    return out


def _prob_summary(values: pd.Series | np.ndarray) -> dict[str, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "mean": round(float(numeric.mean()), 6),
        "p50": round(float(numeric.quantile(0.50)), 6),
        "p90": round(float(numeric.quantile(0.90)), 6),
        "p95": round(float(numeric.quantile(0.95)), 6),
        "p99": round(float(numeric.quantile(0.99)), 6),
        "max": round(float(numeric.max()), 6),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _float_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _positive_float(value: Any) -> float | None:
    numeric = _float_value(value)
    return numeric if numeric is not None and numeric > 0 else None


def _positive_int(value: Any) -> int | None:
    numeric = _positive_float(value)
    return max(1, int(round(numeric))) if numeric is not None else None


def _jsonable(report: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(report, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
