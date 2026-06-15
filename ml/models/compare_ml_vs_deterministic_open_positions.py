"""Ledger-style open-position backtest for ML vs deterministic agents."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.backtest_agent_like import _selection_stats
from ml.models.compare_ml_vs_deterministic import (
    _allocator_regime_labels,
    _delta,
    _deterministic_scores,
    _deterministic_top_n,
    _fmt,
    _positive_float,
    _primary_mask,
    _regime_profile,
    _resolve_optional_artifact_path,
    _resolve_ranker_artifact,
    _resolve_strategy_ranker_paths,
    _score_rankers,
)
from ml.models.evaluate_risk_adjusted_ranking import (
    _score_classifier,
    apply_large_loss_gate,
)
from ml.models.train_baseline import _max_drawdown, _profit_factor
from src.agent_risk import apply_ml_position_sizing, apply_ml_quantity_overlays
from src.greeks import bs_greeks
from src.pick_identity import (
    pick_contract_signature,
    position_contract_signature,
)
from src.pick_selection import select_top_picks_with_scanner_controls
from src.portfolio_risk import GreekExposure, PortfolioRiskService
from src.utils import load_config

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass(frozen=True)
class Snapshot:
    spot: float
    iv: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a true open-position backtest for ML vs deterministic agents."
    )
    parser.add_argument("--input", required=True, help="Dataset directory, parquet, or JSONL.")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--ml-config", default="config.json")
    parser.add_argument("--ml-registry", default="artifacts/model_registry.json")
    parser.add_argument("--det-config", default="../optionwheel/config.json")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_open_position_comparison(
        dataset_path=Path(args.input),
        start_year=args.start_year,
        end_year=args.end_year,
        ml_config_path=Path(args.ml_config),
        ml_registry_path=Path(args.ml_registry),
        deterministic_config_path=Path(args.det_config),
    )
    payload = json.loads(json.dumps(report, default=str))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_open_position_comparison(
    *,
    dataset_path: Path,
    start_year: int,
    end_year: int,
    ml_config_path: Path,
    ml_registry_path: Path,
    deterministic_config_path: Path,
) -> dict[str, Any]:
    if xgb is None:
        raise ImportError("xgboost is required for ML open-position backtesting.")

    full_df = load_dataset(dataset_path)
    full_df = _prepare_dataset(full_df)
    entries = full_df.loc[full_df["entry_year"].between(start_year, end_year, inclusive="both")].copy()
    if entries.empty:
        raise ValueError("No rows found in the requested entry-year window.")

    snapshots = _build_snapshot_map(full_df)
    ml = _run_ml_open_book(
        entries=entries,
        snapshots=snapshots,
        config_path=ml_config_path,
        registry_path=ml_registry_path,
    )
    deterministic = _run_deterministic_open_book(
        entries=entries,
        snapshots=snapshots,
        config_path=deterministic_config_path,
    )

    close_dates = []
    for frame in (ml["selected"], deterministic["selected"]):
        if "close_date" in frame.columns and not frame.empty:
            close_dates.extend([frame["close_date"].min(), frame["close_date"].max()])
    if close_dates:
        close_quarters = _quarter_labels_from_dates(min(close_dates), max(close_dates))
    else:
        close_quarters = []
    comparison = _build_open_book_comparison(
        ml["selected"],
        deterministic["selected"],
        close_quarters,
    )
    return {
        "dataset_path": str(dataset_path),
        "entry_window": {
            "start_year": start_year,
            "end_year": end_year,
        },
        "assumptions": {
            "true_open_position_rules": [
                "Capital stays reserved until the row's historical exit date.",
                "Exact duplicate contracts are blocked while a prior position is still open.",
                "Distinct ladders on the same symbol+strategy are allowed when they are not exact duplicate contracts.",
                "Realized PnL is booked on exit date and quantity-weighted.",
                "New entries request size from ML rank tiers before regime, side-budget, and cluster overlays trim them back.",
            ],
            "deterministic_proxies": {
                "prob_win_proxy": "1 - abs(option_delta)",
                "liquidity_filter": "Historical dataset does not store open interest, so the deterministic live OI gate is skipped.",
            },
        },
        "ml": _strip_selected_frame(ml),
        "deterministic": _strip_selected_frame(deterministic),
        "comparison": comparison,
    }


def _run_ml_open_book(
    *,
    entries: pd.DataFrame,
    snapshots: dict[tuple[str, date], Snapshot],
    config_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    repo_root = config_path.parent.resolve()
    config = load_config(str(config_path))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    ranker_artifact_path = _resolve_ranker_artifact(repo_root, registry)
    strategy_rankers = _resolve_strategy_ranker_paths(
        repo_root,
        config,
        default_ranker=ranker_artifact_path,
        pcs_ranker_artifact_path=None,
        ccs_ranker_artifact_path=None,
    )
    large_loss_artifact_path = _resolve_optional_artifact_path(
        repo_root,
        None,
        str(config.get("ml_scanner", {}).get("large_loss_classifier_path") or ""),
    )
    stop_loss_artifact_path = _resolve_optional_artifact_path(
        repo_root,
        None,
        str(config.get("ml_scanner", {}).get("stop_loss_classifier_path") or ""),
    )
    if large_loss_artifact_path is None:
        raise ValueError("Unable to resolve ML large-loss classifier artifact path.")

    min_dte = int(config.get("ml_scanner", {}).get("min_dte", 7) or 7)
    max_dte = int(config.get("ml_scanner", {}).get("max_dte", config.get("expiry_days_max", 45) or 45) or 45)
    working = entries.copy()
    dte = pd.to_numeric(working["dte"], errors="coerce")
    working = working.loc[dte.between(min_dte, max_dte, inclusive="both")].copy()
    working = _score_rankers(working, strategy_rankers)
    working["large_loss_probability"] = _score_classifier(working, large_loss_artifact_path)
    if stop_loss_artifact_path is not None and stop_loss_artifact_path.exists():
        working["stop_loss_probability"] = _score_classifier(working, stop_loss_artifact_path)
    else:
        working["stop_loss_probability"] = pd.Series(pd.NA, index=working.index, dtype="Float64")
    working["allocator_regime_label"] = _allocator_regime_labels(working)
    working["candidate_score"] = pd.to_numeric(working["prediction"], errors="coerce").fillna(-np.inf)
    working["gate_stage"] = pd.Series(pd.NA, index=working.index, dtype="string")

    ll_threshold = float(config.get("ml_scanner", {}).get("large_loss_veto_threshold", 0.60))
    sl_threshold = float(config.get("ml_scanner", {}).get("stop_loss_veto_threshold", 0.30))
    working["candidate_score"] = apply_large_loss_gate(
        working["candidate_score"],
        working["large_loss_probability"],
        max_large_loss_probability=ll_threshold,
    )
    ll_veto = ~np.isfinite(pd.to_numeric(working["candidate_score"], errors="coerce"))
    working.loc[ll_veto, "gate_stage"] = "large_loss_veto"
    working["candidate_score"] = apply_large_loss_gate(
        working["candidate_score"],
        working["stop_loss_probability"],
        max_large_loss_probability=sl_threshold,
    )
    sl_veto = (
        ~np.isfinite(pd.to_numeric(working["candidate_score"], errors="coerce"))
        & ~ll_veto
    )
    working.loc[sl_veto, "gate_stage"] = "stop_loss_veto"

    selected = _simulate_open_book(
        working,
        config=config,
        snapshots=snapshots,
        side_name="ml",
        candidate_mask=np.isfinite(pd.to_numeric(working["candidate_score"], errors="coerce")),
        top_n=int(config.get("ml_scanner", {}).get("top_n", 10) or 10),
        score_column="candidate_score",
        primary_gate_label="post_model_gate",
    )
    return {
        "config_path": str(config_path),
        "runtime": {
            "min_dte": min_dte,
            "max_dte": max_dte,
            "scanner_top_n": int(config.get("ml_scanner", {}).get("top_n", 10) or 10),
            "large_loss_veto_threshold": ll_threshold,
            "stop_loss_veto_threshold": sl_threshold,
            "max_capital_per_period": float(config.get("max_capital_per_period", 50_000.0)),
        },
        **selected,
    }


def _run_deterministic_open_book(
    *,
    entries: pd.DataFrame,
    snapshots: dict[tuple[str, date], Snapshot],
    config_path: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    working = entries.copy()
    dte = pd.to_numeric(working["dte"], errors="coerce")
    working = working.loc[dte.between(1, int(config.get("expiry_days_max", 14) or 14), inclusive="both")].copy()
    working["allocator_regime_label"] = _allocator_regime_labels(working)
    working["prob_win_proxy"] = 1.0 - pd.to_numeric(working.get("option_delta"), errors="coerce").abs()
    working["candidate_score"] = _deterministic_scores(working)
    working["gate_stage"] = pd.Series(pd.NA, index=working.index, dtype="string")
    primary_mask = _primary_mask(working, config)
    working.loc[~primary_mask, "gate_stage"] = "deterministic_primary_filter"

    selected = _simulate_open_book(
        working,
        config=config,
        snapshots=snapshots,
        side_name="deterministic",
        candidate_mask=primary_mask.fillna(False),
        top_n=_deterministic_top_n(config),
        score_column="candidate_score",
        primary_gate_label="post_primary_gate",
    )
    return {
        "config_path": str(config_path),
        "runtime": {
            "max_dte": int(config.get("expiry_days_max", 14) or 14),
            "scanner_top_n": _deterministic_top_n(config),
            "max_capital_per_period": float(config.get("max_capital_per_period", 50_000.0)),
        },
        **selected,
    }


def _simulate_open_book(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    snapshots: dict[tuple[str, date], Snapshot],
    side_name: str,
    candidate_mask: pd.Series,
    top_n: int,
    score_column: str,
    primary_gate_label: str,
) -> dict[str, Any]:
    working = df.copy()
    working["entry_date"] = pd.to_datetime(working["entry_timestamp"], errors="coerce", utc=True).dt.date
    working["candidate_score"] = pd.to_numeric(working[score_column], errors="coerce")
    account_capital = float(config.get("account_capital") or config.get("max_capital_per_period", 50_000.0))
    capital_budget = float(config.get("max_capital_per_period", 50_000.0))
    open_positions: list[dict[str, Any]] = []
    selected_positions: list[dict[str, Any]] = []

    for current_date, group in working.loc[candidate_mask].groupby("entry_date", sort=True):
        open_positions = [pos for pos in open_positions if pos["close_date"] >= current_date]
        group = group.sort_values("candidate_score", ascending=False).copy()
        regime_label = _group_regime_label(group)
        regime = _regime_profile(config, regime_label)

        picks = [_build_pick_from_row(row, score_column="candidate_score") for _, row in group.iterrows()]
        picks = [pick for pick in picks if pick is not None]
        if not picks:
            continue
        selected_today = select_top_picks_with_scanner_controls(
            picks,
            n=top_n,
            config=config,
            regime_label=regime.label,
        )
        selected_idx = {pick["_row_index"] for pick in selected_today}
        rejected_selection = set(group.index) - selected_idx
        if rejected_selection:
            working.loc[list(rejected_selection), "gate_stage"] = "pick_selection"

        open_contract_signatures = {
            position_contract_signature(pos) for pos in open_positions
        }
        deduped: list[dict[str, Any]] = []
        seen_contract_signatures = set(open_contract_signatures)
        for pick in selected_today:
            contract_signature = pick_contract_signature(pick)
            if contract_signature in seen_contract_signatures:
                working.loc[pick["_row_index"], "gate_stage"] = "dedup_open_position"
                continue
            deduped.append(pick)
            seen_contract_signatures.add(contract_signature)
        if not deduped:
            continue

        remaining_budget = capital_budget - _deployed_capital(open_positions)
        if remaining_budget <= 0:
            for pick in deduped:
                working.loc[pick["_row_index"], "gate_stage"] = "capital_budget"
            continue

        affordable_sorted = sorted(deduped, key=lambda item: item.get("score", 0.0), reverse=True)
        affordable = [pick for pick in affordable_sorted if _capital_for_pick(pick) <= remaining_budget]
        rejected_budget = {pick["_row_index"] for pick in affordable_sorted} - {pick["_row_index"] for pick in affordable}
        if rejected_budget:
            working.loc[list(rejected_budget), "gate_stage"] = "capital_budget"
        if not affordable:
            continue

        affordable = apply_ml_position_sizing(
            affordable,
            config,
            max_contracts=int(config.get("max_contracts_per_pick", 50) or 50),
        )
        directional, overlay_rejections = apply_ml_quantity_overlays(
            affordable,
            open_positions,
            config,
            account_capital=account_capital,
            available_capital=remaining_budget,
            regime=regime,
            max_contracts=int(config.get("max_contracts_per_pick", 50) or 50),
        )
        for rejected_pick in overlay_rejections:
            row_index = rejected_pick.get("_row_index")
            if row_index is None:
                continue
            stage = str(rejected_pick.get("filtered_stage") or "")
            stage_key = {
                "Capital budget": "capital_budget",
                "Directional exposure": "directional_exposure",
                "Correlated cluster": "correlated_cluster",
                "Regime quantity throttle": "regime_quantity",
            }.get(stage, "quantity_overlay")
            working.loc[row_index, "gate_stage"] = stage_key
        directional_idx = {pick["_row_index"] for pick in directional}
        if not directional:
            continue

        for pick in directional:
            working.loc[pick["_row_index"], "gate_stage"] = "selected"
            position = _position_from_pick_row(
                pick,
                row=working.loc[pick["_row_index"]],
                current_date=current_date,
                side_name=side_name,
            )
            open_positions.append(position)
            selected_positions.append(position)

    selected_df = pd.DataFrame(selected_positions)
    if selected_df.empty:
        selected_df = pd.DataFrame(
            columns=[
                "symbol",
                "strategy",
                "quantity",
                "entry_timestamp",
                "exit_timestamp",
                "entry_date",
                "close_date",
                "entry_quarter",
                "close_quarter",
                "weighted_expected_pnl",
                "expected_pnl",
            ]
        )

    reports = _build_open_book_reports(
        selected_df,
        working,
        candidate_mask,
        primary_gate_label=primary_gate_label,
    )
    return {
        "reports": reports,
        "gate_stage_counts": _value_counts(working.get("gate_stage")),
        "selected": selected_df,
    }


def _build_pick_from_row(row: pd.Series, *, score_column: str) -> dict[str, Any] | None:
    strategy = str(row.get("strategy") or "").upper()
    if strategy not in {"PCS", "CCS"}:
        return None
    short_strike = _positive_float(row.get("short_strike") or row.get("strike"))
    long_strike = _positive_float(row.get("long_strike"))
    spot = _positive_float(row.get("underlying_close"))
    dte = int(float(row.get("dte") or 0))
    premium = _positive_float(row.get("entry_credit"))
    score = float(row.get(score_column) or 0.0)
    expiration = row.get("expiration")
    if None in {short_strike, long_strike, spot, premium} or dte <= 0:
        return None
    expiry = _expiry_iso(expiration, pd.to_datetime(row.get("entry_timestamp"), utc=True), dte)
    short_iv = _positive_float(row.get("implied_volatility")) or 0.25
    iv_skew = float(row.get("iv_skew_wing") or 0.0)
    pick: dict[str, Any] = {
        "_row_index": row.name,
        "strategy": strategy,
        "symbol": str(row.get("underlying") or row.get("symbol") or "?"),
        "expiry": expiry,
        "current_price": spot,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "premium": premium,
        "quantity": 1,
        "score": score,
        "short_iv": short_iv,
        "long_iv": max(0.01, short_iv + iv_skew),
    }
    if strategy == "PCS":
        pick["short_put"] = short_strike
        pick["long_put"] = long_strike
    else:
        pick["short_call"] = short_strike
        pick["long_call"] = long_strike
    return pick


def _apply_directional_caps_with_open_positions(
    picks: list[dict[str, Any]],
    *,
    open_positions: list[dict[str, Any]],
    account_capital: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cfg = config.get("risk_parameters", {}).get("directional_exposure_caps", {})
    if not cfg.get("enabled", True):
        return picks
    min_side_cap = float(cfg.get("min_side_cap_dollars", 0.0) or 0.0)
    limits = {
        "put": max(
            float(cfg.get("put", cfg.get("max_put_pct", 0.04))) * account_capital,
            float(cfg.get("min_put_cap_dollars", min_side_cap) or 0.0),
        ),
        "call": max(
            float(cfg.get("call", cfg.get("max_call_pct", 0.04))) * account_capital,
            float(cfg.get("min_call_cap_dollars", min_side_cap) or 0.0),
        ),
    }
    used = {"put": 0.0, "call": 0.0}
    for position in open_positions:
        sides = _strategy_sides(position["strategy"])
        if not sides:
            continue
        per_side_loss = float(position["max_loss_per_contract"]) * int(position["quantity"]) / len(sides)
        for side in sides:
            used[side] += per_side_loss

    kept: list[dict[str, Any]] = []
    for pick in sorted(picks, key=lambda item: item.get("score", 0.0), reverse=True):
        sides = _strategy_sides(str(pick.get("strategy") or ""))
        requested_qty = int(pick.get("quantity") or 1)
        per_contract_loss = _max_loss_per_contract_for_pick(pick)
        if per_contract_loss <= 0 or not sides:
            continue
        per_side_loss = per_contract_loss / len(sides)
        qty_cap = requested_qty
        for side in sides:
            remaining = limits[side] - used[side]
            qty_cap = min(qty_cap, int(remaining // per_side_loss))
        if qty_cap <= 0:
            continue
        pick["quantity"] = qty_cap
        for side in sides:
            used[side] += per_side_loss * qty_cap
        kept.append(pick)
    return kept


def _apply_historical_portfolio_gamma_gate(
    picks: list[dict[str, Any]],
    *,
    open_positions: list[dict[str, Any]],
    current_date: date,
    snapshots: dict[tuple[str, date], Snapshot],
    account_capital: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    svc = PortfolioRiskService(config)
    if not svc.enabled():
        return picks
    base_exposures: list[GreekExposure] = []
    for position in open_positions:
        marked = _mark_position_for_date(position, current_date, snapshots)
        if marked is None:
            if svc._fail_closed():
                return []
            continue
        exposure = _pick_exposure_for_date(marked, int(position["quantity"]), current_date, svc)
        if exposure is None:
            if svc._fail_closed():
                return []
            continue
        base_exposures.append(exposure)

    accepted: list[dict[str, Any]] = []
    running = list(base_exposures)
    for pick in sorted(picks, key=lambda item: item.get("score", 0.0), reverse=True):
        requested_qty = max(1, int(pick.get("quantity") or 1))
        best_qty = 0
        best_summary: dict[str, Any] | None = None
        for qty in range(requested_qty, 0, -1):
            exposure = _pick_exposure_for_date(pick, qty, current_date, svc)
            if exposure is None:
                if svc._fail_closed():
                    best_qty = 0
                else:
                    best_qty = qty
                    best_summary = svc._summarize(running, account_capital)
                break
            summary = svc._summarize(running + [exposure], account_capital)
            if not summary["violations"]:
                best_qty = qty
                best_summary = summary
                break
        if best_qty <= 0:
            continue
        pick["quantity"] = best_qty
        accepted.append(pick)
        exposure = _pick_exposure_for_date(pick, best_qty, current_date, svc)
        if exposure is not None:
            running.append(exposure)
    return accepted


def _pick_exposure_for_date(
    pick: dict[str, Any],
    quantity: int,
    current_date: date,
    svc: PortfolioRiskService,
) -> GreekExposure | None:
    try:
        spot = float(pick.get("current_price") or 0.0)
    except (TypeError, ValueError):
        return None
    expiry = pick.get("expiry")
    if not expiry:
        return None
    try:
        dte = (date.fromisoformat(str(expiry)) - current_date).days
    except Exception:
        return None
    if spot <= 0 or dte <= 0:
        return None
    legs = svc._pick_greek_legs(pick)
    if not legs:
        return None
    net_delta = net_gamma = net_theta = net_vega = 0.0
    for leg in legs:
        greeks = bs_greeks(
            spot,
            float(leg["strike"]),
            float(leg["iv"]),
            dte,
            str(leg["option_type"]),
        )
        sign = -1.0 if leg["position"] == "short" else 1.0
        net_delta += sign * greeks["delta"]
        net_gamma += sign * greeks["gamma"]
        net_theta += sign * greeks["theta"]
        net_vega += sign * greeks["vega"]
    return GreekExposure(
        symbol=str(pick.get("symbol") or "?"),
        expiry=str(expiry),
        dte=dte,
        spot=spot,
        contracts=max(1, int(quantity)),
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_theta=net_theta,
        net_vega=net_vega,
        source="historical",
        ref=f"{pick.get('strategy')} {pick.get('symbol')}",
    )


def _mark_position_for_date(
    position: dict[str, Any],
    current_date: date,
    snapshots: dict[tuple[str, date], Snapshot],
) -> dict[str, Any] | None:
    snapshot = snapshots.get((str(position["symbol"]), current_date))
    if snapshot is None:
        return None
    marked = dict(position)
    marked["current_price"] = snapshot.spot
    short_iv = snapshot.iv if snapshot.iv > 0 else float(position["short_iv"])
    marked["short_iv"] = short_iv
    marked["long_iv"] = max(0.01, short_iv + float(position.get("iv_skew", 0.0)))
    return marked


def _position_from_pick_row(
    pick: dict[str, Any],
    *,
    row: pd.Series,
    current_date: date,
    side_name: str,
) -> dict[str, Any]:
    entry_ts = pd.to_datetime(row.get("entry_timestamp"), errors="coerce", utc=True)
    exit_ts = _exit_timestamp(row)
    quantity = int(pick.get("quantity") or 1)
    expected_pnl = float(row.get("expected_pnl") or 0.0)
    return {
        "side_name": side_name,
        "row_index": pick["_row_index"],
        "symbol": pick["symbol"],
        "strategy": pick["strategy"],
        "quantity": quantity,
        "entry_timestamp": entry_ts,
        "exit_timestamp": exit_ts,
        "entry_date": current_date,
        "close_date": exit_ts.date(),
        "entry_quarter": str(pd.Period(entry_ts, freq="Q")),
        "close_quarter": str(pd.Period(exit_ts, freq="Q")),
        "expected_pnl": expected_pnl,
        "weighted_expected_pnl": expected_pnl * quantity,
        "stop_loss_hit": float(row.get("stop_loss_hit") or 0.0),
        "large_loss_label": float(row.get("large_loss_label") or 0.0),
        "return_on_risk": float(row.get("return_on_risk") or 0.0),
        "max_loss_per_contract": _max_loss_per_contract_for_pick(pick),
        "capital_per_contract": _capital_for_pick(pick),
        "score": float(pick.get("score") or 0.0),
        "expiry": str(pick["expiry"]),
        "current_price": float(pick["current_price"]),
        "short_strike": float(pick["short_strike"]),
        "long_strike": float(pick["long_strike"]),
        "short_iv": float(pick["short_iv"]),
        "long_iv": float(pick["long_iv"]),
        "iv_skew": float(pick["long_iv"]) - float(pick["short_iv"]),
        "allocator_regime_label": str(row.get("allocator_regime_label") or "GREEN"),
        "market_volatility_regime": str(row.get("market_volatility_regime") or ""),
    }


def _build_open_book_reports(
    selected_df: pd.DataFrame,
    working: pd.DataFrame,
    candidate_mask: pd.Series,
    *,
    primary_gate_label: str,
) -> dict[str, Any]:
    candidate = working.loc[candidate_mask].copy()
    return {
        "overall": _summary_open_book(selected_df, working, candidate, primary_gate_label),
        "entry_quarters": {
            quarter: _ledger_stats(
                selected_df.loc[selected_df["entry_quarter"] == quarter].copy(),
                label=quarter,
                date_column="entry_timestamp",
            )
            for quarter in sorted(selected_df.get("entry_quarter", pd.Series(dtype="string")).dropna().unique())
        },
        "close_quarters": {
            quarter: _ledger_stats(
                selected_df.loc[selected_df["close_quarter"] == quarter].copy(),
                label=quarter,
                date_column="exit_timestamp",
            )
            for quarter in sorted(selected_df.get("close_quarter", pd.Series(dtype="string")).dropna().unique())
        },
    }


def _summary_open_book(
    selected_df: pd.DataFrame,
    working: pd.DataFrame,
    candidate: pd.DataFrame,
    primary_gate_label: str,
) -> dict[str, Any]:
    universe_stats = _selection_stats(working)
    candidate_stats = _selection_stats(candidate)
    selected_stats = _ledger_stats(selected_df, label="selected", date_column="exit_timestamp")
    return {
        "universe": universe_stats,
        primary_gate_label: candidate_stats,
        "selected": selected_stats,
        "candidate_rejection_rate": _rate_from_counts(universe_stats["rows"] - candidate_stats["rows"], universe_stats["rows"]),
        "selection_rate": _rate_from_counts(selected_stats["trades"], candidate_stats["rows"]),
        "baseline_mean_pnl_delta": _delta(selected_stats.get("mean_pnl_per_trade"), universe_stats.get("mean_pnl")),
        "baseline_profit_factor_delta": _delta(selected_stats.get("profit_factor"), universe_stats.get("profit_factor")),
    }


def _ledger_stats(df: pd.DataFrame, *, label: str, date_column: str) -> dict[str, Any]:
    if df.empty:
        return {
            "label": label,
            "trades": 0,
            "contracts": 0,
            "dates": 0,
            "avg_trades_per_date": None,
            "total_pnl": None,
            "mean_pnl_per_trade": None,
            "mean_pnl_per_contract": None,
            "profit_factor": None,
            "win_rate": None,
            "max_drawdown": None,
            "strategy_counts": {},
            "strategy_contracts": {},
        }
    weighted = pd.to_numeric(df["weighted_expected_pnl"], errors="coerce").fillna(0.0)
    quantity = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    trade_pnl = weighted.to_numpy(dtype=float)
    dated = pd.to_datetime(df[date_column], errors="coerce")
    daily = (
        pd.DataFrame({"date": dated.dt.date, "pnl": weighted})
        .dropna(subset=["date"])
        .groupby("date", dropna=False)["pnl"]
        .sum()
        .sort_index()
    )
    daily_pnl = daily.to_numpy(dtype=float)
    strategies = df.groupby("strategy", dropna=False)
    return {
        "label": label,
        "trades": int(len(df)),
        "contracts": int(quantity.sum()),
        "dates": int(daily.index.nunique()),
        "avg_trades_per_date": _rate_from_counts(len(df), max(int(daily.index.nunique()), 1)),
        "total_pnl": round(float(weighted.sum()), 6),
        "mean_pnl_per_trade": round(float(weighted.mean()), 6),
        "mean_pnl_per_contract": round(float(weighted.sum() / max(quantity.sum(), 1.0)), 6),
        "profit_factor": _profit_factor(trade_pnl),
        "win_rate": round(float((weighted > 0).mean()), 6),
        "max_drawdown": _max_drawdown(daily_pnl) if len(daily_pnl) else None,
        "strategy_counts": {str(key): int(len(frame)) for key, frame in strategies},
        "strategy_contracts": {str(key): int(pd.to_numeric(frame["quantity"], errors="coerce").fillna(0).sum()) for key, frame in strategies},
    }


def _build_open_book_comparison(
    ml_selected: pd.DataFrame,
    deterministic_selected: pd.DataFrame,
    close_quarters: list[str],
) -> dict[str, Any]:
    overall = _comparison_bucket(ml_selected, deterministic_selected, "overall")
    quarters = {
        quarter: _comparison_bucket(
            ml_selected.loc[ml_selected["close_quarter"] == quarter].copy(),
            deterministic_selected.loc[deterministic_selected["close_quarter"] == quarter].copy(),
            quarter,
        )
        for quarter in close_quarters
    }
    return {
        "overall_realized": overall,
        "close_quarters": quarters,
    }


def _comparison_bucket(ml_df: pd.DataFrame, deterministic_df: pd.DataFrame, label: str) -> dict[str, Any]:
    ml_stats = _ledger_stats(ml_df, label=label, date_column="exit_timestamp")
    det_stats = _ledger_stats(deterministic_df, label=label, date_column="exit_timestamp")
    return {
        "label": label,
        "ml": ml_stats,
        "deterministic": det_stats,
        "delta_ml_minus_det": {
            "trades": ml_stats["trades"] - det_stats["trades"],
            "contracts": ml_stats["contracts"] - det_stats["contracts"],
            "total_pnl": _delta(ml_stats.get("total_pnl"), det_stats.get("total_pnl")),
            "mean_pnl_per_trade": _delta(ml_stats.get("mean_pnl_per_trade"), det_stats.get("mean_pnl_per_trade")),
            "mean_pnl_per_contract": _delta(ml_stats.get("mean_pnl_per_contract"), det_stats.get("mean_pnl_per_contract")),
            "profit_factor": _delta(ml_stats.get("profit_factor"), det_stats.get("profit_factor")),
            "win_rate": _delta(ml_stats.get("win_rate"), det_stats.get("win_rate")),
            "max_drawdown": _delta(det_stats.get("max_drawdown"), ml_stats.get("max_drawdown")),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# True Open-Position Backtest",
        "",
        f"- Dataset: `{report['dataset_path']}`",
        f"- Entry years: {report['entry_window']['start_year']} to {report['entry_window']['end_year']}",
        "",
        "## Overall Realized",
        "",
    ]
    overall = report["comparison"]["overall_realized"]
    delta = overall["delta_ml_minus_det"]
    lines.extend(
        [
            f"- ML total realized PnL: {_fmt(overall['ml']['total_pnl'])}",
            f"- Deterministic total realized PnL: {_fmt(overall['deterministic']['total_pnl'])}",
            f"- Delta realized PnL (ML - deterministic): {_fmt(delta['total_pnl'])}",
            f"- Delta profit factor: {_fmt(delta['profit_factor'])}",
            f"- Delta max drawdown improvement: {_fmt(delta['max_drawdown'])}",
            "",
            "## Close Quarters",
            "",
            "| Quarter | ML PnL | Det PnL | Delta | ML PF | Det PF |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for quarter, payload in report["comparison"]["close_quarters"].items():
        lines.append(
            f"| {quarter} | {_fmt(payload['ml']['total_pnl'])} | {_fmt(payload['deterministic']['total_pnl'])} | "
            f"{_fmt(payload['delta_ml_minus_det']['total_pnl'])} | {_fmt(payload['ml']['profit_factor'])} | "
            f"{_fmt(payload['deterministic']['profit_factor'])} |"
        )
    return "\n".join(lines) + "\n"


def _prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["entry_timestamp"] = pd.to_datetime(working["entry_timestamp"], errors="coerce", utc=True)
    working = working.dropna(subset=["entry_timestamp"]).copy()
    working["entry_year"] = working["entry_timestamp"].dt.year
    return working


def _build_snapshot_map(df: pd.DataFrame) -> dict[tuple[str, date], Snapshot]:
    frame = df[["underlying", "entry_timestamp", "underlying_close", "implied_volatility"]].copy()
    frame["entry_date"] = frame["entry_timestamp"].dt.date
    grouped = (
        frame.groupby(["underlying", "entry_date"], dropna=False)
        .agg(
            spot=("underlying_close", "median"),
            iv=("implied_volatility", "median"),
        )
        .reset_index()
    )
    out: dict[tuple[str, date], Snapshot] = {}
    for row in grouped.itertuples(index=False):
        spot = float(row.spot) if row.spot is not None and not math.isnan(float(row.spot)) else 0.0
        iv = float(row.iv) if row.iv is not None and not math.isnan(float(row.iv)) else 0.25
        out[(str(row.underlying), row.entry_date)] = Snapshot(spot=spot, iv=iv)
    return out


def _quarter_labels_from_dates(start_date: date, end_date: date) -> list[str]:
    labels: list[str] = []
    year = start_date.year
    quarter = ((start_date.month - 1) // 3) + 1
    while True:
        labels.append(f"{year}Q{quarter}")
        if year == end_date.year and quarter == ((end_date.month - 1) // 3) + 1:
            break
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return labels


def _exit_timestamp(row: pd.Series) -> pd.Timestamp:
    exit_ts = pd.to_datetime(row.get("exit_timestamp"), errors="coerce", utc=True)
    if pd.notna(exit_ts):
        return exit_ts
    entry_ts = pd.to_datetime(row.get("entry_timestamp"), errors="coerce", utc=True)
    days = float(row.get("days_to_exit") or 0.0)
    return entry_ts + pd.to_timedelta(days=max(days, 0.0), unit="D")


def _expiry_iso(expiration: Any, entry_ts: pd.Timestamp, dte: int) -> str:
    if pd.notna(expiration):
        try:
            return pd.Timestamp(expiration).date().isoformat()
        except Exception:
            pass
    return (entry_ts.date() + timedelta(days=dte)).isoformat()


def _group_regime_label(group: pd.DataFrame) -> str:
    series = group.get("allocator_regime_label", pd.Series(index=group.index, dtype=object))
    non_null = series.dropna()
    if non_null.empty:
        return "GREEN"
    return str(non_null.iloc[0]).strip().upper() or "GREEN"


def _deployed_capital(open_positions: list[dict[str, Any]]) -> float:
    return round(
        float(
            sum(
                float(position["capital_per_contract"]) * int(position["quantity"])
                for position in open_positions
            )
        ),
        2,
    )


def _capital_for_pick(pick: dict[str, Any]) -> float:
    return abs(float(pick.get("short_strike") or 0.0) - float(pick.get("long_strike") or 0.0)) * 100.0


def _max_loss_per_contract_for_pick(pick: dict[str, Any]) -> float:
    width = abs(float(pick.get("short_strike") or 0.0) - float(pick.get("long_strike") or 0.0))
    premium = max(0.0, float(pick.get("premium") or 0.0))
    return max(0.0, round((width - premium) * 100.0, 2))


def _strategy_sides(strategy: str) -> set[str]:
    strategy = strategy.upper()
    if strategy == "PCS":
        return {"put"}
    if strategy == "CCS":
        return {"call"}
    return set()


def _value_counts(series: pd.Series | None) -> dict[str, int]:
    if series is None:
        return {}
    counts = series.astype("string").dropna().value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _rate_from_counts(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator / denominator), 6)


def _strip_selected_frame(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "selected"}


if __name__ == "__main__":
    raise SystemExit(main())
