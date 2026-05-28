"""Pass 2 — feature-group selection study for the XGBoost RoR ranker.

Fixes the best hyperparameters from Pass 1 and searches over which feature
groups to include. Groups are toggled at the group level (not individual
features) to keep the search space tractable for Optuna.

Usage:
    # After Pass 1 completes, run with its best-params JSON:
    PYTHONPATH=. .venv/bin/python -m ml.models.optimize_features \\
        --hp-params artifacts/optuna/ror_v006c_best_params.json \\
        --study-name feat_v006c \\
        --n-trials 64

    # Or supply hyperparameters manually:
    PYTHONPATH=. .venv/bin/python -m ml.models.optimize_features \\
        --study-name feat_v006c --n-trials 64 \\
        --eta 0.03 --max-depth 5 ...

Results persist in artifacts/optuna/<study-name>.db. Best feature set is
written to artifacts/optuna/<study-name>_best_features.json after every trial.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_DEFAULT = (
    "artifacts/datasets/candidate_rows/"
    "dataset_version="
    "candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k"
)
PYTHON = str(Path(sys.executable))

# ── Feature groups ────────────────────────────────────────────────────────────
# Always included: core trade economics that define the spread itself.
# Without these the model cannot distinguish trade quality at all.
ALWAYS_ON: dict[str, list[str]] = {
    "contract_structure": [
        "is_pcs", "is_ccs", "dte", "strike", "strike_distance_pct", "moneyness",
    ],
    "spread_structure": [
        "spread_width", "entry_credit", "max_profit", "max_loss", "credit_to_width",
        "long_option_entry_price", "long_option_entry_volume",
        "long_option_entry_trade_count", "long_option_entry_vwap",
    ],
}

# Toggleable: Optuna decides include (True) or exclude (False) for each group.
TOGGLEABLE: dict[str, list[str]] = {
    "underlying_price": [
        "underlying_close", "underlying_return_1d", "underlying_return_3d",
        "underlying_return_5d", "underlying_return_20d", "underlying_range_pct",
        "underlying_sma_20_distance_pct", "underlying_above_sma_20", "underlying_volume",
    ],
    "underlying_vol": [
        "underlying_realized_vol_5d", "underlying_realized_vol_10d",
        "underlying_realized_vol_20d", "underlying_skew_5d",
    ],
    "vol_momentum": [
        "underlying_volatility_ratio_5d_20d", "underlying_vol_vs_market", "vol_acceleration",
    ],
    "market_regime": [
        "market_return_5d", "market_return_20d", "market_realized_vol_5d",
        "market_realized_vol_20d", "market_sma_20_distance_pct",
        "market_above_sma_20", "market_volatility_ratio_5d_20d",
    ],
    "option_entry": [
        "option_entry_price", "option_entry_range_pct", "option_entry_volume",
        "option_entry_trade_count", "option_entry_vwap",
    ],
    "option_activity": [
        "option_volume_5d_avg", "option_trade_count_5d_avg", "option_activity_spike",
    ],
    "greeks": [
        "implied_volatility", "option_delta", "option_gamma", "option_theta", "option_vega",
    ],
    "iv_surface": [
        "iv_vs_hv5d", "iv_vs_hv20d", "iv_skew_wing",
    ],
    "vix_features": [
        "vix_regime", "vix_return_5d", "vix_realized_vol_5d",
    ],
    "event_risk": [
        "days_to_earnings", "days_to_ex_dividend", "days_to_fomc", "days_to_macro_event",
    ],
    "credit_efficiency": [
        "credit_per_day_per_risk",
    ],
}

# Champion baseline — all groups included, no exclusions
CHAMPION_GROUPS = {g: True for g in TOGGLEABLE}

# Default hyperparameters (Pass 1 champion, overridden if --hp-params supplied)
DEFAULT_HP = dict(
    eta=0.05, max_depth=3, min_child_weight=5.0, subsample=0.85,
    colsample_bytree=0.85, reg_lambda=10.0, reg_alpha=0.0,
    downside_penalty=2.5, huber_delta=1.0,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Feature-group selection study (Pass 2) for the XGBoost RoR ranker."
    )
    p.add_argument("--input", default=DATASET_DEFAULT)
    p.add_argument("--study-name", default="feat_tuning")
    p.add_argument("--n-trials", type=int, default=64)
    p.add_argument("--storage-dir", default="artifacts/optuna")
    p.add_argument("--num-boost-round", type=int, default=500)
    # Hyperparameter source — prefer loading from Pass 1 output JSON
    p.add_argument(
        "--hp-params",
        default=None,
        help="Path to Pass 1 best_params.json. If supplied, hyperparameters are loaded from it.",
    )
    # Manual HP overrides (used when --hp-params is not available)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-child-weight", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample-bytree", type=float, default=None)
    p.add_argument("--reg-lambda", type=float, default=None)
    p.add_argument("--reg-alpha", type=float, default=None)
    p.add_argument("--downside-penalty", type=float, default=None)
    p.add_argument("--huber-delta", type=float, default=None)
    return p.parse_args()


def _load_hp(args: argparse.Namespace) -> dict:
    """Resolve hyperparameters: file > CLI overrides > defaults."""
    hp = dict(DEFAULT_HP)

    if args.hp_params:
        path = Path(args.hp_params)
        if not path.exists():
            raise FileNotFoundError(f"--hp-params file not found: {path}")
        data = json.loads(path.read_text())
        hp.update(data["params"])
        print(f"Loaded hyperparameters from: {path}")
    else:
        print("No --hp-params supplied — using DEFAULT_HP (Pass 1 champion defaults).")

    # CLI overrides take precedence over file values
    overrides = dict(
        eta=args.eta, max_depth=args.max_depth, min_child_weight=args.min_child_weight,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
        downside_penalty=args.downside_penalty, huber_delta=args.huber_delta,
    )
    for k, v in overrides.items():
        if v is not None:
            hp[k] = v

    return hp


def _excluded_features(group_flags: dict[str, bool]) -> list[str]:
    excluded = []
    for group, include in group_flags.items():
        if not include:
            excluded.extend(TOGGLEABLE[group])
    return excluded


def _score(report: dict) -> tuple[float, dict[str, float]]:
    h = report["holdout"]
    wf = report["walk_forward"]

    ror = h["mean_return_on_risk"]
    pf = h["profit_factor"]
    win_rate = h["win_rate"]
    wf_pf_min = wf["top_profit_factor_min"]
    wf_pf_avg = wf["top_profit_factor_mean"]

    score = ror
    score += 0.05 * max(0.0, wf_pf_avg - 1.0)
    score -= 0.10 * max(0.0, 1.4 - pf)
    score -= 0.10 * max(0.0, 1.2 - wf_pf_min)

    return score, {
        "holdout_ror": round(ror, 6),
        "holdout_pf": round(pf, 6),
        "holdout_win_rate": round(win_rate, 6),
        "wf_pf_min": round(wf_pf_min, 6),
        "wf_pf_avg": round(wf_pf_avg, 6),
    }


def _objective(
    trial: optuna.Trial,
    dataset: str,
    hp: dict,
    num_boost_round: int,
    storage_dir: Path,
) -> float:
    group_flags = {g: trial.suggest_categorical(g, [True, False]) for g in TOGGLEABLE}
    excluded = _excluded_features(group_flags)

    n_included = sum(1 for v in group_flags.values() if v)
    n_total_features = (
        sum(len(v) for v in ALWAYS_ON.values())
        + sum(len(TOGGLEABLE[g]) for g, inc in group_flags.items() if inc)
    )

    trial_dir = storage_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = trial_dir / f"feat_trial_{trial.number}.json"
    model_path = trial_dir / f"feat_trial_{trial.number}.xgboost.json"
    eval_path = trial_dir / f"feat_trial_{trial.number}_exit_criteria.json"

    env = {**os.environ, "PYTHONPATH": "."}

    cmd_train = [
        PYTHON, "-m", "ml.models.train_xgboost",
        "--input", dataset,
        "--output", str(artifact_path),
        "--target", "return_on_risk",
        "--target-scale", "0.10",
        "--target-clip", "5.0",
        "--num-boost-round", str(num_boost_round),
        "--val-fraction", "0.0",
        "--early-stopping-rounds", "0",
        "--embargo-days", "30",
        "--eta", str(hp["eta"]),
        "--max-depth", str(hp["max_depth"]),
        "--min-child-weight", str(hp["min_child_weight"]),
        "--subsample", str(hp["subsample"]),
        "--colsample-bytree", str(hp["colsample_bytree"]),
        "--reg-lambda", str(hp["reg_lambda"]),
        "--reg-alpha", str(hp["reg_alpha"]),
        "--downside-penalty", str(hp["downside_penalty"]),
        "--huber-delta", str(hp["huber_delta"]),
    ]
    if excluded:
        cmd_train += ["--exclude-features", ",".join(excluded)]

    t0 = time.time()
    r = subprocess.run(cmd_train, capture_output=True, env=env)
    if r.returncode != 0:
        print(f"\n[feat trial {trial.number}] TRAIN FAILED:\n{r.stderr.decode()[-800:]}")
        return float("-inf")

    cmd_eval = [
        PYTHON, "-m", "ml.models.evaluate_exit_criteria",
        "--input", dataset,
        "--artifact", str(artifact_path),
        "--json-output", str(eval_path),
    ]
    r = subprocess.run(cmd_eval, capture_output=True, env=env)
    elapsed = time.time() - t0

    artifact_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)

    if r.returncode != 0:
        print(f"\n[feat trial {trial.number}] EVAL FAILED:\n{r.stderr.decode()[-800:]}")
        return float("-inf")

    with open(eval_path) as f:
        report = json.load(f)

    composite, metrics = _score(report)

    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("n_groups_included", n_included)
    trial.set_user_attr("n_features", n_total_features)
    trial.set_user_attr("excluded_groups", [g for g, inc in group_flags.items() if not inc])
    trial.set_user_attr("elapsed_s", round(elapsed))

    included_str = "+".join(g for g, inc in group_flags.items() if inc)
    print(
        f"[feat {trial.number:03d}] score={composite:.4f}  "
        f"RoR={metrics['holdout_ror']:.4f}  PF={metrics['holdout_pf']:.3f}  "
        f"WR={metrics['holdout_win_rate']:.3f}  "
        f"groups={n_included}/{len(TOGGLEABLE)}  feats={n_total_features}  "
        f"({elapsed:.0f}s)"
    )
    if metrics["holdout_ror"] < 0.10:
        print(f"  excluded: {[g for g, inc in group_flags.items() if not inc]}")

    return composite


def _save_best(study: optuna.Study, dataset: str, hp: dict, num_boost_round: int, storage_dir: Path) -> None:
    best = study.best_trial
    attrs = best.user_attrs
    group_flags = {g: best.params[g] for g in TOGGLEABLE}
    excluded = _excluded_features(group_flags)

    exclude_arg = f"--exclude-features {','.join(excluded)}" if excluded else ""
    hp_args = " \\\n  ".join(
        f"--{k.replace('_', '-')} {v}" for k, v in hp.items()
    )
    train_cmd = (
        f"PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \\\n"
        f"  --input {dataset} \\\n"
        f"  --output artifacts/models/xgboost_v006c.json \\\n"
        f"  --target return_on_risk --target-scale 0.10 --target-clip 5.0 \\\n"
        f"  --num-boost-round {num_boost_round} \\\n"
        f"  --val-fraction 0.0 --early-stopping-rounds 0 --embargo-days 30 \\\n"
        f"  {hp_args}"
    )
    if exclude_arg:
        train_cmd += f" \\\n  {exclude_arg}"

    out = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "score": best.value,
        "metrics": attrs,
        "hyperparameters": hp,
        "group_flags": group_flags,
        "excluded_features": excluded,
        "included_groups": [g for g, inc in group_flags.items() if inc],
        "excluded_groups": attrs.get("excluded_groups", []),
        "train_command": train_cmd,
    }
    path = storage_dir / f"{study.study_name}_best_features.json"
    path.write_text(json.dumps(out, indent=2))


def _print_summary(study: optuna.Study, hp: dict) -> None:
    best = study.best_trial
    attrs = best.user_attrs
    group_flags = {g: best.params[g] for g in TOGGLEABLE}

    print(f"\n{'=' * 65}")
    print(f"Best feature trial: #{best.number}  (score={best.value:.4f})")
    print(f"  holdout RoR    = {attrs.get('holdout_ror', '?'):.4f}  (champion: 0.1519)")
    print(f"  holdout PF     = {attrs.get('holdout_pf', '?'):.3f}  (champion: 1.959)")
    print(f"  holdout WR     = {attrs.get('holdout_win_rate', '?'):.3f}  (champion: 0.756)")
    print(f"  WF PF min      = {attrs.get('wf_pf_min', '?'):.3f}  (champion: 1.554)")
    print(f"  Groups used    = {attrs.get('n_groups_included', '?')}/{len(TOGGLEABLE)}")
    print(f"  Total features = {attrs.get('n_features', '?')}")
    print(f"\nGroup decisions (best trial):")
    for group, include in group_flags.items():
        tag = "  INCLUDED" if include else "  dropped "
        print(f"  {tag}  {group}")
    print(f"\nFixed hyperparameters used:")
    for k, v in hp.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 65}")


def main() -> None:
    args = _parse_args()
    hp = _load_hp(args)

    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    db_path = storage_dir / f"{args.study_name}.db"
    storage_url = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("Seeding trial 0 with all feature groups included (champion baseline)...")
        study.enqueue_trial(CHAMPION_GROUPS)
    else:
        print(f"Resuming study with {len(completed)} completed trial(s) already in storage.")

    print(f"Study: {args.study_name}")
    print(f"Storage: {db_path}")
    print(f"Toggleable groups: {list(TOGGLEABLE.keys())}")
    print(f"Running {args.n_trials} trial(s). Ctrl-C to stop early (progress is saved).\n")

    objective = lambda trial: _objective(
        trial, args.input, hp, args.num_boost_round, storage_dir
    )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=False,
            callbacks=[
                lambda study, trial: _save_best(study, args.input, hp, args.num_boost_round, storage_dir)
                if trial.state == optuna.trial.TrialState.COMPLETE
                else None
            ],
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving best params so far.")

    _print_summary(study, hp)
    _save_best(study, args.input, hp, args.num_boost_round, storage_dir)
    best_path = storage_dir / f"{args.study_name}_best_features.json"
    print(f"\nBest feature set saved to: {best_path}")
    out = json.loads(best_path.read_text())
    print("\nCanonical retrain command:")
    print(out["train_command"])


if __name__ == "__main__":
    main()
