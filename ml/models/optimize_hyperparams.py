"""Optuna hyperparameter search for the XGBoost RoR ranker.

Objective: maximise holdout mean_return_on_risk while penalising
walk-forward instability and profit-factor collapse.

Usage:
    PYTHONPATH=. .venv/bin/python -m ml.models.optimize_hyperparams \\
        --input artifacts/datasets/candidate_rows/<dataset_version> \\
        --study-name ror_v006c \\
        --n-trials 100

Results persist in artifacts/optuna/<study-name>.db (SQLite) so you can
stop and resume at any time. Best params are written to
artifacts/optuna/<study-name>_best_params.json after every trial.
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna search for XGBoost RoR ranker hyperparameters."
    )
    p.add_argument("--input", default=DATASET_DEFAULT)
    p.add_argument("--study-name", default="ror_tuning")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--storage-dir", default="artifacts/optuna")
    p.add_argument("--num-boost-round", type=int, default=500)
    return p.parse_args()


def _score(report: dict) -> tuple[float, dict[str, float]]:
    """Return composite Optuna score and the raw metrics dict."""
    h = report["holdout"]
    wf = report["walk_forward"]

    ror = h["mean_return_on_risk"]
    pf = h["profit_factor"]
    win_rate = h["win_rate"]
    wf_pf_min = wf["top_profit_factor_min"]
    wf_pf_avg = wf["top_profit_factor_mean"]
    wf_ror_avg = wf["top_mean_pnl_mean"]  # dollar PnL (no per-fold RoR in report)

    # Primary: maximise holdout RoR.
    # Bonus: walk-forward PF stability (rewards generalisable signal).
    # Penalties: if holdout PF or WF-min-PF collapse, subtract hard.
    score = ror
    score += 0.05 * max(0.0, wf_pf_avg - 1.0)
    score -= 0.10 * max(0.0, 1.4 - pf)
    score -= 0.10 * max(0.0, 1.2 - wf_pf_min)

    metrics = {
        "holdout_ror": round(ror, 6),
        "holdout_pf": round(pf, 6),
        "holdout_win_rate": round(win_rate, 6),
        "wf_pf_min": round(wf_pf_min, 6),
        "wf_pf_avg": round(wf_pf_avg, 6),
        "wf_pnl_avg": round(wf_ror_avg, 4),
    }
    return score, metrics


def _objective(
    trial: optuna.Trial,
    dataset: str,
    num_boost_round: int,
    storage_dir: Path,
) -> float:
    eta = trial.suggest_float("eta", 0.01, 0.20, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 30.0, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.5, 50.0, log=True)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 5.0)
    downside_penalty = trial.suggest_float("downside_penalty", 1.5, 5.0)
    huber_delta = trial.suggest_float("huber_delta", 0.3, 3.0)

    trial_dir = storage_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = trial_dir / f"trial_{trial.number}.json"
    model_path = trial_dir / f"trial_{trial.number}.xgboost.json"
    eval_path = trial_dir / f"trial_{trial.number}_exit_criteria.json"

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
        "--eta", str(eta),
        "--max-depth", str(max_depth),
        "--min-child-weight", str(min_child_weight),
        "--subsample", str(subsample),
        "--colsample-bytree", str(colsample_bytree),
        "--reg-lambda", str(reg_lambda),
        "--reg-alpha", str(reg_alpha),
        "--downside-penalty", str(downside_penalty),
        "--huber-delta", str(huber_delta),
    ]

    t0 = time.time()
    r = subprocess.run(cmd_train, capture_output=True, env=env)
    if r.returncode != 0:
        stderr = r.stderr.decode()[-1000:]
        print(f"\n[trial {trial.number}] TRAIN FAILED:\n{stderr}")
        return float("-inf")

    cmd_eval = [
        PYTHON, "-m", "ml.models.evaluate_exit_criteria",
        "--input", dataset,
        "--artifact", str(artifact_path),
        "--json-output", str(eval_path),
    ]

    r = subprocess.run(cmd_eval, capture_output=True, env=env)
    elapsed = time.time() - t0

    # Always clean up the model binary and artifact JSON — they are large
    artifact_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)

    if r.returncode != 0:
        stderr = r.stderr.decode()[-1000:]
        print(f"\n[trial {trial.number}] EVAL FAILED:\n{stderr}")
        return float("-inf")

    with open(eval_path) as f:
        report = json.load(f)

    composite, metrics = _score(report)

    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("elapsed_s", round(elapsed))

    print(
        f"[trial {trial.number:03d}] score={composite:.4f}  "
        f"RoR={metrics['holdout_ror']:.4f}  "
        f"PF={metrics['holdout_pf']:.3f}  "
        f"WR={metrics['holdout_win_rate']:.3f}  "
        f"WF_PF_min={metrics['wf_pf_min']:.3f}  "
        f"({elapsed:.0f}s)"
    )

    return composite


def _save_best(study: optuna.Study, dataset: str, num_boost_round: int, storage_dir: Path) -> None:
    best = study.best_trial
    params = best.params
    attrs = best.user_attrs

    train_cmd_parts = [
        f"PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \\",
        f"  --input {dataset} \\",
        f"  --output artifacts/models/xgboost_v006c.json \\",
        f"  --target return_on_risk --target-scale 0.10 --target-clip 5.0 \\",
        f"  --num-boost-round {num_boost_round} \\",
        f"  --val-fraction 0.0 --early-stopping-rounds 0 \\",
        f"  --embargo-days 30 \\",
    ]
    for k, v in params.items():
        train_cmd_parts.append(f"  --{k.replace('_', '-')} {v} \\")
    train_cmd = "\n".join(train_cmd_parts).rstrip(" \\")

    out = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "score": best.value,
        "metrics": attrs,
        "params": params,
        "train_command": train_cmd,
    }
    path = storage_dir / f"{study.study_name}_best_params.json"
    path.write_text(json.dumps(out, indent=2))


def _print_summary(study: optuna.Study) -> None:
    best = study.best_trial
    attrs = best.user_attrs
    params = best.params
    print(f"\n{'=' * 65}")
    print(f"Best trial: #{best.number}  (score={best.value:.4f})")
    print(f"  holdout RoR    = {attrs.get('holdout_ror', '?'):.4f}  (champion: 0.1519)")
    print(f"  holdout PF     = {attrs.get('holdout_pf', '?'):.3f}  (champion: 1.959)")
    print(f"  holdout WR     = {attrs.get('holdout_win_rate', '?'):.3f}  (champion: 0.756)")
    print(f"  WF PF min      = {attrs.get('wf_pf_min', '?'):.3f}  (champion: 1.554)")
    print(f"  WF PF avg      = {attrs.get('wf_pf_avg', '?'):.3f}  (champion: 1.813)")
    print(f"\nBest hyperparameters vs champion defaults:")
    defaults = dict(
        eta=0.05, max_depth=3, min_child_weight=5.0, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=10.0, reg_alpha=0.0,
        downside_penalty=2.5, huber_delta=1.0,
    )
    for k, v in params.items():
        arrow = "  <-- changed" if abs(v - defaults.get(k, v)) > 1e-9 else ""
        print(f"  {k:<22} {v:.5g}  (was {defaults.get(k, '?'):.5g}){arrow}")
    print(f"\n{'=' * 65}")


def main() -> None:
    args = _parse_args()
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

    # Seed trial 0 with champion hyperparameters so TPE has a known-good anchor
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("Seeding trial 0 with champion hyperparameters...")
        study.enqueue_trial({
            "eta": 0.05,
            "max_depth": 3,
            "min_child_weight": 5.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 10.0,
            "reg_alpha": 0.0,
            "downside_penalty": 2.5,
            "huber_delta": 1.0,
        })
    else:
        print(f"Resuming study with {len(completed)} completed trial(s) already in storage.")

    print(f"Study: {args.study_name}")
    print(f"Storage: {db_path}")
    print(f"Running {args.n_trials} trial(s). Ctrl-C to stop early (progress is saved).\n")

    objective = lambda trial: _objective(
        trial, args.input, args.num_boost_round, storage_dir
    )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=False,
            callbacks=[
                lambda study, trial: _save_best(study, args.input, args.num_boost_round, storage_dir)
                if trial.state == optuna.trial.TrialState.COMPLETE
                else None
            ],
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving best params so far.")

    _print_summary(study)
    _save_best(study, args.input, args.num_boost_round, storage_dir)
    best_path = storage_dir / f"{args.study_name}_best_params.json"
    print(f"\nBest params saved to: {best_path}")
    print("\nCanonical retrain command:")
    out = json.loads(best_path.read_text())
    print(out["train_command"])


if __name__ == "__main__":
    main()
