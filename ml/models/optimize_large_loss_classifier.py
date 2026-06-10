"""Two-pass Optuna optimiser for the large-loss (and stop-loss) classifier.

Pass 1 — hyperparameter search
    Searches eta, max_depth, min_child_weight, subsample, colsample_bytree,
    reg_lambda, reg_alpha, scale_pos_weight over ~75 trials.
    Objective: walk_forward_auc_mean + small recall-floor penalty.

Pass 2 — feature-group selection
    Fixes the best hyperparameters from Pass 1 and searches which feature
    groups to include. Same 11 toggleable groups as the ranker optimizer;
    the classifier may legitimately keep VIX/market-regime features that hurt
    the ranker, since macro fear is genuinely predictive of large losses.

Works for both targets:
    --target large_loss_label   (default)
    --target stop_loss_hit

Threshold-aware: metrics (recall, precision, F1) are evaluated at the
operating veto threshold (0.60 for LLC, 0.30 for SLC by default) so the
optimizer maximises signal quality at the actual decision boundary.

Usage
-----
# LLC Pass 1
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_large_loss_classifier \\
    --mode hp --target large_loss_label \\
    --study-name llc_hp_v008 --n-trials 75

# LLC Pass 2 (after Pass 1 finishes)
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_large_loss_classifier \\
    --mode features --target large_loss_label \\
    --study-name llc_feat_v008 --n-trials 48 \\
    --hp-params artifacts/optuna/llc_hp_v008_best_params.json

# SLC Pass 1
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_large_loss_classifier \\
    --mode hp --target stop_loss_hit \\
    --study-name slc_hp_v008 --n-trials 75

# SLC Pass 2
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_large_loss_classifier \\
    --mode features --target stop_loss_hit \\
    --study-name slc_feat_v008 --n-trials 48 \\
    --hp-params artifacts/optuna/slc_hp_v008_best_params.json

Results persist in artifacts/optuna/<study-name>.db.
Best params/features are written to artifacts/optuna/ after every trial.
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

from ml.models.feature_groups import ALWAYS_ON, CHAMPION_GROUPS, TOGGLEABLE

DATASET_DEFAULT = (
    "artifacts/datasets/candidate_rows/"
    "dataset_version="
    "candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21"
)
PYTHON = str(Path(sys.executable))

# ── Operating thresholds ─────────────────────────────────────────────────────
# Must match the veto thresholds in config.json / model_scanner.py.
DEFAULT_THRESHOLD: dict[str, float] = {
    "large_loss_label": 0.60,
    "stop_loss_hit": 0.30,
}

# ── Baseline metrics (V006b defaults on DTE≤21 dataset, at operating threshold) ─
BASELINE: dict[str, dict[str, float]] = {
    "large_loss_label": {
        "test_auc": 0.838150,
        "walk_forward_auc_mean": 0.844341,
        "test_recall": 0.689003,
    },
    "stop_loss_hit": {
        "test_auc": 0.797349,
        "walk_forward_auc_mean": 0.815971,
        "test_recall": 0.951192,
    },
}

# Recall floor per target at operating threshold.
# LLC@0.60: baseline recall ~0.69; floor at 0.60 gives ~9pp headroom for AUC improvement.
# SLC@0.30: baseline recall ~0.95; floor at 0.92 gives ~3pp headroom.
RECALL_FLOOR: dict[str, float] = {
    "large_loss_label": 0.60,
    "stop_loss_hit": 0.92,
}

# Default hyperparameters (V006b baseline)
DEFAULT_HP: dict[str, float | int] = dict(
    eta=0.05, max_depth=4, min_child_weight=20.0,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=20.0,
    reg_alpha=0.0, scale_pos_weight=0.0,  # 0.0 = auto-compute from data
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two-pass Optuna optimiser for the large-loss / stop-loss classifier."
    )
    p.add_argument("--mode", choices=["hp", "features"], required=True,
                   help="hp = Pass 1 hyperparameter search. features = Pass 2 feature-group selection.")
    p.add_argument("--target", default="large_loss_label",
                   choices=["large_loss_label", "stop_loss_hit"])
    p.add_argument("--input", default=DATASET_DEFAULT)
    p.add_argument("--study-name", default=None,
                   help="Defaults to llc_hp_<target> or llc_feat_<target>.")
    p.add_argument("--n-trials", type=int, default=75)
    p.add_argument("--threshold", type=float, default=None,
                   help="Classification threshold for recall/precision metrics. "
                        "Auto-resolves from --target if not set (0.60 for LLC, 0.30 for SLC).")
    p.add_argument("--storage-dir", default="artifacts/optuna")
    p.add_argument("--num-boost-round", type=int, default=500,
                   help="Max rounds (early stopping will usually terminate well before this).")
    # Pass 2 only: source of locked hyperparameters
    p.add_argument("--hp-params", default=None,
                   help="Path to Pass 1 best_params.json. Required for --mode features.")
    # Manual HP overrides
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-child-weight", type=float, default=None)
    p.add_argument("--subsample", type=float, default=None)
    p.add_argument("--colsample-bytree", type=float, default=None)
    p.add_argument("--reg-lambda", type=float, default=None)
    p.add_argument("--reg-alpha", type=float, default=None)
    p.add_argument("--scale-pos-weight", type=float, default=None,
                   help="0.0 = auto-compute from data (default).")
    return p.parse_args()


def _resolve_study_name(args: argparse.Namespace) -> str:
    if args.study_name:
        return args.study_name
    suffix = "large_loss" if args.target == "large_loss_label" else "stop_loss"
    return f"llc_{args.mode}_{suffix}"


def _load_hp(args: argparse.Namespace) -> dict:
    hp = dict(DEFAULT_HP)
    if args.hp_params:
        path = Path(args.hp_params)
        if not path.exists():
            raise FileNotFoundError(f"--hp-params file not found: {path}")
        data = json.loads(path.read_text())
        hp.update(data["params"])
        print(f"Loaded hyperparameters from: {path}")
    else:
        print("No --hp-params supplied — using V006b defaults.")
    overrides = dict(
        eta=args.eta, max_depth=args.max_depth, min_child_weight=args.min_child_weight,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
        scale_pos_weight=args.scale_pos_weight,
    )
    for k, v in overrides.items():
        if v is not None:
            hp[k] = v
    return hp


def _excluded_features(group_flags: dict[str, bool]) -> list[str]:
    return [feat for g, inc in group_flags.items() if not inc for feat in TOGGLEABLE[g]]


def _score(metrics: dict, target: str) -> tuple[float, dict]:
    """
    Primary objective: walk_forward_auc_mean.
    Penalty: 0.05 per unit recall drops below the per-target recall floor
    (evaluated at the operating threshold, not the old 0.15).
    """
    wf_auc = float(metrics.get("walk_forward_auc_mean") or 0.0)
    test_recall = float(metrics.get("test_recall") or 0.0)
    test_auc = float(metrics.get("test_auc") or 0.0)

    score = wf_auc
    recall_floor = RECALL_FLOOR.get(target, 0.80)
    score -= 0.05 * max(0.0, recall_floor - test_recall)

    baseline = BASELINE.get(target, {})
    return score, {
        "wf_auc": round(wf_auc, 6),
        "test_auc": round(test_auc, 6),
        "test_recall": round(test_recall, 6),
        "vs_baseline_wf_auc": round(wf_auc - float(baseline.get("walk_forward_auc_mean", 0)), 6),
    }


def _run_trial(
    trial_number: int,
    dataset: str,
    target: str,
    hp: dict,
    num_boost_round: int,
    storage_dir: Path,
    excluded_features: list[str] | None = None,
    threshold: float = 0.15,
) -> dict | None:
    trial_dir = storage_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    prefix = "large_loss" if target == "large_loss_label" else "stop_loss"
    artifact_path = trial_dir / f"{prefix}_trial_{trial_number}.json"
    model_path = trial_dir / f"{prefix}_trial_{trial_number}.xgboost.json"

    cmd = [
        PYTHON, "-m", "ml.models.train_large_loss_classifier",
        "--input", dataset,
        "--output", str(artifact_path),
        "--target", target,
        "--num-boost-round", str(num_boost_round),
        "--val-fraction", "0.15",
        "--early-stopping-rounds", "20",
        "--embargo-days", "30",
        "--threshold", str(threshold),
        "--eta", str(hp["eta"]),
        "--max-depth", str(int(hp["max_depth"])),
        "--min-child-weight", str(hp["min_child_weight"]),
        "--subsample", str(hp["subsample"]),
        "--colsample-bytree", str(hp["colsample_bytree"]),
        "--reg-lambda", str(hp["reg_lambda"]),
    ]
    if float(hp.get("reg_alpha", 0.0)) > 0:
        cmd += ["--reg-alpha", str(hp["reg_alpha"])]  # pass only when non-zero
    spw = float(hp.get("scale_pos_weight", 0.0))
    if spw > 0:
        cmd += ["--scale-pos-weight", str(spw)]
    if excluded_features:
        cmd += ["--exclude-features", ",".join(excluded_features)]

    env = {**os.environ, "PYTHONPATH": "."}
    r = subprocess.run(cmd, capture_output=True, env=env)

    artifact_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)

    if r.returncode != 0:
        print(f"\n[trial {trial_number}] FAILED:\n{r.stderr.decode()[-600:]}")
        return None

    try:
        art = json.loads(r.stdout.decode())
    except Exception:
        print(f"\n[trial {trial_number}] JSON parse error")
        return None

    return art.get("metrics", {})


# ── Pass 1: hyperparameter search ─────────────────────────────────────────────

def _hp_objective(
    trial: optuna.Trial,
    dataset: str,
    target: str,
    num_boost_round: int,
    storage_dir: Path,
    threshold: float = 0.15,
) -> float:
    hp = {
        "eta":              trial.suggest_float("eta", 0.005, 0.10, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 100.0, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 5.0, 200.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 5.0),
        # scale_pos_weight=0 means auto-compute; Optuna searches the manual range
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 3.0, 25.0, log=True),
    }

    t0 = time.time()
    metrics = _run_trial(trial.number, dataset, target, hp, num_boost_round, storage_dir,
                         threshold=threshold)
    elapsed = time.time() - t0

    if metrics is None:
        return float("-inf")

    score, diag = _score(metrics, target)
    for k, v in diag.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("elapsed_s", round(elapsed))
    trial.set_user_attr("threshold", threshold)

    baseline_wf = BASELINE.get(target, {}).get("walk_forward_auc_mean", 0)
    print(
        f"[hp {trial.number:03d}] score={score:.4f}  "
        f"wf_auc={diag['wf_auc']:.4f} ({diag['vs_baseline_wf_auc']:+.4f} vs baseline={baseline_wf:.4f})  "
        f"recall@{threshold}={diag['test_recall']:.4f}  ({elapsed:.0f}s)"
    )
    return score


def _save_hp_best(study: optuna.Study, dataset: str, num_boost_round: int,
                  target: str, storage_dir: Path, threshold: float = 0.15) -> None:
    best = study.best_trial
    spw = best.params.get("scale_pos_weight", 0.0)
    hp_args = " \\\n  ".join(
        f"--{k.replace('_', '-')} {v}" for k, v in best.params.items()
    )
    prefix = "large_loss_classifier" if target == "large_loss_label" else "stop_loss_classifier"
    train_cmd = (
        f"PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \\\n"
        f"  --input {dataset} \\\n"
        f"  --output artifacts/models/{prefix}_v008.json \\\n"
        f"  --target {target} --embargo-days 30 \\\n"
        f"  --threshold {threshold} \\\n"
        f"  --num-boost-round {num_boost_round} \\\n"
        f"  {hp_args}"
    )
    out = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "score": best.value,
        "threshold": threshold,
        "metrics": best.user_attrs,
        "params": best.params,
        "train_command": train_cmd,
    }
    path = storage_dir / f"{study.study_name}_best_params.json"
    path.write_text(json.dumps(out, indent=2))


# ── Pass 2: feature-group selection ──────────────────────────────────────────

def _feat_objective(
    trial: optuna.Trial,
    dataset: str,
    target: str,
    hp: dict,
    num_boost_round: int,
    storage_dir: Path,
    threshold: float = 0.15,
) -> float:
    group_flags = {g: trial.suggest_categorical(g, [True, False]) for g in TOGGLEABLE}
    excluded = _excluded_features(group_flags)

    t0 = time.time()
    metrics = _run_trial(trial.number, dataset, target, hp, num_boost_round, storage_dir,
                         excluded_features=excluded if excluded else None,
                         threshold=threshold)
    elapsed = time.time() - t0

    if metrics is None:
        return float("-inf")

    score, diag = _score(metrics, target)
    n_included = sum(1 for v in group_flags.values() if v)
    n_features = (
        sum(len(v) for v in ALWAYS_ON.values())
        + sum(len(TOGGLEABLE[g]) for g, inc in group_flags.items() if inc)
    )

    for k, v in diag.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("excluded_groups", [g for g, inc in group_flags.items() if not inc])
    trial.set_user_attr("n_features", n_features)
    trial.set_user_attr("elapsed_s", round(elapsed))
    trial.set_user_attr("threshold", threshold)

    baseline_wf = BASELINE.get(target, {}).get("walk_forward_auc_mean", 0)
    print(
        f"[feat {trial.number:03d}] score={score:.4f}  "
        f"wf_auc={diag['wf_auc']:.4f} ({diag['vs_baseline_wf_auc']:+.4f} vs baseline)  "
        f"recall@{threshold}={diag['test_recall']:.4f}  "
        f"groups={n_included}/{len(TOGGLEABLE)}  feats={n_features}  ({elapsed:.0f}s)"
    )
    return score


def _save_feat_best(study: optuna.Study, dataset: str, hp: dict, num_boost_round: int,
                    target: str, storage_dir: Path, threshold: float = 0.15) -> None:
    best = study.best_trial
    group_flags = {g: best.params[g] for g in TOGGLEABLE}
    excluded = _excluded_features(group_flags)
    prefix = "large_loss_classifier" if target == "large_loss_label" else "stop_loss_classifier"

    hp_args = " \\\n  ".join(f"--{k.replace('_', '-')} {v}" for k, v in hp.items()
                              if k != "scale_pos_weight" or float(v) > 0)
    train_cmd = (
        f"PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \\\n"
        f"  --input {dataset} \\\n"
        f"  --output artifacts/models/{prefix}_v008.json \\\n"
        f"  --target {target} --embargo-days 30 \\\n"
        f"  --threshold {threshold} \\\n"
        f"  --num-boost-round {num_boost_round} \\\n"
        f"  {hp_args}"
    )
    if excluded:
        train_cmd += f" \\\n  --exclude-features {','.join(excluded)}"

    out = {
        "study_name": study.study_name,
        "best_trial": best.number,
        "score": best.value,
        "threshold": threshold,
        "metrics": best.user_attrs,
        "hyperparameters": hp,
        "group_flags": group_flags,
        "included_groups": [g for g, inc in group_flags.items() if inc],
        "excluded_groups": best.user_attrs.get("excluded_groups", []),
        "excluded_features": excluded,
        "train_command": train_cmd,
    }
    path = storage_dir / f"{study.study_name}_best_features.json"
    path.write_text(json.dumps(out, indent=2))


# ── Summary printers ──────────────────────────────────────────────────────────

def _print_hp_summary(study: optuna.Study, target: str) -> None:
    best = study.best_trial
    a = best.user_attrs
    baseline = BASELINE.get(target, {})
    print(f"\n{'=' * 65}")
    print(f"Best HP trial: #{best.number}  (score={best.value:.4f})")
    print(f"  wf_auc     = {a.get('wf_auc', '?'):.4f}  (V006b: {baseline.get('walk_forward_auc_mean', '?')})")
    print(f"  test_auc   = {a.get('test_auc', '?'):.4f}  (V006b: {baseline.get('test_auc', '?')})")
    print(f"  test_recall= {a.get('test_recall', '?'):.4f}  (V006b: {baseline.get('test_recall', '?')})")
    print(f"\nBest params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 65}")


def _print_feat_summary(study: optuna.Study, target: str) -> None:
    best = study.best_trial
    a = best.user_attrs
    baseline = BASELINE.get(target, {})
    group_flags = {g: best.params[g] for g in TOGGLEABLE}
    print(f"\n{'=' * 65}")
    print(f"Best feature trial: #{best.number}  (score={best.value:.4f})")
    print(f"  wf_auc     = {a.get('wf_auc', '?'):.4f}  (V006b: {baseline.get('walk_forward_auc_mean', '?')})")
    print(f"  test_recall= {a.get('test_recall', '?'):.4f}  (V006b: {baseline.get('test_recall', '?')})")
    print(f"  n_features = {a.get('n_features', '?')}")
    print(f"\nGroup decisions:")
    for g, inc in group_flags.items():
        print(f"  {'INCLUDED' if inc else 'dropped '} {g}")
    print(f"{'=' * 65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_name = _resolve_study_name(args)
    db_path = storage_dir / f"{study_name}.db"
    storage_url = f"sqlite:///{db_path}"
    baseline = BASELINE.get(args.target, {})
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD.get(args.target, 0.15)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Study: {study_name}  |  mode: {args.mode}  |  target: {args.target}  |  threshold: {threshold}")
    print(f"Storage: {db_path}")
    print(f"Baseline: wf_auc={baseline.get('walk_forward_auc_mean', '?')}, "
          f"recall@{threshold}={baseline.get('test_recall', '?')}")
    print(f"Recall floor: {RECALL_FLOOR.get(args.target, '?')}")
    print(f"Running {args.n_trials} trial(s). Ctrl-C to stop early (progress is saved).\n")

    if args.mode == "hp":
        if not completed:
            print("Seeding trial 0 with defaults...")
            study.enqueue_trial({
                "eta": DEFAULT_HP["eta"],
                "max_depth": int(DEFAULT_HP["max_depth"]),
                "min_child_weight": DEFAULT_HP["min_child_weight"],
                "subsample": DEFAULT_HP["subsample"],
                "colsample_bytree": DEFAULT_HP["colsample_bytree"],
                "reg_lambda": DEFAULT_HP["reg_lambda"],
                "reg_alpha": float(DEFAULT_HP["reg_alpha"]),
                "scale_pos_weight": 8.41,
            })
        else:
            print(f"Resuming with {len(completed)} completed trial(s) in storage.")

        objective = lambda trial: _hp_objective(
            trial, args.input, args.target, args.num_boost_round, storage_dir,
            threshold=threshold,
        )
        try:
            study.optimize(
                objective,
                n_trials=args.n_trials,
                show_progress_bar=False,
                callbacks=[
                    lambda study, trial: _save_hp_best(
                        study, args.input, args.num_boost_round, args.target, storage_dir,
                        threshold=threshold,
                    ) if trial.state == optuna.trial.TrialState.COMPLETE else None
                ],
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving best params so far.")

        _print_hp_summary(study, args.target)
        _save_hp_best(study, args.input, args.num_boost_round, args.target, storage_dir,
                      threshold=threshold)
        best_path = storage_dir / f"{study_name}_best_params.json"
        print(f"\nBest params saved to: {best_path}")
        out = json.loads(best_path.read_text())
        print("\nCanonical retrain command:")
        print(out["train_command"])

    elif args.mode == "features":
        if not args.hp_params:
            print("WARNING: --hp-params not supplied. Using defaults for hyperparameters.")
        hp = _load_hp(args)

        if not completed:
            print("Seeding trial 0 with all feature groups included (champion baseline)...")
            study.enqueue_trial(CHAMPION_GROUPS)
        else:
            print(f"Resuming with {len(completed)} completed trial(s) in storage.")

        objective = lambda trial: _feat_objective(
            trial, args.input, args.target, hp, args.num_boost_round, storage_dir,
            threshold=threshold,
        )
        try:
            study.optimize(
                objective,
                n_trials=args.n_trials,
                show_progress_bar=False,
                callbacks=[
                    lambda study, trial: _save_feat_best(
                        study, args.input, hp, args.num_boost_round, args.target, storage_dir,
                        threshold=threshold,
                    ) if trial.state == optuna.trial.TrialState.COMPLETE else None
                ],
            )
        except KeyboardInterrupt:
            print("\nInterrupted — saving best features so far.")

        _print_feat_summary(study, args.target)
        _save_feat_best(study, args.input, hp, args.num_boost_round, args.target, storage_dir,
                        threshold=threshold)
        best_path = storage_dir / f"{study_name}_best_features.json"
        print(f"\nBest feature set saved to: {best_path}")
        out = json.loads(best_path.read_text())
        print("\nCanonical retrain command:")
        print(out["train_command"])


if __name__ == "__main__":
    main()
