"""Unified 2-pass Optuna optimizer for the intraday risk monitor.

Pass 1 (hp):       search XGBoost hyper-parameters at a fixed operating threshold.
Pass 2 (features): search feature-group toggles using the best HPs from pass 1.

Usage:
    # HP search at threshold 0.08
    PYTHONPATH=. .venv/bin/python -m ml.models.optimize_intraday_risk_v2 \
        --mode hp --threshold 0.08 --study-name irm_hp_t008_v004 --n-trials 50

    # Feature search using HP results
    PYTHONPATH=. .venv/bin/python -m ml.models.optimize_intraday_risk_v2 \
        --mode features --threshold 0.08 --study-name irm_feat_t008_v004 --n-trials 24 \
        --hp-params artifacts/optuna/irm_hp_t008_v004_best_params.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import optuna

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.intraday_risk_feature_groups import ALWAYS_ON, CHAMPION_GROUPS, TOGGLEABLE
from ml.models.train_intraday_risk_monitor import train_intraday_risk_monitor

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_DEFAULT = (
    "artifacts/datasets/intraday_risk_rows/"
    "dataset_version="
    "intraday_risk_rows_parquet_regime_balanced_1m_broad_etfs_20220602_20260601_v001"
)

RECALL_FLOOR: dict[float, float] = {
    0.05: 0.65,
    0.08: 0.55,
    0.10: 0.48,
    0.12: 0.42,
    0.15: 0.35,
}

FALSE_CLOSE_CEIL: dict[float, float] = {
    0.05: 0.10,
    0.08: 0.07,
    0.10: 0.06,
    0.12: 0.05,
    0.15: 0.04,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified 2-pass Optuna optimizer for intraday risk monitor."
    )
    parser.add_argument("--mode", required=True, choices=["hp", "features"])
    parser.add_argument("--threshold", type=float, required=True,
                        help="Fixed operating threshold for metric evaluation.")
    parser.add_argument("--input", default=DATASET_DEFAULT)
    parser.add_argument("--target", default="stop_loss_hit_30m")
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--storage-dir", default="artifacts/optuna")
    parser.add_argument("--hp-params", default=None,
                        help="Path to best HP params JSON (required for features mode).")
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--min-walk-forward-train-groups", type=int, default=1000)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    return parser.parse_args()


def _score(
    metrics: dict[str, Any],
    threshold: float,
    recall_floor: float,
    false_close_ceil: float,
) -> float:
    wf_auc = float(metrics.get("walk_forward_auc_mean") or 0.0)
    wf_f2 = float(metrics.get("walk_forward_f2_mean") or 0.0)
    wf_recall_min = float(metrics.get("walk_forward_recall_min") or 0.0)
    wf_close = float(metrics.get("walk_forward_close_rate_mean") or 1.0)
    wf_false_close = float(metrics.get("walk_forward_false_close_rate_mean") or 1.0)
    wf_profit_take_fc = float(metrics.get("walk_forward_profit_take_false_close_rate_mean") or 1.0)
    wf_tp_lead = float(metrics.get("walk_forward_mean_true_positive_minutes_to_exit_mean") or 0.0)
    test_auc = float(metrics.get("test_auc") or 0.0)
    test_recall = float(metrics.get("test_recall") or 0.0)
    test_false_close = float(metrics.get("test_false_close_rate") or 1.0)

    score = wf_auc + 0.35 * wf_f2 + 0.10 * test_auc + 0.10 * test_recall
    score += min(wf_tp_lead / 390.0, 2.0) * 0.05
    score -= 1.50 * max(0.0, recall_floor - wf_recall_min)
    score -= 2.50 * max(0.0, wf_false_close - false_close_ceil)
    score -= 1.75 * max(0.0, test_false_close - false_close_ceil)
    score -= 0.75 * wf_profit_take_fc
    return score


def _set_user_attrs(trial: optuna.Trial, metrics: dict[str, Any], threshold: float) -> None:
    for key in (
        "walk_forward_auc_mean", "walk_forward_f2_mean", "walk_forward_recall_min",
        "walk_forward_close_rate_mean", "walk_forward_false_close_rate_mean",
        "walk_forward_profit_take_false_close_rate_mean",
        "walk_forward_mean_true_positive_minutes_to_exit_mean",
        "test_auc", "test_recall", "test_close_rate", "test_false_close_rate",
    ):
        val = metrics.get(key)
        if val is not None:
            trial.set_user_attr(key, round(float(val), 6))
    trial.set_user_attr("fixed_threshold", threshold)


def _load_hp(path: str | None) -> dict[str, Any]:
    defaults = {
        "params": {
            "max_depth": 5,
            "eta": 0.025,
            "subsample": 0.80,
            "colsample_bytree": 0.58,
            "min_child_weight": 39.0,
            "lambda": 5.15,
            "alpha": 0.77,
        },
        "scale_pos_weight": 10.09,
    }
    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_params = payload.get("params", {})
        defaults["params"] = {
            "max_depth": int(raw_params.get("max_depth", 5)),
            "eta": float(raw_params.get("eta", 0.025)),
            "subsample": float(raw_params.get("subsample", 0.80)),
            "colsample_bytree": float(raw_params.get("colsample_bytree", 0.58)),
            "min_child_weight": float(raw_params.get("min_child_weight", 39.0)),
            "lambda": float(raw_params.get("lambda", 5.15)),
            "alpha": float(raw_params.get("alpha", 0.77)),
        }
        defaults["scale_pos_weight"] = float(raw_params.get("scale_pos_weight", 10.09))
    return defaults


def _included_features(group_flags: dict[str, bool]) -> list[str]:
    features: list[str] = []
    for columns in ALWAYS_ON.values():
        features.extend(columns)
    for group, columns in TOGGLEABLE.items():
        if group_flags.get(group):
            features.extend(columns)
    seen: set[str] = set()
    ordered: list[str] = []
    for f in features:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _run_hp_search(args: argparse.Namespace, df, storage_dir: Path) -> None:
    study_path = storage_dir / f"{args.study_name}.db"
    threshold = args.threshold
    recall_floor = RECALL_FLOOR.get(threshold, 0.45)
    false_close_ceil = FALSE_CLOSE_CEIL.get(threshold, 0.06)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 80.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "lambda": trial.suggest_float("lambda", 3.0, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
        }
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 30.0, log=True)
        trial_model = storage_dir / "trials" / f"irm_hp_trial_{trial.number}.xgboost.json"
        trial_model.parent.mkdir(parents=True, exist_ok=True)

        artifact = train_intraday_risk_monitor(
            df,
            model_output=trial_model,
            target_column=args.target,
            walk_forward_folds=args.walk_forward_folds,
            min_walk_forward_train_groups=args.min_walk_forward_train_groups,
            embargo_days=args.embargo_days,
            num_boost_round=args.num_boost_round,
            val_fraction=args.val_fraction,
            early_stopping_rounds=args.early_stopping_rounds,
            scale_pos_weight=scale_pos_weight,
            fixed_threshold=threshold,
            params=params,
        )
        trial_model.unlink(missing_ok=True)

        s = _score(artifact.metrics, threshold, recall_floor, false_close_ceil)
        _set_user_attrs(trial, artifact.metrics, threshold)
        trial.set_user_attr("recommended_close_threshold", artifact.recommended_close_threshold)
        return s

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17),
    )
    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, args.n_trials - completed)
    if remaining > 0:
        print(f"[HP] {completed} completed, running {remaining} more trials at threshold {threshold}")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"[HP] Already have {completed} completed trials, skipping")

    _save_hp_best(study, args, storage_dir)


def _save_hp_best(study: optuna.Study, args: argparse.Namespace, storage_dir: Path) -> None:
    if not study.best_trial:
        return
    best = study.best_trial
    payload = {
        "study_name": args.study_name,
        "input": args.input,
        "target": args.target,
        "threshold": args.threshold,
        "score": study.best_value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    path = storage_dir / f"{args.study_name}_best_params.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best HP saved to {path}")
    print(f"  Score: {study.best_value:.6f}")
    for k in ("walk_forward_auc_mean", "test_recall", "test_false_close_rate", "walk_forward_recall_min"):
        print(f"  {k}: {best.user_attrs.get(k, '?')}")


def _run_feat_search(args: argparse.Namespace, df, storage_dir: Path) -> None:
    if not args.hp_params:
        raise ValueError("--hp-params is required for features mode")
    study_path = storage_dir / f"{args.study_name}.db"
    threshold = args.threshold
    recall_floor = RECALL_FLOOR.get(threshold, 0.45)
    false_close_ceil = FALSE_CLOSE_CEIL.get(threshold, 0.06)
    hp = _load_hp(args.hp_params)

    def objective(trial: optuna.Trial) -> float:
        group_flags = {group: trial.suggest_categorical(group, [True, False]) for group in TOGGLEABLE}
        include_features = _included_features(group_flags)
        trial_model = storage_dir / "trials" / f"irm_feat_trial_{trial.number}.xgboost.json"
        trial_model.parent.mkdir(parents=True, exist_ok=True)

        artifact = train_intraday_risk_monitor(
            df,
            model_output=trial_model,
            target_column=args.target,
            walk_forward_folds=args.walk_forward_folds,
            min_walk_forward_train_groups=args.min_walk_forward_train_groups,
            embargo_days=args.embargo_days,
            num_boost_round=args.num_boost_round,
            val_fraction=args.val_fraction,
            early_stopping_rounds=args.early_stopping_rounds,
            scale_pos_weight=hp["scale_pos_weight"],
            fixed_threshold=threshold,
            include_features=include_features,
            params=hp["params"],
        )
        trial_model.unlink(missing_ok=True)

        s = _score(artifact.metrics, threshold, recall_floor, false_close_ceil)
        _set_user_attrs(trial, artifact.metrics, threshold)
        trial.set_user_attr("included_groups", [g for g, on in group_flags.items() if on])
        trial.set_user_attr("excluded_groups", [g for g, on in group_flags.items() if not on])
        trial.set_user_attr("included_features", include_features)
        trial.set_user_attr("recommended_close_threshold", artifact.recommended_close_threshold)
        return s

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17),
    )
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.enqueue_trial(CHAMPION_GROUPS)

    completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    remaining = max(0, args.n_trials - completed)
    if remaining > 0:
        print(f"[FEAT] {completed} completed, running {remaining} more trials at threshold {threshold}")
        study.optimize(objective, n_trials=remaining)
    else:
        print(f"[FEAT] Already have {completed} completed trials, skipping")

    _save_feat_best(study, args, hp, storage_dir)


def _save_feat_best(
    study: optuna.Study, args: argparse.Namespace, hp: dict[str, Any], storage_dir: Path
) -> None:
    if not study.best_trial:
        return
    best = study.best_trial
    include_features = list(best.user_attrs.get("included_features", []))
    train_cmd = (
        "PYTHONPATH=. .venv/bin/python -m ml.models.train_intraday_risk_monitor "
        f"--input {args.input} "
        f"--output artifacts/models/intraday_risk_monitor_stop30m_v004_t{int(args.threshold*100):02d}.json "
        f"--model-output artifacts/models/intraday_risk_monitor_stop30m_v004_t{int(args.threshold*100):02d}.xgboost.json "
        f"--target {args.target} "
        f"--fixed-threshold {args.threshold} "
        f"--walk-forward-folds {args.walk_forward_folds} "
        f"--min-walk-forward-train-groups {args.min_walk_forward_train_groups} "
        f"--embargo-days {args.embargo_days} "
        f"--num-boost-round {args.num_boost_round} "
        f"--val-fraction {args.val_fraction} "
        f"--early-stopping-rounds {args.early_stopping_rounds} "
        f"--eta {hp['params']['eta']} "
        f"--max-depth {hp['params']['max_depth']} "
        f"--min-child-weight {hp['params']['min_child_weight']} "
        f"--subsample {hp['params']['subsample']} "
        f"--colsample-bytree {hp['params']['colsample_bytree']} "
        f"--reg-lambda {hp['params']['lambda']} "
        f"--reg-alpha {hp['params']['alpha']} "
        f"--scale-pos-weight {hp['scale_pos_weight']} "
        f"--include-features {','.join(include_features)}"
    )
    payload = {
        "study_name": args.study_name,
        "input": args.input,
        "target": args.target,
        "threshold": args.threshold,
        "score": study.best_value,
        "best_trial": best.number,
        "hyperparameters": hp,
        "group_flags": {group: best.params[group] for group in TOGGLEABLE},
        "included_groups": best.user_attrs.get("included_groups", []),
        "excluded_groups": best.user_attrs.get("excluded_groups", []),
        "included_features": include_features,
        "metrics": {k: v for k, v in best.user_attrs.items() if k not in ("included_features", "included_groups", "excluded_groups")},
        "train_command": train_cmd,
    }
    path = storage_dir / f"{args.study_name}_best_features.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Best features saved to {path}")
    print(f"  Score: {study.best_value:.6f}")
    print(f"  Included: {best.user_attrs.get('included_groups', [])}")
    print(f"  Excluded: {best.user_attrs.get('excluded_groups', [])}")
    for k in ("walk_forward_auc_mean", "test_recall", "test_false_close_rate", "walk_forward_recall_min"):
        print(f"  {k}: {best.user_attrs.get(k, '?')}")


def main() -> int:
    args = parse_args()
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.input} ...")
    df = load_dataset(Path(args.input))
    print(f"Loaded {len(df):,} rows")

    if args.mode == "hp":
        _run_hp_search(args, df, storage_dir)
    else:
        _run_feat_search(args, df, storage_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
