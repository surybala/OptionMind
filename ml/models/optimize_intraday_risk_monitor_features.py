"""Optuna feature-group search for the intraday risk monitor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import optuna

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.intraday_risk_feature_groups import ALWAYS_ON, CHAMPION_GROUPS, TOGGLEABLE
from ml.models.train_intraday_risk_monitor import train_intraday_risk_monitor

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_DEFAULT = (
    "artifacts/datasets/intraday_risk_rows/"
    "dataset_version="
    "intraday_risk_rows_parquet_regime_seed_raw_broad_etfs_20220602_20260601_v001"
)
HP_DEFAULT = "artifacts/optuna/intraday_risk_monitor_hp_v001_best_params.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna feature-group search for the intraday XGBoost risk monitor.")
    parser.add_argument("--input", default=DATASET_DEFAULT)
    parser.add_argument("--target", default="stop_loss_hit_30m")
    parser.add_argument("--study-name", default="intraday_risk_monitor_feat_v001")
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--storage-dir", default="artifacts/optuna")
    parser.add_argument("--hp-params", default=HP_DEFAULT)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--min-walk-forward-train-groups", type=int, default=1000)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--min-threshold-recall", type=float, default=0.60)
    parser.add_argument("--max-threshold-close-rate", type=float, default=0.12)
    parser.add_argument("--max-threshold-false-close-rate", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_path = storage_dir / f"{args.study_name}.db"
    df = load_dataset(Path(args.input))
    hp = _load_hp_params(Path(args.hp_params), args.num_boost_round)

    def objective(trial: optuna.Trial) -> float:
        group_flags = {group: trial.suggest_categorical(group, [True, False]) for group in TOGGLEABLE}
        include_features = _included_features(group_flags)
        trial_model_path = storage_dir / "trials" / f"intraday_risk_feat_trial_{trial.number}.xgboost.json"
        trial_model_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = train_intraday_risk_monitor(
            df,
            model_output=trial_model_path,
            target_column=args.target,
            walk_forward_folds=args.walk_forward_folds,
            min_walk_forward_train_groups=args.min_walk_forward_train_groups,
            embargo_days=args.embargo_days,
            num_boost_round=args.num_boost_round,
            val_fraction=args.val_fraction,
            early_stopping_rounds=args.early_stopping_rounds,
            scale_pos_weight=hp["scale_pos_weight"],
            min_threshold_recall=args.min_threshold_recall,
            max_threshold_close_rate=args.max_threshold_close_rate,
            max_threshold_false_close_rate=args.max_threshold_false_close_rate,
            include_features=include_features,
            params=hp["params"],
        )
        trial_model_path.unlink(missing_ok=True)

        metrics = artifact.metrics
        wf_auc = float(metrics.get("walk_forward_auc_mean") or 0.0)
        wf_f2 = float(metrics.get("walk_forward_f2_mean") or 0.0)
        wf_recall_min = float(metrics.get("walk_forward_recall_min") or 0.0)
        wf_close = float(metrics.get("walk_forward_close_rate_mean") or 1.0)
        wf_false_close = float(metrics.get("walk_forward_false_close_rate_mean") or 1.0)
        wf_profit_take_false_close = float(metrics.get("walk_forward_profit_take_false_close_rate_mean") or 1.0)
        wf_tp_lead = float(metrics.get("walk_forward_mean_true_positive_minutes_to_exit_mean") or 0.0)
        test_auc = float(metrics.get("test_auc") or 0.0)
        test_recall = float(metrics.get("test_recall") or 0.0)
        test_close = float(metrics.get("test_close_rate") or 1.0)
        test_false_close = float(metrics.get("test_false_close_rate") or 1.0)

        score = wf_auc + 0.35 * wf_f2 + 0.10 * test_auc + 0.10 * test_recall
        score += min(wf_tp_lead / 390.0, 2.0) * 0.05
        score -= 1.50 * max(0.0, args.min_threshold_recall - wf_recall_min)
        score -= 2.00 * max(0.0, wf_close - args.max_threshold_close_rate)
        score -= 2.50 * max(0.0, wf_false_close - args.max_threshold_false_close_rate)
        score -= 1.50 * max(0.0, test_close - args.max_threshold_close_rate)
        score -= 1.75 * max(0.0, test_false_close - args.max_threshold_false_close_rate)
        score -= 0.75 * wf_profit_take_false_close

        trial.set_user_attr("included_groups", [group for group, enabled in group_flags.items() if enabled])
        trial.set_user_attr("excluded_groups", [group for group, enabled in group_flags.items() if not enabled])
        trial.set_user_attr("included_features", include_features)
        trial.set_user_attr("recommended_close_threshold", artifact.recommended_close_threshold)
        trial.set_user_attr("walk_forward_auc_mean", round(wf_auc, 6))
        trial.set_user_attr("walk_forward_f2_mean", round(wf_f2, 6))
        trial.set_user_attr("walk_forward_recall_min", round(wf_recall_min, 6))
        trial.set_user_attr("walk_forward_close_rate_mean", round(wf_close, 6))
        trial.set_user_attr("walk_forward_false_close_rate_mean", round(wf_false_close, 6))
        trial.set_user_attr(
            "walk_forward_profit_take_false_close_rate_mean",
            round(wf_profit_take_false_close, 6),
        )
        trial.set_user_attr("walk_forward_mean_true_positive_minutes_to_exit_mean", round(wf_tp_lead, 6))
        trial.set_user_attr("test_auc", round(test_auc, 6))
        trial.set_user_attr("test_recall", round(test_recall, 6))
        trial.set_user_attr("test_close_rate", round(test_close, 6))
        trial.set_user_attr("test_false_close_rate", round(test_false_close, 6))
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17),
    )
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        study.enqueue_trial(CHAMPION_GROUPS)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        callbacks=[lambda current_study, trial: _save_best(current_study, args, hp, storage_dir)],
    )
    _save_best(study, args, hp, storage_dir)
    best_path = storage_dir / f"{args.study_name}_best_features.json"
    print(best_path.read_text(encoding="utf-8"))
    return 0


def _load_hp_params(path: Path, num_boost_round: int) -> dict[str, object]:
    params = {
        "max_depth": 5,
        "eta": 0.026674693796823173,
        "subsample": 0.9705203508329543,
        "colsample_bytree": 0.717786894882143,
        "min_child_weight": 7.949216978230875,
        "lambda": 7.161579772528922,
        "alpha": 1.8885267774102836,
    }
    scale_pos_weight = 5.63643038573485
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        params.update(
            {
                "max_depth": int(payload["params"]["max_depth"]),
                "eta": float(payload["params"]["eta"]),
                "subsample": float(payload["params"]["subsample"]),
                "colsample_bytree": float(payload["params"]["colsample_bytree"]),
                "min_child_weight": float(payload["params"]["min_child_weight"]),
                "lambda": float(payload["params"]["lambda"]),
                "alpha": float(payload["params"]["alpha"]),
            }
        )
        scale_pos_weight = float(payload["params"]["scale_pos_weight"])
    return {
        "params": params,
        "scale_pos_weight": scale_pos_weight,
        "num_boost_round": num_boost_round,
    }


def _included_features(group_flags: dict[str, bool]) -> list[str]:
    features: list[str] = []
    for columns in ALWAYS_ON.values():
        features.extend(columns)
    for group, columns in TOGGLEABLE.items():
        if group_flags.get(group):
            features.extend(columns)
    seen: set[str] = set()
    ordered: list[str] = []
    for feature in features:
        if feature in seen:
            continue
        seen.add(feature)
        ordered.append(feature)
    return ordered


def _save_best(study: optuna.Study, args: argparse.Namespace, hp: dict[str, object], storage_dir: Path) -> None:
    if study.best_trial is None:
        return
    best = study.best_trial
    include_features = list(best.user_attrs.get("included_features", []))
    train_command = (
        "PYTHONPATH=. .venv/bin/python -m ml.models.train_intraday_risk_monitor "
        f"--input {args.input} "
        "--output artifacts/models/intraday_risk_monitor_stop30m_fullraw_v002.json "
        "--model-output artifacts/models/intraday_risk_monitor_stop30m_fullraw_v002.xgboost.json "
        f"--target {args.target} "
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
        f"--min-threshold-recall {args.min_threshold_recall} "
        f"--max-threshold-close-rate {args.max_threshold_close_rate} "
        f"--max-threshold-false-close-rate {args.max_threshold_false_close_rate} "
        f"--include-features {','.join(include_features)}"
    )
    payload = {
        "study_name": args.study_name,
        "input": args.input,
        "target": args.target,
        "score": study.best_value,
        "best_trial": best.number,
        "hyperparameters": hp,
        "group_flags": {group: best.params[group] for group in TOGGLEABLE},
        "included_groups": best.user_attrs.get("included_groups", []),
        "excluded_groups": best.user_attrs.get("excluded_groups", []),
        "included_features": include_features,
        "metrics": best.user_attrs,
        "train_command": train_command,
    }
    path = storage_dir / f"{args.study_name}_best_features.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
