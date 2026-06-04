"""Optuna hyperparameter search for the grouped intraday risk monitor."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import optuna

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.models.train_intraday_risk_monitor import train_intraday_risk_monitor

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_DEFAULT = (
    "artifacts/datasets/intraday_risk_rows/"
    "dataset_version="
    "intraday_risk_rows_parquet_regime_balanced_1m_broad_etfs_20220602_20260601_v001"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for the intraday XGBoost risk monitor.")
    parser.add_argument("--input", default=DATASET_DEFAULT)
    parser.add_argument("--target", default="stop_loss_hit_30m")
    parser.add_argument("--study-name", default="intraday_risk_monitor_hp_v001")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--storage-dir", default="artifacts/optuna")
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--walk-forward-folds", type=int, default=4)
    parser.add_argument("--min-walk-forward-train-groups", type=int, default=1000)
    parser.add_argument("--embargo-days", type=int, default=1)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)
    parser.add_argument("--min-threshold-recall-min", type=float, default=0.65)
    parser.add_argument("--min-threshold-recall-max", type=float, default=0.90)
    parser.add_argument("--max-threshold-close-rate", type=float, default=0.15)
    parser.add_argument("--max-threshold-false-close-rate", type=float, default=0.12)
    parser.add_argument("--feature-config", default=None, help="Path to a *_best_features.json file.")
    parser.add_argument("--include-features", default=None, help="Comma-separated feature columns to force include.")
    parser.add_argument("--output-artifact", default="artifacts/models/intraday_risk_monitor_v001.json")
    parser.add_argument(
        "--model-output",
        default="artifacts/models/intraday_risk_monitor_v001.xgboost.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    study_path = storage_dir / f"{args.study_name}.db"
    df = load_dataset(Path(args.input))
    include_features = _resolve_include_features(args)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 80.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 5.0, 100.0, log=True),
            "alpha": trial.suggest_float("alpha", 0.0, 5.0),
        }
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 25.0, log=True)
        min_threshold_recall = trial.suggest_float(
            "min_threshold_recall",
            args.min_threshold_recall_min,
            args.min_threshold_recall_max,
        )
        trial_model_path = storage_dir / "trials" / f"intraday_risk_trial_{trial.number}.xgboost.json"
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
            scale_pos_weight=scale_pos_weight,
            min_threshold_recall=min_threshold_recall,
            max_threshold_close_rate=args.max_threshold_close_rate,
            max_threshold_false_close_rate=args.max_threshold_false_close_rate,
            include_features=include_features,
            params=params,
        )
        trial_model_path.unlink(missing_ok=True)

        metrics = artifact.metrics
        wf_auc = float(metrics.get("walk_forward_auc_mean") or 0.0)
        wf_f2 = float(metrics.get("walk_forward_f2_mean") or 0.0)
        wf_recall_min = float(metrics.get("walk_forward_recall_min") or 0.0)
        wf_close_rate = float(metrics.get("walk_forward_close_rate_mean") or 1.0)
        wf_false_close = float(metrics.get("walk_forward_false_close_rate_mean") or 1.0)
        wf_profit_take_false_close = float(metrics.get("walk_forward_profit_take_false_close_rate_mean") or 1.0)
        wf_tp_lead = float(metrics.get("walk_forward_mean_true_positive_minutes_to_exit_mean") or 0.0)
        test_auc = float(metrics.get("test_auc") or 0.0)
        test_recall = float(metrics.get("test_recall") or 0.0)
        test_close_rate = float(metrics.get("test_close_rate") or 1.0)
        test_false_close_rate = float(metrics.get("test_false_close_rate") or 1.0)

        score = wf_auc + 0.35 * wf_f2 + 0.10 * test_auc + 0.10 * test_recall
        score += min(wf_tp_lead / 390.0, 2.0) * 0.05
        score -= 1.50 * max(0.0, min_threshold_recall - wf_recall_min)
        score -= 2.00 * max(0.0, wf_close_rate - args.max_threshold_close_rate)
        score -= 2.50 * max(0.0, wf_false_close - args.max_threshold_false_close_rate)
        score -= 1.50 * max(0.0, test_close_rate - args.max_threshold_close_rate)
        score -= 1.75 * max(0.0, test_false_close_rate - args.max_threshold_false_close_rate)
        score -= 0.75 * wf_profit_take_false_close

        trial.set_user_attr("recommended_close_threshold", artifact.recommended_close_threshold)
        trial.set_user_attr("included_features", include_features or artifact.feature_columns)
        trial.set_user_attr("walk_forward_auc_mean", round(wf_auc, 6))
        trial.set_user_attr("walk_forward_f2_mean", round(wf_f2, 6))
        trial.set_user_attr("walk_forward_recall_min", round(wf_recall_min, 6))
        trial.set_user_attr("walk_forward_close_rate_mean", round(wf_close_rate, 6))
        trial.set_user_attr("walk_forward_false_close_rate_mean", round(wf_false_close, 6))
        trial.set_user_attr(
            "walk_forward_profit_take_false_close_rate_mean",
            round(wf_profit_take_false_close, 6),
        )
        trial.set_user_attr("walk_forward_mean_true_positive_minutes_to_exit_mean", round(wf_tp_lead, 6))
        trial.set_user_attr("test_auc", round(test_auc, 6))
        trial.set_user_attr("test_recall", round(test_recall, 6))
        trial.set_user_attr("test_close_rate", round(test_close_rate, 6))
        trial.set_user_attr("test_false_close_rate", round(test_false_close_rate, 6))
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{study_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=17),
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = {
        "study_name": args.study_name,
        "input": args.input,
        "target": args.target,
        "score": study.best_value,
        "params": study.best_trial.params,
        "user_attrs": study.best_trial.user_attrs,
        "training_command": (
            "PYTHONPATH=. .venv/bin/python -m ml.models.train_intraday_risk_monitor "
            f"--input {args.input} "
            f"--output {args.output_artifact} "
            f"--model-output {args.model_output} "
            f"--target {args.target} "
            f"--walk-forward-folds {args.walk_forward_folds} "
            f"--min-walk-forward-train-groups {args.min_walk_forward_train_groups} "
            f"--embargo-days {args.embargo_days} "
            f"--num-boost-round {args.num_boost_round} "
            f"--val-fraction {args.val_fraction} "
            f"--early-stopping-rounds {args.early_stopping_rounds} "
            f"--eta {study.best_trial.params['eta']} "
            f"--max-depth {study.best_trial.params['max_depth']} "
            f"--min-child-weight {study.best_trial.params['min_child_weight']} "
            f"--subsample {study.best_trial.params['subsample']} "
            f"--colsample-bytree {study.best_trial.params['colsample_bytree']} "
            f"--reg-lambda {study.best_trial.params['lambda']} "
            f"--reg-alpha {study.best_trial.params['alpha']} "
            f"--scale-pos-weight {study.best_trial.params['scale_pos_weight']} "
            f"--min-threshold-recall {study.best_trial.params['min_threshold_recall']} "
            f"--max-threshold-close-rate {args.max_threshold_close_rate} "
            f"--max-threshold-false-close-rate {args.max_threshold_false_close_rate}"
        ),
    }
    included_features = study.best_trial.user_attrs.get("included_features", include_features)
    if included_features:
        best["training_command"] += f" --include-features {','.join(included_features)}"
    best_path = storage_dir / f"{args.study_name}_best_params.json"
    best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(best, indent=2, sort_keys=True))
    return 0


def _resolve_include_features(args: argparse.Namespace) -> list[str] | None:
    if args.include_features:
        return _parse_feature_list(args.include_features)
    if args.feature_config:
        payload = json.loads(Path(args.feature_config).read_text(encoding="utf-8"))
        features = payload.get("included_features")
        if isinstance(features, list):
            return [str(feature) for feature in features]
    return None


def _parse_feature_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
