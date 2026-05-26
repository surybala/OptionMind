import pandas as pd

from ml.models.evaluate_exit_criteria import (
    CriterionResult,
    ExitCriteriaConfig,
    _evaluate_gates,
    _selection_metrics,
)


def test_selection_metrics_reports_tail_and_concentration():
    df = pd.DataFrame(
        {
            "prediction": [0.1, 0.2, 0.3, 0.4],
            "expected_pnl": [-100, 50, 100, -25],
            "max_profit": [100, 50, 100, 25],
            "max_adverse_excursion": [125, 25, 10, 50],
            "return_on_risk": [-1.0, 0.5, 1.0, -0.25],
            "large_loss_label": [1, 0, 0, 0],
            "stop_loss_hit": [1, 0, 0, 1],
            "strategy": ["PCS", "PCS", "CCS", "CCS"],
            "underlying": ["SPY", "SPY", "QQQ", "QQQ"],
        }
    )

    metrics = _selection_metrics(df, "prediction", ExitCriteriaConfig(selection_fraction=0.5))

    assert metrics["selected_rows"] == 2
    assert metrics["mean_pnl"] == 37.5
    assert metrics["profit_factor"] == 4.0
    assert metrics["slippage_adjusted_mean_pnl"] == 25.0
    assert metrics["win_rate"] == 0.5
    assert metrics["large_loss_rate"] == 0.0
    assert metrics["max_adverse_excursion"] == 50.0
    assert metrics["single_underlying_share"] == 1.0


def test_exit_gates_fail_on_unstable_walk_forward():
    cfg = ExitCriteriaConfig(
        min_dataset_rows=4,
        min_test_rows=2,
        min_top_selection_rows=1,
        min_top_selection_entry_dates=1,
    )
    df = pd.DataFrame({"expected_pnl": [1, 2, 3, 4]})
    artifact = {
        "metrics": {
            "train_top_decile_actual_mean": 100.0,
            "train_top_decile_profit_factor": 3.0,
        }
    }
    holdout = {
        "rows": 2,
        "selected_rows": 1,
        "selected_entry_dates": 1,
        "single_underlying_share": 0.10,
        "top5_underlying_share": 0.50,
        "mean_pnl": 30.0,
        "slippage_adjusted_mean_pnl": 20.0,
        "slippage_adjusted_profit_factor": 1.2,
        "profit_factor": 1.50,
        "win_rate": 0.70,
        "mean_return_on_risk": 0.30,
        "p05_pnl": -200.0,
        "p01_pnl": -300.0,
        "worst_pnl": -400.0,
        "large_loss_rate": 0.05,
        "stop_loss_rate": 0.10,
        "max_drawdown": 100.0,
        "max_adverse_excursion": 100.0,
        "drawdown_to_gross_profit": 0.10,
    }
    walk_forward = {
        "folds": 3,
        "top_mean_pnl_min": -5.0,
        "top_mean_pnl_mean": 20.0,
        "top_profit_factor_min": 1.10,
        "top_profit_factor_mean": 1.30,
        "top_win_rate_min": 0.60,
        "top_p05_pnl_min": -250.0,
        "top_worst_pnl_min": -1000.0,
        "top_max_drawdown_max": 100.0,
        "top_max_adverse_excursion_max": 100.0,
        "feature_stability": {"top_feature_overlap": 3},
    }

    results = _evaluate_gates(df, artifact, holdout, walk_forward, cfg)
    failed = {item.name for item in results if not item.passed}

    assert "walk_forward_min_top_mean_pnl" in failed
    assert "walk_forward_avg_top_mean_pnl" in failed
    assert "walk_forward_min_top_profit_factor" in failed
    assert "walk_forward_avg_top_profit_factor" in failed
    assert all(isinstance(item, CriterionResult) for item in results)


def test_exit_gates_fail_when_fold_feature_stability_is_missing():
    cfg = ExitCriteriaConfig(
        min_dataset_rows=1,
        min_test_rows=1,
        min_top_selection_rows=1,
        min_top_selection_entry_dates=1,
        min_walk_forward_folds=1,
    )
    df = pd.DataFrame({"expected_pnl": [1]})
    artifact = {"metrics": {"train_top_decile_actual_mean": 10.0, "train_top_decile_profit_factor": 1.5}}
    holdout = {
        "rows": 1,
        "selected_rows": 1,
        "selected_entry_dates": 1,
        "single_underlying_share": 0.0,
        "top5_underlying_share": 0.0,
        "mean_pnl": 30.0,
        "slippage_adjusted_mean_pnl": 20.0,
        "slippage_adjusted_profit_factor": 1.2,
        "profit_factor": 1.5,
        "win_rate": 0.7,
        "mean_return_on_risk": 0.3,
        "p05_pnl": -100.0,
        "p01_pnl": -100.0,
        "worst_pnl": -100.0,
        "large_loss_rate": 0.0,
        "stop_loss_rate": 0.0,
        "max_drawdown": 100.0,
        "max_adverse_excursion": 100.0,
        "drawdown_to_gross_profit": 0.1,
    }
    walk_forward = {
        "folds": 1,
        "top_mean_pnl_min": 30.0,
        "top_mean_pnl_mean": 30.0,
        "top_profit_factor_min": 1.5,
        "top_profit_factor_mean": 1.5,
        "top_win_rate_min": 0.7,
        "top_p05_pnl_min": -100.0,
        "top_worst_pnl_min": -100.0,
        "top_max_drawdown_max": 100.0,
        "top_max_adverse_excursion_max": 100.0,
        "feature_stability": {"top_feature_overlap": None},
    }

    results = _evaluate_gates(df, artifact, holdout, walk_forward, cfg)

    assert "walk_forward_top_feature_overlap" in {item.name for item in results if not item.passed}
