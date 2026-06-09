import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.models import backtest_agent_like as target


def test_run_backtest_applies_stop_loss_veto_after_large_loss_gate(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    registry_path = tmp_path / "registry.json"
    large_loss_artifact = tmp_path / "large_loss.json"
    stop_loss_artifact = tmp_path / "stop_loss.json"
    ranker_artifact = tmp_path / "ranker.json"

    config_path.write_text(
        json.dumps(
            {
                "ml_scanner": {
                    "min_dte": 7,
                    "max_dte": 21,
                    "large_loss_classifier_path": str(large_loss_artifact),
                    "large_loss_veto_threshold": 0.60,
                    "stop_loss_classifier_path": str(stop_loss_artifact),
                    "stop_loss_veto_threshold": 0.30,
                    "top_n": 10,
                },
                "pick_selection": {"mode": "model_ranked"},
                "max_capital_per_period": 50000.0,
            }
        ),
        encoding="utf-8",
    )
    registry_path.write_text(json.dumps({"champion_model_id": "stub", "models": []}), encoding="utf-8")
    large_loss_artifact.write_text("{}", encoding="utf-8")
    stop_loss_artifact.write_text("{}", encoding="utf-8")
    ranker_artifact.write_text("{}", encoding="utf-8")

    dataset = pd.DataFrame(
        {
            "entry_timestamp": [
                "2024-01-03T14:30:00+00:00",
                "2024-01-04T14:30:00+00:00",
                "2024-01-05T14:30:00+00:00",
                "2024-01-08T14:30:00+00:00",
            ],
            "dte": [14, 14, 14, 28],
            "expected_pnl": [-100.0, -50.0, 200.0, 500.0],
            "return_on_risk": [-0.5, -0.2, 1.0, 2.0],
            "large_loss_label": [1, 0, 0, 0],
            "stop_loss_hit": [1, 1, 0, 0],
            "strategy": ["PCS", "CCS", "PCS", "PCS"],
            "underlying": ["SMH", "SOXX", "XLF", "TLT"],
            "market_trend_regime": ["sideways", "sideways", "uptrend", "uptrend"],
            "market_volatility_regime": ["normal", "normal", "low", "low"],
        }
    )

    monkeypatch.setattr(target, "load_dataset", lambda path: dataset.copy())
    monkeypatch.setattr(target, "_resolve_ranker_artifact", lambda repo_root, registry: ranker_artifact)
    monkeypatch.setattr(target, "_resolve_strategy_ranker_paths", lambda repo_root, config, **kwargs: {})
    monkeypatch.setattr(
        target,
        "_score_rankers",
        lambda df, artifacts: df.assign(prediction=np.array([0.9, 0.8, 0.7], dtype=float)),
    )

    def fake_score_classifier(df, artifact_path: Path):
        if artifact_path == large_loss_artifact:
            return np.array([0.95, 0.10, 0.10], dtype=float)
        if artifact_path == stop_loss_artifact:
            return np.array([0.20, 0.80, 0.10], dtype=float)
        raise AssertionError(f"unexpected artifact {artifact_path}")

    monkeypatch.setattr(target, "_score_classifier", fake_score_classifier)

    seen = {}

    def fake_portfolio_risk_controls(df, score_column, **kwargs):
        seen["account_capital"] = kwargs["account_capital"]
        seen["rows_seen"] = len(df)
        portfolio_score = pd.to_numeric(df[score_column], errors="coerce")
        finite = np.isfinite(portfolio_score)
        diagnostics = pd.DataFrame(
            {
                "gate_stage": pd.Series(np.where(finite, "selected", "pick_selection"), index=df.index, dtype="string"),
                "gate_reason": pd.Series(pd.NA, index=df.index, dtype="string"),
                "gate_violation_codes": pd.Series(pd.NA, index=df.index, dtype="string"),
                "directional_reduced": pd.Series(False, index=df.index, dtype=bool),
                "portfolio_gamma_reduced": pd.Series(False, index=df.index, dtype=bool),
            }
        )
        return portfolio_score, diagnostics

    monkeypatch.setattr(target, "apply_portfolio_risk_controls", fake_portfolio_risk_controls)

    report = target.run_backtest(
        dataset_path=tmp_path / "dataset",
        year=2024,
        config_path=config_path,
        registry_path=registry_path,
    )

    assert report["stop_loss_artifact"] == str(stop_loss_artifact)
    assert report["runtime"]["stop_loss_veto_threshold"] == 0.30
    assert report["runtime"]["min_dte"] == 7
    assert report["runtime"]["max_dte"] == 21
    assert report["runtime"]["max_capital_per_period"] == 50000.0
    assert report["gate_diagnostics"]["gate_stage_counts"] == {
        "large_loss_veto": 1,
        "stop_loss_veto": 1,
        "selected": 1,
    }
    assert seen["account_capital"] == 50000.0
    assert seen["rows_seen"] == 3
    assert report["overall"]["selected"]["rows"] == 1
    assert report["overall"]["selected"]["total_pnl"] == 200.0
    assert report["overall"]["post_large_loss_gate"]["rows"] == 1
