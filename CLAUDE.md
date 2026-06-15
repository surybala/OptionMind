# OptionMind — Claude Development Guide

## ML Training Workflow

### Primary model: XGBoost ranker

The canonical training target is **return on risk** (`return_on_risk = expected_pnl / max_loss`).
This normalises dollar PnL by capital at risk so the model ranks capital efficiency, not raw dollar size.
Do not switch back to `expected_pnl` without explicit instruction.

Training parameters that must hold by default:
- `--target return_on_risk`
- `--target-scale 0.10` (RoR values live in the 0–1 range)
- `--target-clip 5.0`
- `--num-boost-round 500` (fixed rounds, no early stopping)
- `--val-fraction 0.0 --early-stopping-rounds 0`
- `--downside-penalty 2.5` (heavier tail-loss punishment)
- `--embargo-days 30`

Canonical training command:
```
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/xgboost_<version>.json \
  --target return_on_risk \
  --target-scale 0.10 --target-clip 5.0 \
  --num-boost-round 500 \
  --val-fraction 0.0 --early-stopping-rounds 0 \
  --downside-penalty 2.5 \
  --embargo-days 30
```

### Risk classifiers (train before ranker evaluation)

Always retrain classifiers on the same dataset version as the ranker.
Both use the same script — only `--target` differs.

```
# Large-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/large_loss_classifier_<version>.json \
  --target large_loss_label --embargo-days 30 --num-boost-round 300

# Stop-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/stop_loss_classifier_<version>.json \
  --target stop_loss_hit --embargo-days 30 --num-boost-round 300
```

### Evaluation workflow (always run in this order)

**Step 1 — Ranker quality gate** (pure model signal, no portfolio controls):
```
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_exit_criteria \
  --input    artifacts/datasets/candidate_rows/<dataset_version> \
  --artifact artifacts/models/xgboost_<version>.json \
  --json-output artifacts/models/xgboost_<version>_exit_criteria.json
```

**Step 2 — Combined classifier + ranker evaluation** (classifiers as hard pre-filter):
```
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input              artifacts/datasets/candidate_rows/<dataset_version> \
  --ranker-artifact    artifacts/models/xgboost_<version>.json \
  --large-loss-artifact artifacts/models/large_loss_classifier_<version>.json \
  --json-output        artifacts/models/risk_adjusted_<version>_eval.json
```

Note: `--stop-loss-artifact` is not a recognised argument for `evaluate_risk_adjusted_ranking`; omit it.

Step 2 defaults to hard-filter mode: classifiers with the live runtime thresholds (`p(large_loss) > 0.60`, `p(stop_loss_hit) > 0.30`) veto a trade outright;
no soft penalty is applied. Portfolio controls are available via `--apply-portfolio-risk-controls`
for final execution-layer simulation but are NOT part of training or exit-criteria evaluation.

## Architecture Decisions

### Training vs execution layers are separate

**Training** (`train_xgboost`, `evaluate_exit_criteria`):  
- Operates on the full holdout (~360K rows)
- Measures pure model signal quality
- No portfolio risk controls — they would collapse holdout to ~600 rows and destroy statistical power
- Exit criteria gates are calibrated in dollar PnL terms

**Execution** (`evaluate_risk_adjusted_ranking`, live agent):  
- Classifiers hard-filter high-risk trades (`p(large_loss) > 0.60`, `p(stop_loss_hit) > 0.30`)
- Ranker scores the surviving candidates by predicted RoR
- Portfolio controls (gamma stress, concentration limits) applied last
- `portfolio_risk.py` is the single canonical implementation shared by both layers

### Dollar PnL metrics are always in dollars

`_credit_spread_selection_metrics` in `train_xgboost.py` always reads `expected_pnl` from the
DataFrame to report top-decile dollar metrics (`top_decile_actual_mean`, `top_decile_profit_factor`,
etc.) regardless of training target. This keeps all exit criteria thresholds calibrated in dollars
even when training on `return_on_risk`.

### Golden datasets

Base dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched`  
1.44M rows, 39 ETF underlyings, `features_v005`, `credit_spread_labels_v002`.

Current live entry-training dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21`  
500K rows, DTE<=21, derived via sqrt-frequency hierarchical sampling (max 12% per underlying).  
Located at: `artifacts/datasets/candidate_rows/dataset_version=<name>`

## Current live model artifacts

The live entry stack is loaded from `artifacts/model_registry.json`. The live open-position exit model is loaded from `artifacts/risk_model_registry.json`.

| Layer | Artifact | File | Notes |
|------|----------|------|-------|
| Entry ranker (champion) | `xgboost_v007b_dte21_quant` | `artifacts/models/xgboost_v007b_dte21_quant.json` | Holdout RoR `0.4717`, PF `1.733`, WR `68.9%`, mean PnL `$46.39`, WF PF min `1.874`; 43 features including 5 quant-structural features |
| Entry veto 1 | `large_loss_classifier_v008` | `artifacts/models/large_loss_classifier_v008.json` | Threshold `0.60`; holdout AUC `0.843878`, recall `0.910883`, precision `0.38261`; 34 features |
| Entry veto 2 | `stop_loss_classifier_v008` | `artifacts/models/stop_loss_classifier_v008.json` | Threshold `0.30`; holdout AUC `0.800183`, recall `0.995387`, precision `0.365765`; 41 features |
| Exit-risk monitor (champion) | `intraday_risk_monitor_stop30m_v004` | `artifacts/models/intraday_risk_monitor_stop30m_v004.json` | Threshold `0.08`; holdout AUC `0.930315`, recall `0.691576`, close rate `0.063957`, false-close rate `0.061815`; 45 realtime features |

### Live runtime parameters

- Entry ranker champion: `xgboost_v007b_dte21_quant`
- Entry rollback ranker: `xgboost_v006c_500r_dp28`
- `ml_scanner.large_loss_veto_threshold = 0.60`
- `ml_scanner.stop_loss_veto_threshold = 0.30`
- `risk_parameters.ml_exit_risk.threshold = 0.08`
- `risk_parameters.ml_exit_risk.confirmations_required = 2`
- `risk_parameters.ml_exit_risk.min_age_minutes = 10`
- Dashboard open-position `risk_level` is now derived purely from ML exit-risk score proximity, not the old heuristic `critical/caution` labelling

### Entry stack notes

- The current ranker is a DTE<=21 regime-matched `return_on_risk` model tuned for the live scanner rather than a generic broad-DTE ranking benchmark.
- `large_loss_classifier_v008` is intentionally stricter than the previous champion family on threshold calibration and feature pruning; it keeps only the feature groups that held up best in Optuna search.
- `stop_loss_classifier_v008` remains a high-recall veto layer. It is designed to reject likely stop-loss candidates, not to maximize precision.

### Exit-risk stack notes

- The current open-position model predicts `stop_loss_hit_30m`, not eventual trade profitability.
- Runtime order in `src/position_monitor.py` is: ML exit-risk gate, then profit-take, then deterministic stop-loss fallback.
- A live close requires `2` consecutive ML confirmations above the `0.08` threshold.

### Historical references

- `docs/V006_MODEL_STITCHING.md` is now a historical promotion record for the older `v006`/`v006b` family.
- `docs/ML_TRADE_PIPELINE.md` is the canonical entry-funnel evaluation contract.
- `docs/INTRADAY_RISK_DATASET.md` documents the separate training flow for the open-position risk monitor.
