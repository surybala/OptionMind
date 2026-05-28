# ML Trade Pipeline

Last updated: 2026-05-27

This is the canonical offline evaluation funnel for ML-selected credit-spread trades. Training runs should be judged by this pipeline because it mirrors the intended live selection path.

## Selection Funnel

1. Score candidate trades with the XGBoost ranker.
2. Apply the large-loss classifier and eliminate candidates whose predicted tail-loss probability is above the configured cap.
3. Apply portfolio gamma stress caps through the shared portfolio risk controls.
4. Evaluate only trades that pass all gates.

The evaluator records this as `trade_pipeline_selection` in the JSON report. That is the final tradeable-selection metric block to compare across experiments.

## Current Champion: V006b

All three artifacts trained on the balanced 500K-row dataset (`v006_balanced_cap12_500k`):

| Artifact | File |
|----------|------|
| Ranker (champion) | `artifacts/models/xgboost_v006b_500r_dp25.json` |
| Large-loss classifier | `artifacts/models/large_loss_classifier_v006b.json` |
| Stop-loss classifier | `artifacts/models/stop_loss_classifier_v006b.json` |

Champion ranker holdout (top-10% selection, 12,636 trades):
- PF: 1.96 | Win rate: 75.6% | Mean PnL: $56 | Mean RoR: 15.2%
- p05 PnL: -$314 | p01 PnL: -$935 | Worst: -$1,710
- Large-loss rate: 4.7% | Stop-loss rate: 11.5%
- SMH share: 21.5% | Top-5 share: 51.2%

## Training Commands

### 1. Train the ranker

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/xgboost_<version>.json \
  --target return_on_risk \
  --target-scale 0.10 --target-clip 5.0 \
  --num-boost-round 500 \
  --val-fraction 0.0 --early-stopping-rounds 0 \
  --downside-penalty 2.5 \
  --embargo-days 30
```

### 2. Train classifiers (same dataset)

```bash
# Large-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/large_loss_classifier_<version>.json \
  --target large_loss_label --embargo-days 30 --num-boost-round 300

# Stop-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/stop_loss_classifier_<version>.json \
  --target stop_loss_hit --embargo-days 30 --num-boost-round 300
```

## Evaluation Commands

Always run in this order: classifiers first, then ranker quality gate, then combined pipeline evaluation.

### Step 1 — Ranker quality gate (pure model signal, no portfolio controls)

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_exit_criteria \
  --input    artifacts/datasets/candidate_rows/dataset_version=<dataset_version> \
  --artifact artifacts/models/xgboost_<version>.json \
  --json-output artifacts/models/xgboost_<version>_exit_criteria.json
```

### Step 2 — Combined classifier + ranker evaluation

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input              artifacts/datasets/candidate_rows/dataset_version=<dataset_version> \
  --ranker-artifact    artifacts/models/xgboost_<version>.json \
  --large-loss-artifact artifacts/models/large_loss_classifier_<version>.json \
  --json-output        artifacts/models/risk_adjusted_<version>_eval.json
```

Step 2 defaults to hard-filter mode: candidates with `p(large_loss) > 0.70` are vetoed outright.
Portfolio controls are available via `--portfolio-risk-controls` for execution-layer simulation but
are NOT part of training or exit-criteria evaluation — they collapse holdout rows to ~600 and destroy
statistical power.

**Note:** `evaluate_risk_adjusted_ranking` does not accept a `--stop-loss-artifact` argument. The
stop-loss classifier is a separate training artifact used for diagnostics only.

## Report Fields

Use these fields when deciding whether a model is improving:

```text
raw_selection                 # XGBoost top-selection before gates
large_loss_gate_selection     # after large-loss classifier threshold
trade_pipeline_selection      # after large-loss gate and portfolio gamma stress caps
trade_pipeline_deltas         # change from raw_selection to final pipeline
trade_pipeline_eligible_rows  # number of rows that passed all gates
```

Older fields such as `risk_adjusted_selection` and `portfolio_risk_selection` remain for comparison, but they are not the canonical promotion metric unless explicitly chosen for an experiment.

## Promotion Rule

Do not promote a model based only on ranker metrics. A candidate must:

1. Pass all hard gates in `evaluate_exit_criteria` (ranker-only).
2. Have acceptable `trade_pipeline_selection` metrics in `evaluate_risk_adjusted_ranking`.
3. Pass a paper/shadow gate (≥30 trading days or 200 selected trades, profit factor ≥ 1.25).

Register the model artifact with `ml.models.registry.register_model_artifact`, then call
`ml.models.registry.promote_model` to make it the champion. The live scanner loads the champion
automatically from `artifacts/model_registry.json`.
