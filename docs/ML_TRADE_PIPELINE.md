# ML Trade Pipeline

Last updated: 2026-06-15

This is the canonical offline evaluation funnel for ML-selected credit-spread trades. Training runs should be judged by this pipeline because it mirrors the intended live selection path.

## Selection Funnel

1. Score candidate trades with the XGBoost ranker.
2. Apply the large-loss classifier and eliminate candidates whose predicted tail-loss probability is above the configured cap.
3. Apply the stop-loss classifier and eliminate candidates whose predicted stop-loss probability is above the configured cap.
4. Apply portfolio and scanner controls through the shared portfolio risk controls when the goal is live-faithful execution evaluation.
5. Evaluate only trades that pass all gates.

The evaluator records this as `trade_pipeline_selection` in the JSON report. That is the final tradeable-selection metric block to compare across experiments.

All train/test and walk-forward splits are now timestamp-based. `embargo_days` is enforced in calendar time for entry models and by grouped entry timestamp for intraday risk models, so the holdout gap reflects the actual forward horizon instead of an approximate row-count proxy.

## Current Live Entry Stack

All three entry artifacts are trained on the balanced 500K-row DTE<=21 dataset (`v006_balanced_cap12_500k_dte21`):

| Artifact | File |
|----------|------|
| Ranker (champion) | `artifacts/models/xgboost_v007b_dte21_quant.json` |
| Large-loss classifier | `artifacts/models/large_loss_classifier_v008.json` |
| Stop-loss classifier | `artifacts/models/stop_loss_classifier_v008.json` |

Current live metrics from the registries:
- Ranker `xgboost_v007b_dte21_quant`: holdout RoR `0.4717`, PF `1.733`, win rate `68.9%`, mean PnL `$46.39`, walk-forward PF min `1.874`, walk-forward PF avg `2.604`
- Large-loss classifier `large_loss_classifier_v008`: holdout AUC `0.843878`, recall `0.910883`, precision `0.38261`, walk-forward AUC `0.852065`
- Stop-loss classifier `stop_loss_classifier_v008`: holdout AUC `0.800183`, recall `0.995387`, precision `0.365765`, walk-forward AUC `0.822999`

Current live thresholds from `config.json`:
- `ml_scanner.large_loss_veto_threshold = 0.60`
- `ml_scanner.stop_loss_veto_threshold = 0.30`

The entry evaluator in this document covers the new-trade funnel only. Open-position exits are handled by the separate `intraday_risk_monitor_stop30m_v004` model in `artifacts/risk_model_registry.json`, currently configured at `threshold = 0.08` with `confirmations_required = 2`.

## Training Commands

### 1. Train the ranker

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21 \
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
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21 \
  --output artifacts/models/large_loss_classifier_<version>.json \
  --target large_loss_label --embargo-days 30 --num-boost-round 300

# Stop-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21 \
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
  --stop-loss-artifact artifacts/models/stop_loss_classifier_<version>.json \
  --runtime-config     config.json \
  --apply-portfolio-risk-controls \
  --json-output        artifacts/models/risk_adjusted_<version>_eval.json
```

Step 2 can now run in either mode:

- pure ML gates only: ranker -> large-loss gate -> stop-loss gate
- live-faithful trade pipeline: ranker -> large-loss gate -> stop-loss gate -> portfolio controls

Use the live-faithful mode when deciding whether a challenger is good enough to promote. Use the pure ML-gate mode when isolating model signal from portfolio allocation effects.

## Report Fields

Use these fields when deciding whether a model is improving:

```text
raw_selection                 # XGBoost top-selection before gates
large_loss_gate_selection     # after large-loss classifier threshold
stop_loss_gate_selection      # after stop-loss classifier threshold
trade_pipeline_selection      # after both ML gates and optional portfolio controls
trade_pipeline_deltas         # change from raw_selection to final pipeline
trade_pipeline_eligible_rows  # number of rows that passed all gates
trade_pipeline                # stage order, applied thresholds, account capital
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

## Provenance Expectations

Every promoted artifact should now carry enough provenance to reconstruct the run:

- dataset root path, manifest path, manifest hash, and dataset metadata
- exact training command
- applied data-quality filters and filter drop counts
- split summary including requested and actual embargo gap
- data fingerprint derived from dataset identity plus split metadata

If an artifact is missing that metadata, treat it as incomplete for promotion purposes.
