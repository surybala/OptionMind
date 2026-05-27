# ML Trade Pipeline

Last updated: 2026-05-27

This is the canonical offline evaluation funnel for ML-selected credit-spread trades. Training runs should be judged by this pipeline because it mirrors the intended live selection path.

## Selection Funnel

1. Score candidate trades with the XGBoost ranker.
2. Apply the large-loss classifier and eliminate candidates whose predicted tail-loss probability is above the configured cap.
3. Apply portfolio gamma stress caps through the shared portfolio risk controls.
4. Evaluate only trades that pass all gates.

The evaluator records this as `trade_pipeline_selection` in the JSON report. That is the final tradeable-selection metric block to compare across experiments.

## Current Command

```bash
.venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input artifacts/datasets/candidate_rows/dataset_version=<dataset_version> \
  --ranker-artifact artifacts/models/<ranker>.json \
  --large-loss-artifact artifacts/models/<large_loss_classifier>.json \
  --max-large-loss-probability 0.70 \
  --portfolio-risk-controls \
  --scanner-config-path config.json \
  --json-output artifacts/reports/<run>_trade_pipeline_eval.json
```

The stop-loss classifier can still be supplied for diagnostics and legacy risk-adjusted comparisons, but the canonical trade pipeline gate is the large-loss classifier plus portfolio gamma stress caps.

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

Do not promote a model based only on ranker metrics. A candidate must have acceptable `trade_pipeline_selection` metrics and then pass the paper/shadow gate before live rollout.

