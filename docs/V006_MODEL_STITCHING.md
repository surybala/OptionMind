# V006 / V006b Model Analysis And Promotion Record

Last updated: 2026-05-27

This note stitches together the v006 dataset family, rankers, binary risk classifiers, and risk-adjusted evaluator. The general pipeline contract lives in `docs/ML_TRADE_PIPELINE.md`.

---

## V006b: Current Champion (Promoted 2026-05-27)

The v006b models are trained on the balanced 500K-row dataset derived from v006 via sqrt-frequency hierarchical sampling (max 12% per underlying). This resolves the SMH/SOXX concentration problem that blocked v006 promotion.

### Champion Artifact Map

Training dataset:

```text
artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k
```

Artifacts:

```text
artifacts/models/xgboost_v006b_500r_dp25.json         # RoR ranker — CHAMPION (registered in model_registry.json)
artifacts/models/large_loss_classifier_v006b.json      # p(large_loss_label)
artifacts/models/stop_loss_classifier_v006b.json       # p(stop_loss_hit)
```

Evaluation reports:

```text
artifacts/models/xgboost_v006b_500r_dp25_exit_criteria.json
artifacts/models/risk_adjusted_v006b_eval.json
```

### Champion Ranker Metrics (top-10% holdout, 12,636 trades)

```text
mean PnL:                       $56
profit factor:                  1.96
win rate:                       75.6%
mean return on risk:            15.2%
large-loss rate:                4.7%
stop-loss rate:                 11.5%
p05 / p01 / worst PnL:          -$314 / -$935 / -$1,710
SMH share:                      21.5%
top-5 underlying share:         51.2%
```

All exit criteria pass. SMH concentration drops from 37% (v006 RoR ranker) to 21.5% due to balanced training.

### Classifier Metrics

```text
large_loss_classifier_v006b:
  test AUC:       0.848
  recall@0.70:    98.2%

stop_loss_classifier_v006b:
  test AUC:       0.804
  recall@0.70:    99.7%
```

### V006b Training Commands

```bash
# Ranker
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/xgboost_v006b_500r_dp25.json \
  --target return_on_risk \
  --target-scale 0.10 --target-clip 5.0 \
  --num-boost-round 500 \
  --val-fraction 0.0 --early-stopping-rounds 0 \
  --downside-penalty 2.5 \
  --embargo-days 30

# Large-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/large_loss_classifier_v006b.json \
  --target large_loss_label --embargo-days 30 --num-boost-round 300

# Stop-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/stop_loss_classifier_v006b.json \
  --target stop_loss_hit --embargo-days 30 --num-boost-round 300
```

---

## V006: Rollback Reference (Not Promoted)

The v006 models remain as rollback artifacts but should not be promoted.

### V006 Artifact Map

Dataset:

```text
artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched
```

Models:

```text
artifacts/models/xgboost_v006_500r_dp25.json       # expected_pnl ranker
artifacts/models/xgboost_v006_ror.json             # return_on_risk ranker
artifacts/models/large_loss_classifier_v006.json   # p(large_loss_label)
artifacts/models/stop_loss_classifier_v006.json    # p(stop_loss_hit)
```

Evaluations:

```text
artifacts/models/xgboost_v006_500r_dp25_exit_criteria_v2.json
artifacts/models/xgboost_v006_ror_exit_criteria.json
artifacts/models/risk_adjusted_v006_eval.json
artifacts/models/risk_adjusted_v006_ror_eval.json
```

Ignore `artifacts/models/xgboost_v006_500r_dp25_exit_criteria.json` — it is an earlier small-sample report (615 holdout rows only). The `_v2` report re-scores the full 360K-row holdout.

### V006 Dataset Findings

v006 has 1,441,233 rows, 39 ETF underlyings. Key issue: SMH accounts for 322,527 rows (22.4%), teaching the model too much regime-specific signal. v006b fixes this via balanced sampling.

Key dataset rates:

```text
profit_label=1:       849,756 / 1,441,233 = 58.96%
large_loss_label=1:   170,112 / 1,441,233 = 11.80%
stop_loss_hit=1:      390,793 / 1,441,233 = 27.12%
PCS rows:             788,734
CCS rows:             652,499
```

### V006 RoR Ranker Run (Failed Gates)

Latest v006 ranker artifact: `xgboost_v006_ror.json` (trained 2026-05-27T03:59:16Z).

Ranker-only holdout selection:

```text
selected rows:                  36,031 / 360,308
mean PnL:                       $55.04
profit factor:                  1.59
win rate:                       74.07%
mean return on risk:            0.161
SMH share:                      37.33%
top-5 underlying share:         66.46%
p05 / p01 / worst PnL:          -$626 / -$1,188 / -$1,655
```

Failed gates: return-on-risk, concentration, tail percentiles, worst-trade, drawdown, feature-stability, train-vs-holdout PF. SMH at 37% drove most of the failures. This was the direct motivation for the balanced v006b dataset.

### V006 Expected-PnL Ranker Run (Failed Gates)

`xgboost_v006_500r_dp25.json` — good average holdout dollars but chases concentrated premium.

```text
mean PnL:                       $41.21
profit factor:                  1.97
win rate:                       82.68%
SMH share:                      49.71%
top-5 underlying share:         73.63%
p05 / p01 / worst PnL:          -$263.50 / -$896 / -$1,898
```

---

## Pipeline Flow (Canonical)

1. Build or select one candidate-row dataset.
2. Audit the dataset before training.
3. Train classifiers on the same dataset.
4. Train the primary ranker.
5. Evaluate the ranker alone with `evaluate_exit_criteria`.
6. Evaluate ranker + classifiers with `evaluate_risk_adjusted_ranking`.
7. Register the artifact and promote as champion via `ml.models.registry`.
8. Complete paper/shadow gate before live rollout.

The canonical offline selection metric is `trade_pipeline_selection`: XGBoost ranker score → large-loss classifier gate → portfolio gamma stress caps.
