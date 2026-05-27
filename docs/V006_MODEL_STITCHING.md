# v006 Model Stitching And Run Analysis

Last updated: 2026-05-27

This note stitches together the current v006 dataset, rankers, binary risk classifiers, and risk-adjusted evaluator. None of the current v006 artifacts should be promoted as a live champion yet. The general pipeline contract lives in `docs/ML_TRADE_PIPELINE.md`.

## Artifact Map

Dataset:

```text
artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched
```

Quality report:

```text
artifacts/reports/candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched_quality.json
```

Current v006 models:

```text
artifacts/models/xgboost_v006_500r_dp25.json       # expected_pnl ranker
artifacts/models/xgboost_v006_ror.json             # return_on_risk ranker, latest ranker run
artifacts/models/large_loss_classifier_v006.json   # p(large_loss_label)
artifacts/models/stop_loss_classifier_v006.json    # p(stop_loss_hit)
```

Current v006 evaluations:

```text
artifacts/models/xgboost_v006_500r_dp25_exit_criteria_v2.json
artifacts/models/xgboost_v006_ror_exit_criteria.json
artifacts/models/risk_adjusted_v006_eval.json
artifacts/models/risk_adjusted_v006_ror_eval.json
```

Ignore `artifacts/models/xgboost_v006_500r_dp25_exit_criteria.json` for decision-making. It is an earlier small-sample report with only 615 holdout rows. The `_v2` report re-scores the full 360,308-row chronological holdout.

## Intended Flow

1. Build or select one candidate-row dataset.
2. Audit the dataset before training.
3. Train one primary ranker.
4. Train binary risk classifiers on the same dataset and chronological split.
5. Evaluate the ranker alone with hard exit criteria.
6. Evaluate ranker plus the trade pipeline gates with `evaluate_risk_adjusted_ranking`.
7. Promote only a ranker artifact that passes offline gates and then a paper/shadow gate.

The canonical offline selection metric is now `trade_pipeline_selection`: XGBoost ranker score, then large-loss classifier gate, then portfolio gamma stress caps. The live scanner currently loads one champion ranker from `artifacts/model_registry.json`. A production risk-classifier bundle would need a separate registry/inference change before live inference fully matches the offline funnel.

## Commands

Dataset audit:

```bash
.venv/bin/python -m ml.datasets.audit_candidate_dataset \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --json-output artifacts/reports/candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched_quality.json
```

Primary expected-PnL ranker:

```bash
.venv/bin/python -m ml.models.train_xgboost \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --output artifacts/models/xgboost_v006_500r_dp25.json \
  --target expected_pnl \
  --target-scale 100.0 \
  --target-clip 5000.0 \
  --downside-penalty 2.5 \
  --num-boost-round 500 \
  --max-depth 3 \
  --eta 0.05 \
  --test-fraction 0.25 \
  --walk-forward-folds 3 \
  --embargo-days 30
```

Return-on-risk experiment:

```bash
.venv/bin/python -m ml.models.train_xgboost \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --output artifacts/models/xgboost_v006_ror.json \
  --target return_on_risk \
  --target-scale 0.10 \
  --target-clip 5.0 \
  --downside-penalty 2.5 \
  --num-boost-round 500 \
  --max-depth 3 \
  --eta 0.05 \
  --test-fraction 0.25 \
  --walk-forward-folds 3 \
  --embargo-days 30
```

Risk classifiers:

```bash
.venv/bin/python -m ml.models.train_large_loss_classifier \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --output artifacts/models/large_loss_classifier_v006.json \
  --target large_loss_label \
  --test-fraction 0.25 \
  --walk-forward-folds 3 \
  --embargo-days 30

.venv/bin/python -m ml.models.train_large_loss_classifier \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --output artifacts/models/stop_loss_classifier_v006.json \
  --target stop_loss_hit \
  --test-fraction 0.25 \
  --walk-forward-folds 3 \
  --embargo-days 30
```

Ranker-only gate:

```bash
.venv/bin/python -m ml.models.evaluate_exit_criteria \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --artifact artifacts/models/xgboost_v006_500r_dp25.json \
  --json-output artifacts/models/xgboost_v006_500r_dp25_exit_criteria_v2.json
```

Trade-pipeline evaluation for an expected-PnL ranker:

```bash
.venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --ranker-artifact artifacts/models/xgboost_v006_500r_dp25.json \
  --large-loss-artifact artifacts/models/large_loss_classifier_v006.json \
  --max-large-loss-probability 0.70 \
  --json-output artifacts/models/risk_adjusted_v006_eval.json
```

Trade-pipeline evaluation for a return-on-risk ranker:

```bash
.venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched \
  --ranker-artifact artifacts/models/xgboost_v006_ror.json \
  --large-loss-artifact artifacts/models/large_loss_classifier_v006.json \
  --risk-penalty-basis return_on_risk \
  --max-large-loss-probability 0.70 \
  --json-output artifacts/models/risk_adjusted_v006_ror_eval.json
```

## v006 Dataset Findings

v006 has 1,441,233 rows versus 500,000 in the prior v005 balanced view. It is broader and more realistic, but it is also more concentrated.

Key dataset rates:

```text
profit_label=1:       849,756 / 1,441,233 = 58.96%
large_loss_label=1:   170,112 / 1,441,233 = 11.80%
stop_loss_hit=1:      390,793 / 1,441,233 = 27.12%
PCS rows:             788,734
CCS rows:             652,499
```

The biggest concentration problem is SMH: 322,527 rows, about 22.4% of the whole dataset. v005 capped SMH at 60,000 rows out of 500,000, about 12.0%.

## Latest Run Analysis

The latest ranker artifact by timestamp is `xgboost_v006_ror.json`, created at `2026-05-27T03:59:16Z`. It trained on `return_on_risk`, not dollar PnL.

Ranker-only holdout selection:

```text
selected rows:                  36,031 / 360,308
mean PnL:                       $55.04
slippage-adjusted mean PnL:     $3.47
profit factor:                  1.59
slippage-adjusted PF:           1.03
win rate:                       74.07%
large-loss rate:                6.52%
stop-loss rate:                 16.66%
mean return on risk:            0.161
SMH share:                      37.33%
top-5 underlying share:         66.46%
p05 / p01 / worst PnL:          -$626 / -$1,188 / -$1,655
max drawdown:                   $675,878
```

The good news: the RoR ranker cuts tail-label rates compared with the expected-PnL v006 raw selection. The bad news: it still fails return-on-risk, concentration, tail percentile, worst-trade, drawdown, feature-stability, and train-vs-holdout PF gates.

`risk_adjusted_v006_ror_eval.json` did not actually adjust the RoR ranking: it used zero penalty multiples and no probability caps, so raw and risk-adjusted selections are identical. Portfolio controls then reduced the selection to only 17 rows, which is too small to treat as a robust validation sample.

## Expected-PnL v006 Comparison

`xgboost_v006_500r_dp25.json` performs well on average holdout dollars but chases concentrated premium.

Ranker-only holdout selection from `_exit_criteria_v2`:

```text
mean PnL:                       $41.21
slippage-adjusted mean PnL:     $13.38
profit factor:                  1.97
slippage-adjusted PF:           1.27
win rate:                       82.68%
mean return on risk:            0.626
large-loss rate:                5.79%
stop-loss rate:                 10.45%
SMH share:                      49.71%
top-5 underlying share:         73.63%
p05 / p01 / worst PnL:          -$263.50 / -$896 / -$1,898
max drawdown:                   $178,706
```

The risk-adjusted evaluator with dollar penalties deconcentrates the selected set, but it gives up too much edge:

```text
mean PnL:                       $19.79
slippage-adjusted mean PnL:     -$6.90
profit factor:                  2.01
slippage-adjusted PF:           0.80
win rate:                       67.57%
large-loss rate:                14.57%
stop-loss rate:                 19.72%
SMH share:                      10.99%
top-5 underlying share:         48.54%
```

That is close on tail concentration, but not tradeable after slippage.

## Classifier Read

The v006 classifiers are useful but not promotion-quality filters by themselves.

```text
large_loss_classifier_v006:
  test AUC:       0.841
  precision@0.15: 17.5%
  recall@0.15:    98.5%

stop_loss_classifier_v006:
  test AUC:       0.796
  precision@0.15: 31.4%
  recall@0.15:    99.6%
```

They are deliberately recall-heavy at the default 0.15 threshold. Use them as rank penalties or probability caps, then validate selected-trade economics. Do not treat `p > 0.15` as a standalone reject rule without threshold tuning.

## Recommendation

Do not promote any v006 model yet.

For the next run:

1. Use `expected_pnl` as the primary champion target and keep `return_on_risk` as a comparison target until the RoR evaluator and live scanner are fully calibrated.
2. Cap or balance per underlying before training. SMH and SOXX are teaching the model too much of the edge and too much of the tail.
3. Re-run the evaluator and compare `trade_pipeline_selection`. A `max_large_loss_probability=0.70` cap is a reasonable starting point.
4. Add a per-entry-date selection view before portfolio controls. Evaluating top 10% of all holdout rows overcounts many overlapping same-day variants.
5. Only register a candidate after exit criteria pass. Only promote after a separate paper/shadow run passes.
