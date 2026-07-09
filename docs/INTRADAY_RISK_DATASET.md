# Intraday Risk Dataset

Last updated: 2026-06-15

This dataset is the first training corpus for the live risk-monitoring model.
It is intentionally separate from `candidate_rows`:

- `candidate_rows` trains entry selection.
- `intraday_risk_rows` trains exit hazard and volatility-spike detection.

Current live champion: `intraday_risk_monitor_stop30m_v004` in `artifacts/risk_model_registry.json`
with `operating_threshold = 0.08`, `confirmations_required = 2`, `min_age_minutes = 10` in `config.json`, and target
`stop_loss_hit_30m`.

## Design

The builder seeds from an existing spread candidate dataset and then pulls
minute bars from Massive/Polygon for:

- the underlying
- the short option leg
- the long option leg

For each spread candidate it rebuilds the spread path with minute bars, applies
the same credit-spread exit rules used elsewhere in the repo, and emits one row
per sampled minute-state up to the intraday exit.

Each row includes:

- current spread debit, P&L, profit captured, stop distance
- trailing underlying returns and realized vol
- future 5m / 15m / 30m worst-debit and best-debit windows
- horizon labels for stop-loss and profit-take hits
- the recomputed intraday exit timestamp and exit reason

## Why seed from candidate rows

Pulling every option contract for four years of minute history would explode
storage and still include mostly untradeable junk. Seeding from `candidate_rows`
keeps the corpus anchored to the spreads the strategy would actually consider.

## Build flow

1. Build a minute-seeded candidate corpus from the downloaded parquet minute bars:

```bash
.venv/bin/python -m ml.datasets.build_candidate_dataset \
  --provider parquet \
  --stock-provider parquet \
  --option-price-provider same \
  --stock-dataset-root artifacts/datasets/massive_flatfiles/dataset_version=massive_stocks_minute_broad_etfs_20220602_20260601_v001 \
  --option-dataset-root artifacts/datasets/massive_flatfiles/dataset_version=massive_options_minute_broad_etfs_20220602_20260601_v001 \
  --underlying-preset broad-etfs \
  --entry-start 2022-06-02 \
  --entry-end 2026-06-01 \
  --contract-status all \
  --strategy-family credit-spreads \
  --strategy-types PCS,CCS \
  --spread-widths 5,10 \
  --stock-timeframe 1Min \
  --option-timeframe 1Min \
  --sample-every-n-bars 60 \
  --max-contracts 300 \
  --build-window-days 5 \
  --stock-lookback-days 3 \
  --event-provider none \
  --economic-calendar none \
  --dividend-provider none \
  --volatility-provider none \
  --dataset-version candidate_rows_parquet_intraday_seed_v001
```

2. Build the intraday risk dataset from that seed corpus:

```bash
.venv/bin/python -m ml.datasets.build_intraday_risk_dataset \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_parquet_intraday_seed_v001 \
  --provider parquet \
  --stock-dataset-root artifacts/datasets/massive_flatfiles/dataset_version=massive_stocks_minute_broad_etfs_20220602_20260601_v001 \
  --option-dataset-root artifacts/datasets/massive_flatfiles/dataset_version=massive_options_minute_broad_etfs_20220602_20260601_v001 \
  --sample-every-n-candidates 5 \
  --sample-every-n-minutes 15 \
  --dataset-version intraday_risk_rows_parquet_v001
```

3. Append additional entry-date batches until the four-year backfill is complete.

## Balanced sampling flow

If the goal is a compact training corpus instead of a full brute-force backfill,
sample representative candidate months across regimes, then balance the raw
intraday rows down to the final target size.

1. Build a candidate seed pool from a handful of representative regime windows.
2. Balance the candidate pool:

```bash
.venv/bin/python -m ml.datasets.balance_candidate_dataset \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_parquet_intraday_seed_v001 \
  --dataset-version candidate_rows_parquet_intraday_seed_balanced_v001 \
  --target-rows 6500 \
  --group-columns underlying,market_volatility_regime,market_trend_regime \
  --max-underlying-share 0.08
```

3. Expand only that balanced seed set into minute-level risk rows, then trim the
   result to an exact training size:

```bash
.venv/bin/python -m ml.datasets.balance_intraday_risk_dataset \
  --input artifacts/datasets/intraday_risk_rows/dataset_version=intraday_risk_rows_parquet_v001 \
  --dataset-version intraday_risk_rows_parquet_balanced_1m_v001 \
  --target-rows 1000000 \
  --max-underlying-share 0.08
```

The `intraday_risk_rows` balancer uses the propagated
`market_trend_regime` / `market_volatility_regime` seed labels plus
`intraday_exit_reason` to keep the final corpus broad without oversampling a
single ETF or a single market state.

## Model training

The first live early-exit model is a grouped XGBoost classifier trained on
`intraday_risk_rows` with trade-level chronological splits. All minute rows
from the same spread entry stay in the same fold.

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.train_intraday_risk_monitor \
  --input artifacts/datasets/intraday_risk_rows/dataset_version=intraday_risk_rows_parquet_balanced_1m_v001 \
  --output artifacts/models/intraday_risk_monitor_v001.json \
  --model-output artifacts/models/intraday_risk_monitor_v001.xgboost.json \
  --target stop_loss_hit_30m \
  --walk-forward-folds 4 \
  --min-walk-forward-train-groups 1000 \
  --embargo-days 1
```

Use Optuna on the grouped walk-forward objective before locking the final
artifact:

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_intraday_risk_monitor \
  --input artifacts/datasets/intraday_risk_rows/dataset_version=intraday_risk_rows_parquet_balanced_1m_v001 \
  --target stop_loss_hit_30m \
  --study-name intraday_risk_monitor_hp_v001 \
  --n-trials 40
```

For a more standalone realtime risk model, run a second Optuna pass on the
full raw intraday-risk dataset to search richer minute-state feature groups
while penalizing over-eager close policies:

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_intraday_risk_monitor_features \
  --input artifacts/datasets/intraday_risk_rows/dataset_version=intraday_risk_rows_parquet_regime_seed_raw_broad_etfs_20220602_20260601_v001 \
  --target stop_loss_hit_30m \
  --study-name intraday_risk_monitor_feat_v001 \
  --n-trials 24
```

That writes the recommended retrain command to
`artifacts/optuna/intraday_risk_monitor_feat_v001_best_features.json`.

After the feature set is fixed, run a second Optuna pass on that winner to
retune hyperparameters on the same raw dataset:

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.optimize_intraday_risk_monitor \
  --input artifacts/datasets/intraday_risk_rows/dataset_version=intraday_risk_rows_parquet_regime_seed_raw_broad_etfs_20220602_20260601_v001 \
  --target stop_loss_hit_30m \
  --study-name intraday_risk_monitor_hp_fullraw_v002 \
  --n-trials 32 \
  --feature-config artifacts/optuna/intraday_risk_monitor_feat_fullraw_v001_best_features.json \
  --max-threshold-close-rate 0.12 \
  --max-threshold-false-close-rate 0.10 \
  --output-artifact artifacts/models/intraday_risk_monitor_stop30m_v004.json \
  --model-output artifacts/models/intraday_risk_monitor_stop30m_v004.xgboost.json
```

For spread seeding, keep `--max-contracts` high enough to preserve adjacent
strikes within the same expiry. Very small caps can leave only one strike per
expiry/type bucket, which eliminates valid `PCS` / `CCS` width pairs.

## Notes

- Massive aggregate requests for minute bars are chunked automatically so
  multi-month and multi-year pulls do not truncate at the 50k base-aggregate cap.
- The current builder uses minute bars only. Raw option-trade enrichment can be
  layered in later once the first hazard model is trained.
