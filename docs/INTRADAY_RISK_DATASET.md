# Intraday Risk Dataset

Last updated: 2026-06-01

This dataset is the first training corpus for the live risk-monitoring model.
It is intentionally separate from `candidate_rows`:

- `candidate_rows` trains entry selection.
- `intraday_risk_rows` trains exit hazard and volatility-spike detection.

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

1. Build a minute-seeded candidate corpus from Massive:

```bash
.venv/bin/python -m ml.datasets.build_candidate_dataset \
  --provider massive \
  --underlying-preset broad-etfs \
  --entry-start 2022-06-01 \
  --entry-end 2026-05-31 \
  --strategy-family credit-spreads \
  --strategy-types PCS,CCS \
  --spread-widths 5,10,15 \
  --stock-provider yfinance \
  --stock-timeframe 1Day \
  --option-timeframe 1Min \
  --sample-every-n-bars 15 \
  --max-contracts 300 \
  --dataset-version candidate_rows_massive_intraday_seed_v001
```

2. Build the intraday risk dataset from that seed corpus:

```bash
.venv/bin/python -m ml.datasets.build_intraday_risk_dataset \
  --input artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_intraday_seed_v001 \
  --provider massive \
  --sample-every-n-candidates 5 \
  --sample-every-n-minutes 5 \
  --dataset-version intraday_risk_rows_v001
```

3. Append additional entry-date batches until the four-year backfill is complete.

If REST minute bars are sparse for historical options on your plan, use the
flat-file path described in [FLATFILES_INGESTION.md](/Users/surya/IdeaProjects/OptionMind/docs/FLATFILES_INGESTION.md)
and feed the resulting parquet minute bars back through
`ml.providers.parquet_minute.ParquetMinuteBarProvider`.

## Notes

- Massive aggregate requests for minute bars are chunked automatically so
  multi-month and multi-year pulls do not truncate at the 50k base-aggregate cap.
- The current builder uses minute bars only. Raw option-trade enrichment can be
  layered in later once the first hazard model is trained.
