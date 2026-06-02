# Massive Flat-File Ingestion

Last updated: 2026-06-01

This is the bulk-data path for the live risk-monitoring roadmap.

Massive flat files are exposed through an S3-compatible endpoint, not the REST
API. According to Massive's flat-file quickstart, the standard configuration is:

- Endpoint: `https://files.massive.com`
- Bucket: `flatfiles`
- Credentials: dedicated S3 access key + secret from the Massive dashboard

Official docs:

- [Flat Files Quickstart](https://massive.com/docs/flat-files/quickstart)
- [Options Flat Files Overview](https://polygon.io/docs/flat-files/options/overview)

## Environment

Set these env vars:

```text
MASSIVE_S3_ACCESS_KEY
MASSIVE_S3_SECRET_KEY
```

Legacy aliases also work:

```text
POLYGON_S3_ACCESS_KEY
POLYGON_S3_SECRET_KEY
```

## Implemented Pieces

- `ml.providers.massive_flatfiles.MassiveFlatFilesClient`
  Downloads and caches daily flat files from Massive's S3-compatible endpoint.

- `ml.datasets.ingest_massive_flatfiles`
  Filters daily CSV.gz files by ticker and writes partitioned parquet datasets.

- `ml.providers.parquet_minute.ParquetMinuteBarProvider`
  Reads the ingested parquet minute bars back as a local provider for training.

## Options Minute Aggregates

The options minute aggregate dataset root is:

```text
us_options_opra/minute_aggs_v1
```

Daily files are laid out as:

```text
us_options_opra/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
```

The stock minute aggregate dataset uses:

```text
us_stocks_sip/minute_aggs_v1
```

The CSV schema follows the standard Massive minute aggregate structure described
in the quickstart sample:

```text
ticker,volume,open,close,high,low,window_start,transactions
```

`window_start` is a UTC Unix timestamp in nanoseconds.

## Example

Ingest option minute bars for a focused ticker list:

```bash
.venv/bin/python -m ml.datasets.ingest_massive_flatfiles \
  --asset-class options \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --tickers SPY240216P00490000,SPY240216P00495000 \
  --dataset-version massive_options_minute_jan2024_v001
```

Ingest stock minute bars for the underlyings:

```bash
.venv/bin/python -m ml.datasets.ingest_massive_flatfiles \
  --asset-class stocks \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --tickers SPY,QQQ,IWM \
  --dataset-version massive_stocks_minute_jan2024_v001
```

## Next Step

The intended training flow is:

1. Seed candidate spreads from `candidate_rows`.
2. Build ticker lists for short legs, long legs, and underlyings.
3. Ingest flat-file minute aggregates for only those symbols and dates.
4. Point `ParquetMinuteBarProvider` at the ingested parquet roots.
5. Build `intraday_risk_rows` locally without hitting the REST API.
