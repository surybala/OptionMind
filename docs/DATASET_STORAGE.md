# Dataset Storage

Last updated: 2026-05-24

OptionMind writes generated ML datasets as partitioned Parquet plus a manifest.

## Storage Utilities

```text
ml/storage/
  dataset_writer.py
  manifest.py
  partitions.py
```

The main writer is `ParquetDatasetWriter`.

## Layout

Candidate rows are written under:

```text
artifacts/datasets/candidate_rows/
  dataset_version=<version>/
    _manifest.json
    source=<provider>/
      underlying=<symbol>/
        entry_date=<YYYY-MM-DD>/
          part-00000.parquet
```

If a dataset build returns zero rows, the writer still emits:

```text
part-00000.parquet
_manifest.json
```

The empty parquet file preserves schema for downstream tooling.

## Manifest

The manifest records:

- dataset version
- dataset type
- created timestamp
- row count
- root path
- file format
- partition columns
- parquet files
- dataset metadata such as provider, symbols, feature version, and label version

## Dependency

Parquet writing requires `pyarrow` or `fastparquet`. OptionMind uses `pyarrow` in `requirements.txt`.

## Current CLI

```bash
.venv/bin/python -m ml.datasets.build_candidate_dataset \
  --provider alpaca \
  --underlyings SPY \
  --entry-start 2025-05-14 \
  --entry-end 2025-05-15 \
  --contract-status inactive \
  --dataset-version candidate_rows_v001 \
  --output-dir artifacts/datasets
```

Generated artifacts live under `artifacts/`, which is ignored by git.
