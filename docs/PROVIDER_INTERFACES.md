# OptionMind Provider Interfaces

Last updated: 2026-05-24

The ML pipeline should not depend directly on Alpaca SDK objects or any single vendor. It should consume normalized provider contracts from `ml/providers`.

## Provider Packages

- `ml.providers.models`: vendor-neutral data shapes.
- `ml.providers.protocols`: protocols used by dataset builders and feature generators.
- `ml.providers.alpaca`: Alpaca adapter that maps Alpaca SDK responses into normalized shapes.
- `ml.providers.massive`: Massive/Polygon adapter for historical stock bars, options contracts, option bars, and option trades.

## Current Protocols

- `MarketDataProvider`: underlying stock/ETF bars.
- `OptionContractProvider`: option contract metadata.
- `OptionChainProvider`: current option-chain snapshots.
- `OptionPriceProvider`: historical option bars and trades.
- `VolatilityDataProvider`: VIX or volatility-derived series.
- `EventDataProvider`: earnings, macro, dividends, and event records.

## Design Rule

Dataset generation, feature engineering, labeling, training, and inference should depend on protocols and normalized models. Provider-specific SDKs should stay inside provider adapters.

This lets us:

- Use Alpaca for live trading and current inference.
- Add Cboe, OptionMetrics, OPRA-grade, or local parquet providers for historical training.
- Reuse feature and label code across providers.
- Compare provider quality without changing the model pipeline.

## Massive / Polygon

Massive is the current Polygon.io API brand. The adapter accepts either:

```text
MASSIVE_API_KEY
POLYGON_API_KEY
```

Use it with the dataset CLI:

```bash
.venv/bin/python -m ml.datasets.build_candidate_dataset \
  --provider massive \
  --underlyings SPY \
  --entry-start 2025-05-14 \
  --entry-end 2025-05-15 \
  --contract-status inactive \
  --max-contracts 10 \
  --dataset-version candidate_rows_massive_v001 \
  --output-dir artifacts/datasets
```

The provider normalizes Massive option tickers by stripping the API `O:` prefix
inside OptionMind rows, while adding it back for Massive API calls.

## Next Integration Point

The next implementation should build a small dataset prototype that accepts these protocols and creates candidate rows from provider data, starting with Alpaca for recent/replayable samples.
