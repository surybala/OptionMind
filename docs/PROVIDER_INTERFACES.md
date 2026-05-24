# OptionMind Provider Interfaces

Last updated: 2026-05-24

The ML pipeline should not depend directly on Alpaca SDK objects or any single vendor. It should consume normalized provider contracts from `ml/providers`.

## Provider Packages

- `ml.providers.models`: vendor-neutral data shapes.
- `ml.providers.protocols`: protocols used by dataset builders and feature generators.
- `ml.providers.alpaca`: Alpaca adapter that maps Alpaca SDK responses into normalized shapes.

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

## Next Integration Point

The next implementation should build a small dataset prototype that accepts these protocols and creates candidate rows from provider data, starting with Alpaca for recent/replayable samples.
