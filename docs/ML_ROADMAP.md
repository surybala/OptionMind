# OptionMind ML Roadmap

Last updated: 2026-05-24

This roadmap turns the north-star spec into buildable milestones. The priority is to create a reliable data and validation foundation before training complex models.

## Phase 0: Data Source Audit

Goal: learn what data we can actually retrieve and where provider gaps exist.

Deliverables:

- Alpaca audit script for historical stock bars, option contracts, option bars/trades, current option chains, and opening-window data.
- JSON and Markdown audit reports.
- Gap list for data that Alpaca does not provide or does not provide historically.
- Candidate list of complementary providers.

Coach note: this is the first serious ML step. A model cannot learn signal that the data pipeline cannot observe consistently.

## Phase 1: Provider Interfaces

Goal: make the data layer provider-pluggable so Alpaca is a source, not a lock-in.

Interfaces:

- `MarketDataProvider`
- `OptionContractProvider`
- `OptionChainProvider`
- `OptionPriceProvider`
- `VolatilityDataProvider`
- `EventDataProvider`

Deliverables:

- Typed provider protocols.
- Alpaca provider implementation for available data.
- Stub/provider adapters for future external datasets.
- Data quality checks per provider.

## Phase 2: Historical Candidate Dataset

Goal: produce one row per historical option candidate with only decision-time features.

Dataset row families:

- Contract identity and metadata
- Option price/quote fields
- Greeks and IV where available or computed
- Underlying price action
- Opening behavior
- Market regime
- Event risk
- Execution quality
- Forward-simulated labels

Leakage rule: no feature may use data from after the candidate decision timestamp.

## Phase 3: Labeling Engine

Goal: turn historical price paths into P&L-centered targets.

Initial labels:

- `expected_pnl`
- `profit_label`
- `stop_loss_hit`
- `large_loss_label`
- `max_adverse_excursion`
- `max_favorable_excursion`
- `days_to_exit`
- `exit_reason`

The first labels should use standardized exit rules so the model learns against a stable definition.

## Phase 4: Baseline Model

Goal: train a simple, inspectable model before neural networks.

Recommended first models:

- LightGBM
- XGBoost
- CatBoost

Evaluation:

- Time-based split
- Walk-forward validation
- Calibration
- Top-decile candidate quality
- Expected P&L
- Profit factor
- Max drawdown
- Tail-loss behavior

## Phase 5: Model Registry And Champion Selection

Goal: create a repeatable model lifecycle.

Each model version records:

- Training data range
- Feature set version
- Label definition
- Model type and parameters
- Validation metrics
- Backtest metrics
- Promotion decision

Champion selection must be out-of-sample and risk-aware.

## Phase 6: Paper And Shadow Inference

Goal: score real market candidates without risking capital first.

Flow:

1. Generate market-universe candidates.
2. Score candidates with the champion model.
3. Log predictions and rejected candidates.
4. Run deterministic risk checks.
5. Paper trade or shadow trade.
6. Compare predicted outcomes with realized outcomes.

## Phase 7: Limited Live Deployment

Goal: carefully introduce live trades with deterministic controls still active.

Controls:

- Small allocation
- Strict buying-power checks
- Max daily loss
- Max position size
- Max concentration
- Stop-loss monitor
- Kill switch
- Rollback to previous champion

## Immediate Next Task

Build a first historical candidate dataset prototype on top of the provider protocols, starting with Alpaca for recent/replayable samples while keeping room for a richer historical-options provider.
