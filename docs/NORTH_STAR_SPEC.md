# OptionMind North Star Spec

Last updated: 2026-05-24

## Read This First

OptionMind is not meant to remain a deterministic scanner-based options bot.

The current codebase was inherited from OptionWheel and contains useful production plumbing: Alpaca integration, option execution, position monitoring, stop-loss handling, account-level risk gates, portfolio controls, tests, and a dashboard. Those pieces are valuable. The old scanner and its deterministic ranking rules are not the future intelligence layer.

New development should treat the deterministic scanner as legacy baseline behavior, not as the product direction.

## North Star

Build an ML-first options trading system that learns from historical options-market opportunities, predicts P&L-centered outcomes, ranks high-potential option trades, and improves through a closed feedback loop.

The desired end state is an intelligent option trader that:

- Builds candidates from market data, not from the old deterministic scan filters.
- Scores candidates using learned models trained on historical option, underlying, regime, event, and execution data.
- Optimizes for expected P&L, drawdown, loss probability, and risk-adjusted return.
- Keeps deterministic guardrails for account risk, stop loss, concentration, exposure, and emergency shutdown.
- Logs every prediction and realized outcome so future model versions can improve.
- Promotes models only after strict out-of-sample validation, backtesting, and paper/shadow trading.

## What This Is

OptionMind should become:

- A historical options data ingestion and feature engineering platform.
- A candidate generation system independent of legacy deterministic filters.
- A model training and evaluation lab.
- A champion/challenger model lifecycle system.
- A live inference engine for ranking option trade opportunities.
- A controlled execution system where deterministic risk controls remain active.
- A learning system that improves through repeated training, validation, and outcome logging.

## What This Is Not

OptionMind should not be treated as:

- Merely a renamed OptionWheel project.
- A fixed-rule scanner with a model bolted onto the end.
- A system where delta or probability-of-expiring-OTM is the primary signal.
- A strategy that optimizes for win rate while ignoring tail losses.
- A bot that lets an ML model bypass hard account and stop-loss controls.
- A training loop that chooses winners based on in-sample performance.

## Core Thesis

Greeks are necessary but insufficient.

Delta, gamma, theta, vega, rho, IV, and DTE describe option sensitivity and theoretical pricing behavior. They do not fully explain realized trade P&L. A real model must also understand:

- Underlying price behavior
- Gap and opening candle risk
- Volatility regime
- Market-wide risk-on/risk-off state
- Liquidity and slippage
- Earnings and macro event risk
- Recent realized volatility
- Sector and index movement
- Portfolio context

The model should predict trade economics, not just theoretical probability.

## First Learning Target

The first serious model should answer:

> Given an option trade candidate at decision time, what are the expected P&L, probability of profit, probability of stop loss, and probability of large loss?

Initial labels should include:

- `expected_pnl`
- `profit_label`
- `stop_loss_hit`
- `large_loss_label`
- `max_adverse_excursion`
- `max_favorable_excursion`
- `days_to_exit`
- `exit_reason`

The first model should likely use LightGBM, XGBoost, or CatBoost before neural networks. Neural networks come after the dataset, labels, validation, and leakage controls are solid.

## Candidate Generation Principle

Candidate generation should be market-universe based, not scanner-rule based.

For a historical timestamp:

1. Reconstruct available option contracts.
2. Generate candidate sell-side opportunities from the available chain.
3. Attach entry-time features only.
4. Simulate the forward outcome under standardized exit rules.
5. Store the labeled candidate row.

The old deterministic scanner may be used as a baseline comparator, but not as the source of truth for what the model is allowed to consider.

## Risk Control Principle

The model recommends. The risk engine governs.

ML may rank, reject, size-suggest, or prioritize candidates. Deterministic controls should still enforce:

- Max account exposure
- Max position size
- Max symbol/sector concentration
- Max daily loss
- Stop-loss behavior
- Buying-power checks
- Liquidity and tradability checks
- Kill switch behavior
- Execution reconciliation

No model confidence score should override these controls.

## Data Source Principle

Alpaca is a key source and execution broker, but not the only acceptable data provider.

Use Alpaca where it is strong:

- Trading API
- Paper/live execution
- Current option chains and snapshots
- Option contract metadata
- Historical option bars/trades when available
- Stock/ETF bars

Use additional providers for gaps:

- Historical full option chains
- Historical IV and Greeks
- Corporate actions and dividends
- Earnings calendars
- Macro event calendars
- VIX and volatility term structure
- OPRA-grade historical data
- Market breadth, sentiment, and regime datasets

The data layer should be provider-pluggable.

## Model Evolution Loop

The long-running training system should produce many candidate model versions, but the champion is selected only by rigorous validation.

Each model version must record:

- Training data range
- Feature set version
- Label definition
- Model type
- Hyperparameters
- Validation metrics
- Backtest metrics
- Paper/shadow metrics when available
- Promotion or rejection reason

Promotion criteria must include:

- Out-of-sample improvement
- Walk-forward stability
- Better or equal drawdown behavior
- Better or equal tail-loss behavior
- Reasonable calibration
- Adequate trade frequency
- No obvious data leakage

## Session Guidance For Future Codex Runs

When a new session opens this project:

1. Read this spec before assuming the README describes the future architecture.
2. Treat `docs/LEGACY_OPTIONWHEEL_README.md` and `src/scanner.py` as legacy documentation/reference for the inherited system.
3. Preserve deterministic risk, monitoring, and execution infrastructure unless explicitly replacing it.
4. Use `src/model_scanner.py` as the app-facing candidate hook. The legacy deterministic scanner should not be used as the default fallback.
5. Prefer transparent, testable ML infrastructure over clever model complexity.
6. Coach Surya through ML system concepts while implementing them.

## Near-Term Build Milestones

1. Create a data-source audit to test Alpaca and identify gaps.
2. Design provider interfaces for market data, option chains, events, and volatility data.
3. Build a historical candidate dataset pipeline.
4. Add feature generation with leakage guards.
5. Add P&L-centered label generation.
6. Train a first baseline tabular model.
7. Evaluate by time-split and walk-forward validation.
8. Add model registry and champion/challenger selection.
9. Add inference scoring for live/paper candidates.
10. Integrate deterministic risk gates after model scoring.
