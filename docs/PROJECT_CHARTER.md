# OptionMind Project Charter

Last updated: 2026-05-24

## Mission

OptionMind is evolving from a deterministic options-selling system into a learning-driven option trading platform. The goal is to build a foundational model that can evaluate option trade candidates from historical market behavior, predict P&L-centered outcomes, and eventually guide an intelligent option trader while deterministic controls continue to enforce safety.

The system should become better through evidence, not intuition: every prediction, decision, trade, and realized outcome should feed a closed learning loop.

## Coaching Agreement

Codex should act as Surya's ML systems coach while building OptionMind. This means:

- Explain ML concepts as they become relevant to the implementation.
- Prefer clear, inspectable first versions before complex models.
- Teach the reasoning behind modeling, validation, feature design, and deployment choices.
- Keep the project moving through working code, tests, and measurable milestones.
- Call out overfitting, data leakage, survivorship bias, weak validation, and unsafe deployment paths early.

The coaching goal is not only to build the system, but to help Surya learn how to design, train, evaluate, and operate an ML trading system responsibly.

## Strategic Direction

The learned model should not depend on the existing deterministic scan filters for candidate generation. The new predictor should learn from historical options-market opportunities directly.

The current deterministic system remains valuable as the control layer:

- Account-level risk limits
- Position sizing constraints
- Stop-loss controls
- Portfolio concentration controls
- Kill switches and operational safeguards
- Monitoring, reconciliation, and execution plumbing

The ML layer proposes and ranks candidates. The deterministic layer decides what is allowed.

## Modeling Focus

Delta and other Greeks are useful inputs, but they are not enough. Delta is a theoretical probability proxy, not a complete P&L forecast. OptionMind models should emphasize realized trade economics and loss risk.

Primary prediction targets should include:

- Expected P&L
- Probability of profit
- Probability of stop-loss hit
- Probability of large loss
- Max adverse excursion
- Max favorable excursion
- Expected holding period
- Risk-adjusted expected return

Candidate ranking should optimize expected utility, not raw win rate.

## Feature Philosophy

Greeks are one feature family, not the full model. Feature sets should include:

- Option contract features: type, strike, expiration, DTE, moneyness, bid/ask, spread, volume, open interest
- Greeks and volatility features: delta, gamma, theta, vega, rho, IV, IV rank, IV percentile, skew, term structure
- Underlying behavior: gaps, candles, ATR, realized volatility, trend, VWAP distance, prior range, opening behavior
- Market regime: SPY/QQQ/IWM movement, VIX level/change, breadth, sector movement, macro volatility
- Event risk: earnings, CPI, FOMC, jobs reports, ex-dividend dates, major known catalysts
- Execution quality: liquidity, slippage estimates, quote quality, option volume, open interest
- Portfolio context: correlation, existing exposure, buying power, symbol/sector concentration

Opening behavior is a first-class feature family because large sell-offs, gap moves, and high-range opening candles can materially change option-selling risk.

## Data Strategy

Alpaca is an important broker, execution venue, and market-data source, but OptionMind should not be limited to Alpaca data.

Use Alpaca where it is strong:

- Live and historical stock/ETF bars
- Live option chains and snapshots
- Option contract metadata
- Historical option bars/trades where available
- Paper/live trading execution

Complement Alpaca with additional sources when needed:

- Historical option chain datasets
- Historical IV and Greeks
- Corporate actions and dividends
- Earnings calendars
- Macro event calendars
- VIX and volatility term-structure data
- Market breadth and sentiment datasets
- Vendor-grade OPRA or options analytics datasets

The data layer should make sources pluggable so we can audit data quality and replace or augment providers without rewriting the training system.

## Closed Learning Loop

OptionMind should operate as a model lifecycle system:

1. Build historical candidate datasets.
2. Generate features using only information available at decision time.
3. Label outcomes through forward simulation and realized trade results.
4. Train many model candidates.
5. Evaluate on strict out-of-sample time periods.
6. Backtest the trading policy, not just the prediction score.
7. Select a champion only when it beats the current champion on risk-adjusted metrics.
8. Paper trade or shadow trade before live deployment.
9. Log every live prediction, decision, and outcome.
10. Feed those outcomes into future training iterations.

Models should be versioned with their data range, feature set, labels, parameters, metrics, and promotion decision.

## Validation Rules

OptionMind should treat these as non-negotiable:

- No future data in features.
- No random train/test splits for time-series trading data.
- Use walk-forward and regime-aware validation.
- Evaluate P&L, drawdown, calibration, and tail losses, not only accuracy.
- Promote models only when they improve out-of-sample performance and do not hide larger tail risk.
- Keep deterministic risk controls active even when the model is confident.

## First ML Milestone

Build an Alpaca-plus-data-source audit and a first dataset pipeline that answers:

- Can we retrieve enough historical option contracts and bars?
- Can we reconstruct historical candidate universes?
- Do we need to compute historical IV and Greeks ourselves?
- Which external data sources fill the gaps?
- Can we produce one training row per historical candidate with entry-time features and forward-simulated P&L labels?

The first working model should be simple and inspectable, likely LightGBM, XGBoost, or CatBoost, before neural networks are introduced.
