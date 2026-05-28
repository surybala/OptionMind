# XGBoost Credit Spread Exit Criteria

This is the production-candidate gate for the PCS/CCS ranker. An AutoML run
may use a continuous score to rank experiments, but it must not stop until every
hard gate below passes.

## Scope

- Strategies: put credit spreads and call credit spreads only.
- Universe: broad ETF options.
- Primary ranker: XGBoost score used to select the highest-ranked candidate
  spreads.
- Evaluation selection: top 10% of scored candidates on strictly chronological
  holdout and walk-forward folds.
- Primary training target: `return_on_risk` (expected PnL normalised by max loss).
- Dollar PnL metrics are always reported in dollars per spread regardless of training target.

## Required Data Gate

The model is not eligible unless:

- Dataset rows are at least `500,000`.
- Chronological holdout rows are at least `100,000`.
- Top-selection rows are at least `100`.
- Top-selection entry dates are at least `60`.
- Walk-forward folds are at least `3`.
- The evaluation uses a 30-calendar-day embargo, matching the forward label
  horizon.

Rationale: small samples can make top-decile selection look stable when it is
only selecting a few clustered trades. The entry-date requirement reduces the
chance that a top decile is merely many overlapping versions of the same trade.

## Required Edge Gate

On the final chronological holdout top 10%, all must pass:

- Mean P&L >= `$20`.
- Mean P&L after a `20%` slippage haircut to spread credit >= `$0`.
- Profit factor after a `20%` slippage haircut to spread credit >= `1.00`.
- Profit factor >= `1.35`.
- Win rate >= `58%`.
- Mean return on risk >= `0.13`. _(champion v006b: 0.152)_
- 5th percentile P&L >= `-$340`. _(champion v006b: -$314)_
- 1st percentile P&L >= `-$1,050`. _(champion v006b: -$935)_
- Worst selected trade >= `-$1,800`. _(champion v006b: -$1,710)_

On walk-forward top 10%, all must pass:

- Every fold mean P&L >= `$10`.
- Average fold mean P&L >= `$25`.
- Every fold profit factor >= `1.20` independently.
- Average fold profit factor >= `1.35`.
- Every fold win rate >= `55%`.
- Every fold 5th percentile P&L >= `-$450`. _(champion v006b: -$400)_
- Every fold worst selected trade >= `-$2,000`.

Rationale: the model must have an edge in each market slice, not only in the
most recent holdout. A high average profit factor is not enough; a single fold
below `1.20` means the edge is regime-dependent and the AutoML run must
continue.

## Required Slippage Gate

Credit spreads are evaluated with a simulated slippage penalty:

```text
slippage_adjusted_pnl = expected_pnl - 0.20 * max_profit
```

`max_profit` is the collected credit in dollars per spread. This assumes the
backtest fills near mid-price and production gives up 20% of the credit to
spread crossing, commissions, and quote friction.

The model is not eligible unless the selected holdout top 10% remains profitable
after this haircut.

## Required Tail Gate

On the final chronological holdout top 10%, all must pass:

- `large_loss_label` rate <= `15%`.
- `stop_loss_hit` rate <= `30%`.
- Max drawdown / gross selected profit <= `45%`.
- Max drawdown <= `50%` of the catastrophic account limit.
- Max adverse excursion <= `50%` of the catastrophic account limit.

On each walk-forward selected top 10%, all must pass:

- Max drawdown <= `50%` of the catastrophic account limit.
- Max adverse excursion <= `50%` of the catastrophic account limit.

Rationale: selling credit spreads dies by left-tail clustering. We accept that
losses happen; we do not accept a selection rule that repeatedly picks tail
trades.

The default `catastrophic_account_limit` used during offline evaluation is
`$200,000`. This is calibrated for a holdout sequence of ~12,500 selected
trades; the drawdown gate checks cumulative sequence risk, not per-position risk.
Live per-position and per-portfolio limits are enforced exclusively by
`portfolio_controls.py` and are set much lower. Override the evaluation limit
with `--catastrophic-account-limit` if evaluating a different holdout size.

## Required Feature Stability Gate

The top-3 fold-level XGBoost gain-importance features must remain stable between
the first and last walk-forward folds:

- Top-2 feature overlap between the first and last fold must be >= `2`.
- Missing fold-level feature importance fails the gate.

Rationale: if the first fold is driven by one set of features and the last fold
is driven by a completely different set, the model is likely curve-fitting
regime-specific noise rather than learning a persistent spread-selling edge.

## Required Stability Gate

The model is not eligible if training performance is too far ahead of holdout:

- Train top-decile mean P&L / holdout top-decile mean P&L <= `2.5`.
- Train top-decile profit factor / holdout top-decile profit factor <= `2.5`.

If the holdout value is non-positive, the ratio gate fails automatically.

Rationale: this catches memorization and over-tuned hyperparameters.

## Required Concentration Gate

On the final chronological holdout top 10%:

- No single ETF may exceed `25%` of selected trades.
- Top 5 ETFs may not exceed `65%` of selected trades.

Rationale: an ETF-specific anomaly is not a robust broad-ETF edge.

## AutoML Stop Rule

The AutoML loop should:

1. Train a candidate model with fixed chronological validation and walk-forward
   folds.
2. Run `ml.models.evaluate_exit_criteria`.
3. Record the report JSON with the model artifact.
4. Stop only when `passed == true`.
5. Otherwise continue search, ranking failures by `score`.

The score is only a prioritization heuristic. A high score with one failed
tail gate is still not production-candidate ready.

Example (v006b champion command):

```bash
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_exit_criteria \
  --input    artifacts/datasets/candidate_rows/dataset_version=candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --artifact artifacts/models/xgboost_v006b_500r_dp25.json \
  --json-output artifacts/models/xgboost_v006b_500r_dp25_exit_criteria.json
```

## Threshold Calibration Note

Thresholds above are set at approximately 85–88% of the v006b champion's actual values. This means:

- Any future challenger must beat the champion to clear the gate.
- The headroom is intentional — marginal regression from the champion still fails.
- When a new champion is promoted, update the thresholds accordingly in
  `ml/models/evaluate_exit_criteria.py` (`ExitCriteriaConfig` dataclass defaults).

## Production Promotion Gate

Passing the offline exit criteria is necessary but not sufficient for real
capital. Before trading live, require a shadow/paper run:

- At least 30 trading days or 200 selected paper trades, whichever is longer.
- Paper top-selection profit factor >= `1.25`.
- Paper large-loss rate <= `15%`.
- No single ETF exceeds `25%` of selected paper trades.
- Observed slippage and fill assumptions are within the backtest model budget.
- No data freshness, option-chain quality, or quote-staleness incident remains
  unresolved.

Only after both offline and shadow gates pass should the model be eligible for a
small-capital production rollout.
