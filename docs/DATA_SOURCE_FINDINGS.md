# OptionMind Data Source Findings

Last updated: 2026-05-24

## Alpaca Coverage Matrix Run

Command:

```bash
.venv/bin/python -m ml.data_audit.alpaca_audit --coverage-matrix --coverage-years 0,1,3,5,7 --underlyings SPY,QQQ,AAPL
```

Generated report:

```text
artifacts/data_audit/alpaca_audit_20260524_193744.md
```

The artifact directory is intentionally ignored by git, so this file records the durable interpretation.

## Result Summary

Alpaca passed the basic authenticated capability audit:

- Historical stock bars are available.
- Current active option contracts are available.
- Recent inactive or expired option contracts are available.
- Current option chain snapshots are available.
- Current option snapshots can include IV and Greeks.
- Historical option bars are available for sampled recent contracts.
- Historical option trades are available for sampled recent contracts.
- Minute bars around the market open are available for underlying symbols.

The expanded coverage matrix returned:

```text
4/15 symbol-year rows had stock, opening-window, option-contract, and option price/trade coverage.
0/15 rows exposed historical Greeks or IV directly from historical bar/trade responses.
```

## Matrix Highlights

For `SPY`, `QQQ`, and `AAPL`:

- Underlying daily bars were available for all tested windows: current, 1 year, 3 years, 5 years, and 7 years.
- Underlying minute bars around market open were available for all tested windows.
- Current option chain snapshots exposed Greeks and IV.
- Historical option bar/trade responses exposed price, volume, VWAP, trade count, and trade details, but not Greeks or IV.
- Option contract lookup worked for current and 1-year-back windows.
- Option contract lookup returned no sampled contracts for 3-, 5-, and 7-year-back windows in this audit.

## Interpretation

Alpaca appears strong enough for:

- Live and paper execution.
- Current option-chain inference.
- Underlying feature generation, including market-open behavior.
- Recent option price/trade replay experiments.
- Initial provider integration and dataset pipeline scaffolding.

Alpaca alone does not yet look sufficient for the full north-star training dataset because:

- We did not observe multi-year historical option contract coverage beyond the recent/1-year window.
- Historical bar/trade responses did not include point-in-time Greeks or IV.
- A model trained on millions of historical option opportunities will need reliable full-chain historical coverage across many years.

## Data Gap Decisions

For v1, use Alpaca to build the provider interface and prototype the training-row pipeline.

For the real historical training corpus, investigate complementary data sources for:

- Multi-year historical full option chains.
- Point-in-time historical IV and Greeks, or enough inputs to compute them reliably.
- Corporate actions and dividends.
- Earnings and macro event calendars.
- VIX and volatility term structure.

Candidate provider categories:

- OptionMetrics IvyDB-style datasets.
- Cboe DataShop historical options datasets.
- OPRA-grade historical vendors.
- Earnings/event calendar providers.
- FRED/Cboe-style macro and volatility feeds.

## Next Build Implication

The next implementation should define provider interfaces before building the dataset generator. Alpaca should become the first provider implementation, but the interfaces must allow a second historical-options provider without rewriting feature generation or labeling.
