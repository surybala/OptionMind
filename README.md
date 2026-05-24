# OptionMind

OptionMind is an ML-first options trading research and execution platform.

The old OptionWheel deterministic scanner is no longer the product direction. It remains in the repository only as legacy reference code and for historical tests. New work should build the learned model, dataset pipeline, model registry, and inference scanner described in the docs below.

## Start Here

Read these before making architecture decisions:

- [North Star Spec](docs/NORTH_STAR_SPEC.md)
- [Project Charter](docs/PROJECT_CHARTER.md)
- [ML Roadmap](docs/ML_ROADMAP.md)
- [Data Source Findings](docs/DATA_SOURCE_FINDINGS.md)
- [Provider Interfaces](docs/PROVIDER_INTERFACES.md)
- [Candidate Dataset Prototype](docs/CANDIDATE_DATASET_PROTOTYPE.md)
- [Dataset Storage](docs/DATASET_STORAGE.md)

The inherited deterministic system notes were moved to [Legacy OptionWheel README](docs/LEGACY_OPTIONWHEEL_README.md). Treat that document as historical context, not the current north star.

## Current Architecture Direction

OptionMind should:

- Learn from historical options-market opportunities.
- Generate candidates from market/provider data, not fixed scanner rules.
- Predict P&L-centered outcomes such as expected P&L, stop-loss probability, large-loss probability, and max adverse excursion.
- Rank candidates with a trained model.
- Keep deterministic account-level risk, position sizing, stop-loss, exposure, monitoring, reconciliation, and execution controls.
- Log every prediction, decision, and realized outcome so future model versions improve.

## Current Runtime State

The app-facing scanner path now uses an ML scanner hook:

- `src/model_scanner.py`

Until a trained model or inference provider is plugged in, the hook returns no trade candidates. That is intentional. We should not fall back to the old deterministic scanner just because a model is not ready yet.

The legacy scanner remains here only for reference and tests:

- `src/scanner.py`
- `src/scan_filters/`
- `src/scan_strategies/`

Future sessions should avoid extending those legacy scanner modules unless the user explicitly asks to inspect or migrate old behavior.

## ML Scanner Hook

Configure a future inference provider under `ml_scanner` in `config.json`:

```json
{
  "ml_scanner": {
    "enabled": true,
    "provider": "some.module:function_or_class",
    "min_score": 0.0
  }
}
```

The provider should return candidate dictionaries compatible with the existing risk and execution pipeline. At minimum, a candidate should include:

```text
symbol
strategy
expiry
premium
current_price
quantity
model_score
model_version
```

For executable spread-style candidates, include the relevant leg fields such as `short_strike`, `long_strike`, `short_put`, `long_put`, `short_call`, and `long_call`.

## Next Build Step

Extend the historical candidate dataset prototype with richer features, then train a transparent baseline model before wiring live inference into `src/model_scanner.py`.
