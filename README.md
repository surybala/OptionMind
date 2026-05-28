# OptionMind

OptionMind is an ML-first options trading research and execution platform focused on credit-spread strategies (put credit spreads and call credit spreads) across broad ETF underlyings.

## Current ML Pipeline (Live)

The platform runs a three-gate selection funnel end-to-end:

```
Market candidates
      │
      ▼
[1] XGBoost ranker          — scores each spread by predicted return-on-risk
      │
      ▼
[2] Large-loss classifier   — vetoes spreads where p(large_loss) > 0.70
      │
      ▼
[3] Portfolio risk controls — gamma stress cap, concentration limits
      │
      ▼
Final trade picks
```

The champion ranker is registered in `artifacts/model_registry.json` and is loaded automatically at runtime.

### Current Champion: V006b

| Artifact | File | Key Metric |
|----------|------|------------|
| XGBoost ranker | `artifacts/models/xgboost_v006b_500r_dp25.json` | PF 1.96, win rate 75.6%, mean PnL $56, top-decile RoR 15.2% |
| Large-loss classifier | `artifacts/models/large_loss_classifier_v006b.json` | AUC 0.848, recall 98.2% |
| Stop-loss classifier | `artifacts/models/stop_loss_classifier_v006b.json` | AUC 0.804, recall 99.7% |

Training dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k`
(500K rows, sqrt-frequency hierarchical sampling across 39 ETF underlyings, max 12% per underlying)

### Runtime Configuration

`config.json` controls the ML scanner:

```json
{
  "ml_scanner": {
    "enabled": true,
    "registry_path": "artifacts/model_registry.json",
    "large_loss_classifier_path": "artifacts/models/large_loss_classifier_v006b.json",
    "large_loss_veto_threshold": 0.70,
    "min_dte": 7,
    "max_dte": 45
  }
}
```

## ML Documentation

- [ML Trade Pipeline](docs/ML_TRADE_PIPELINE.md) — three-gate funnel, evaluation commands, promotion rules
- [XGBoost Exit Criteria](docs/XGBOOST_CREDIT_SPREAD_EXIT_CRITERIA.md) — hard quality gates a ranker must pass before promotion
- [V006 / V006b Model Analysis](docs/V006_MODEL_STITCHING.md) — artifact map, run analysis, v006b champion promotion record
- [ML Roadmap](docs/ML_ROADMAP.md) — completed phases and current work
- [North Star Spec](docs/NORTH_STAR_SPEC.md) — product direction
- [Project Charter](docs/PROJECT_CHARTER.md)
- [Provider Interfaces](docs/PROVIDER_INTERFACES.md)
- [Data Source Findings](docs/DATA_SOURCE_FINDINGS.md)
- [Dataset Storage](docs/DATASET_STORAGE.md)

## Key Source Files

| File | Purpose |
|------|---------|
| `src/model_scanner.py` | Live inference: loads champion ranker, applies large-loss classifier, emits picks |
| `ml/models/train_xgboost.py` | XGBoost ranker training (target: `return_on_risk`) |
| `ml/models/train_large_loss_classifier.py` | Binary classifier training (targets: `large_loss_label`, `stop_loss_hit`) |
| `ml/models/evaluate_exit_criteria.py` | Ranker-only quality gate |
| `ml/models/evaluate_risk_adjusted_ranking.py` | Combined ranker + classifier evaluation |
| `ml/models/registry.py` | Model registry: register, promote, load champion |
| `src/portfolio_controls.py` | Portfolio risk service: gamma stress caps, concentration limits |
| `agent.py` | Main agent loop |

## Setup

```bash
./setup.sh
```

This creates a virtual environment, installs dependencies, validates your `config.json`, runs the test suite, and launches the agent.

Alpaca paper credentials must be set in `config.json`:

```json
{
  "alpaca": {
    "api_key": "YOUR_KEY",
    "api_secret": "YOUR_SECRET",
    "paper": true
  }
}
```

## Legacy Code

The old deterministic OptionWheel scanner (`src/scanner.py`, `src/scan_filters/`, `src/scan_strategies/`) remains in the repository for historical reference and existing tests only. Do not extend those modules for new work.

See [Legacy OptionWheel README](docs/LEGACY_OPTIONWHEEL_README.md) for historical context.
