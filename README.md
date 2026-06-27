# OptionMind

OptionMind is an ML-first options trading research and execution platform focused on credit-spread strategies (put credit spreads and call credit spreads) across broad ETF underlyings.

## Current ML Pipelines (Live)

The live system has two distinct ML layers:

- `Entry pipeline`: decides which new spreads are worth opening.
- `Risk pipeline`: monitors open positions in realtime and closes trades that look likely to deteriorate into stop-loss events.

## Entry Pipeline

The platform runs a three-gate entry funnel end-to-end:

```
Market candidates
      │
      ▼
[1] XGBoost ranker          — scores each spread by predicted return-on-risk
      │
      ▼
[2] Large-loss classifier   — vetoes spreads where p(large_loss) > 0.60
      │
      ▼
[3] Stop-loss classifier    — vetoes spreads where p(stop_loss_hit) > 0.30
      │
      ▼
[4] Portfolio risk controls — gamma stress cap, concentration limits
      │
      ▼
Final trade picks
```

The ranker champion is registered in [artifacts/model_registry.json](/Users/surya/IdeaProjects/OptionMind/artifacts/model_registry.json) and is loaded automatically at runtime.

### Current Live Entry Models

| Artifact | File | Key Metric |
|----------|------|------------|
| Champion RoR ranker | `artifacts/models/xgboost_v007b_dte21_quant.json` | Holdout RoR `0.4717`, PF `1.733`, win rate `68.9%`, mean PnL `$46.39`, WF PF min `1.874` |
| Large-loss classifier | `artifacts/models/large_loss_classifier_v008.json` | Holdout AUC `0.8439`, recall `91.1%` at `0.60`, precision `38.3%`, WF AUC `0.8521` |
| Stop-loss classifier | `artifacts/models/stop_loss_classifier_v008.json` | Holdout AUC `0.8002`, recall `99.5%` at `0.30`, precision `36.6%`, WF AUC `0.8230` |

Training dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21`
(500K rows, DTE<=21, sqrt-frequency hierarchical sampling across 39 ETF underlyings, max 12% per underlying)

The current champion ranker (`xgboost_v007b_dte21_quant`) is a DTE-matched return-on-risk model promoted for the live entry funnel:

- Target: `return_on_risk`
- Feature set: `43` engineered features from `features_v006`, including `vrp_5d`, `vrp_20d`, `gamma_theta_ratio`, `sigma_buffer`, and `vega_margin_ratio`
- Training workflow: `100` Optuna hyperparameter trials on the DTE<=21 live-matched dataset
- Promotion status: current live champion in `artifacts/model_registry.json` with rollback target `xgboost_v006c_500r_dp28`

The two entry-time classifiers serve different purposes:

- `large_loss_classifier_v008` catches tail-risk setups that are likely to become outsized losers and vetoes them at `p(large_loss) > 0.60`.
- `stop_loss_classifier_v008` catches trades that are likely to trip deterministic stop-loss logic and vetoes them at `p(stop_loss_hit) > 0.30`.

## Open-Position ML Risk Monitor

Once a spread is open, a separate intraday model monitors the live position state:

```
Open position
      │
      ▼
[1] ML risk-exit model          — proactive close on rising short-horizon stop risk
      │
      ▼
[2] Profit-take rule            — lock in captured premium when no ML close fires
      │
      ▼
[3] Deterministic stop-loss     — final fallback catch-all
```

The live monitor currently evaluates the ML exit-risk gate before profit-take. A close only fires after the configured number of consecutive confirmations.

This model is registered separately in [artifacts/risk_model_registry.json](/Users/surya/IdeaProjects/OptionMind/artifacts/risk_model_registry.json).

### Current Live Risk Models

| Artifact | File | Role | Key Metric |
|----------|------|------|------------|
| Champion risk monitor | `artifacts/models/intraday_risk_monitor_stop30m_v004.json` | Active live proactive exit model | Holdout AUC `0.9303`, recall `69.2%`, close rate `6.40%`, false-close rate `6.18%` |
| Rollback risk monitor | `artifacts/models/intraday_risk_monitor_stop30m_fullraw_v003.json` | Previous champion / rollback target | Holdout AUC `0.9458`, recall `71.6%`, close rate `5.48%`, false-close rate `5.34%` |

Risk-monitor training corpus:

- Raw intraday risk rows: about `3.9M`
- Final objective: predict `stop_loss_hit_30m`
- Feature set: live-compatible intraday spread-state features, realized vol, and option-leg mark structure
- Current live feature version: `intraday_risk_live_monitor_features_v001`

The active champion (`intraday_risk_monitor_stop30m_v004`) runs at a higher operating threshold (`0.08` instead of `0.05`) and requires two consecutive confirmations before closing. The promotion notes record it as the winning challenger over `v003` at matched thresholds, while the live runtime uses the stricter `0.08` operating point to reduce single-minute churn.

### How The Risk Model Prevents Losses

The model does not try to predict final trade P&L in the abstract. It predicts whether the current open spread state is likely to hit stop loss within the next `30` minutes.

In practice that means it can react to:

- spread debit expanding toward the stop
- loss as a fraction of max loss
- live leg-mark deterioration
- short-horizon realized volatility expansion
- time-in-trade and time-to-expiry context

Important operational details:

- It uses live market inputs from the position monitor, not historical parquet training data.
- In the current `hft_mode=true` setup, predictions are fed by Alpaca live option snapshots and Alpaca spot prices.
- The model is configured as a proactive layer, while deterministic stop-loss remains active as the final safety net.

### Runtime Configuration

`config.json` controls both the entry scanner and the open-position ML risk service:

```json
{
  "universe": "etf",
  "ml_scanner": {
    "enabled": true,
    "registry_path": "artifacts/model_registry.json",
    "large_loss_classifier_path": "artifacts/models/large_loss_classifier_v008.json",
    "large_loss_veto_threshold": 0.60,
    "stop_loss_classifier_path": "artifacts/models/stop_loss_classifier_v008.json",
    "stop_loss_veto_threshold": 0.30,
    "min_dte": 7,
    "max_dte": 21
  },
  "risk_parameters": {
    "stop_loss_max_loss_pct": 0.30,
    "new_position_grace_minutes": 2,
    "ml_exit_risk": {
      "enabled": true,
      "registry_path": "artifacts/risk_model_registry.json",
      "threshold": 0.08,
      "confirmations_required": 1,
      "min_age_minutes": 5
    }
  }
}
```

Notes:

- `universe: "etf"` now means the curated stable ETF preset, not every listed ETF.
- `stop_loss_max_loss_pct = 0.30` caps spread losers once 30% of max loss is reached.
- `new_position_grace_minutes = 2` keeps only a short post-entry quote-settle buffer.
- `ml_exit_risk.threshold = 0.08` is the risk-score trigger cutoff.
- `confirmations_required = 1` closes on the first qualifying ML exit-risk breach.
- `min_age_minutes = 5` keeps a brief warm-up window before ML exits can fire.
- `risk_parameters.max_loss_multiple` is enabled to reject underpaid spread geometry before execution.
- Dashboard open-position severity badges are now driven purely by ML exit-risk score proximity (`SAFE` / `WATCH` / `CAUTION` / `CRITICAL`), not the legacy heuristic stop/gamma labels.

## ML Documentation

- [ML Trade Pipeline](docs/ML_TRADE_PIPELINE.md) — three-gate funnel, evaluation commands, promotion rules
- [Intraday Risk Dataset](docs/INTRADAY_RISK_DATASET.md) — intraday risk corpus, training flow, Optuna tuning, model promotion
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
| `ml/models/train_intraday_risk_monitor.py` | Intraday risk-exit training (target: `stop_loss_hit_30m`) |
| `ml/models/optimize_intraday_risk_monitor.py` | Hyperparameter tuning for the intraday risk monitor |
| `ml/models/optimize_intraday_risk_monitor_features.py` | Feature-group search for the intraday risk monitor |
| `ml/models/backtest_intraday_risk_monitor.py` | Trade-path backtesting for risk-exit models |
| `ml/models/evaluate_exit_criteria.py` | Ranker-only quality gate |
| `ml/models/evaluate_risk_adjusted_ranking.py` | Combined ranker + classifier evaluation |
| `ml/models/registry.py` | Model registry: register, promote, load champion |
| `src/risk_ml.py` | Live ML risk-exit model loading and scoring for open positions |
| `src/portfolio_risk.py` | Portfolio risk service: gamma stress caps, concentration limits |
| `src/position_monitor.py` | Open-position monitor: profit-take, ML risk exit, deterministic stop-loss fallback |
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
