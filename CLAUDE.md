# OptionMind — Claude Development Guide

## ML Training Workflow

### Primary model: XGBoost ranker

The canonical training target is **return on risk** (`return_on_risk = expected_pnl / max_loss`).
This normalises dollar PnL by capital at risk so the model ranks capital efficiency, not raw dollar size.
Do not switch back to `expected_pnl` without explicit instruction.

Training parameters that must hold by default:
- `--target return_on_risk`
- `--target-scale 0.10` (RoR values live in the 0–1 range)
- `--target-clip 5.0`
- `--num-boost-round 500` (fixed rounds, no early stopping)
- `--val-fraction 0.0 --early-stopping-rounds 0`
- `--downside-penalty 2.5` (heavier tail-loss punishment)
- `--embargo-days 30`

Canonical training command:
```
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/xgboost_<version>.json \
  --target return_on_risk \
  --target-scale 0.10 --target-clip 5.0 \
  --num-boost-round 500 \
  --val-fraction 0.0 --early-stopping-rounds 0 \
  --downside-penalty 2.5 \
  --embargo-days 30
```

### Risk classifiers (train before ranker evaluation)

Always retrain classifiers on the same dataset version as the ranker.
Both use the same script — only `--target` differs.

```
# Large-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/large_loss_classifier_<version>.json \
  --target large_loss_label --embargo-days 30 --num-boost-round 300

# Stop-loss classifier
PYTHONPATH=. .venv/bin/python -m ml.models.train_large_loss_classifier \
  --input  artifacts/datasets/candidate_rows/<dataset_version> \
  --output artifacts/models/stop_loss_classifier_<version>.json \
  --target stop_loss_hit --embargo-days 30 --num-boost-round 300
```

### Evaluation workflow (always run in this order)

**Step 1 — Ranker quality gate** (pure model signal, no portfolio controls):
```
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_exit_criteria \
  --input    artifacts/datasets/candidate_rows/<dataset_version> \
  --artifact artifacts/models/xgboost_<version>.json \
  --json-output artifacts/models/xgboost_<version>_exit_criteria.json
```

**Step 2 — Combined classifier + ranker evaluation** (classifiers as hard pre-filter):
```
PYTHONPATH=. .venv/bin/python -m ml.models.evaluate_risk_adjusted_ranking \
  --input              artifacts/datasets/candidate_rows/<dataset_version> \
  --ranker-artifact    artifacts/models/xgboost_<version>.json \
  --large-loss-artifact artifacts/models/large_loss_classifier_<version>.json \
  --json-output        artifacts/models/risk_adjusted_<version>_eval.json
```

Note: `--stop-loss-artifact` is not a recognised argument for `evaluate_risk_adjusted_ranking`; omit it.

Step 2 defaults to hard-filter mode: classifiers with `p > 0.70` veto a trade outright;
no soft penalty is applied. Portfolio controls are available via `--portfolio-risk-controls`
for final execution-layer simulation but are NOT part of training or exit-criteria evaluation.

## Architecture Decisions

### Training vs execution layers are separate

**Training** (`train_xgboost`, `evaluate_exit_criteria`):  
- Operates on the full holdout (~360K rows)
- Measures pure model signal quality
- No portfolio risk controls — they would collapse holdout to ~600 rows and destroy statistical power
- Exit criteria gates are calibrated in dollar PnL terms

**Execution** (`evaluate_risk_adjusted_ranking`, live agent):  
- Classifiers hard-filter high-risk trades (`p(large_loss) > 0.70`, `p(stop_loss) > 0.70`)
- Ranker scores the surviving candidates by predicted RoR
- Portfolio controls (gamma stress, concentration limits) applied last
- `portfolio_controls.py` is the single canonical implementation shared by both layers

### Dollar PnL metrics are always in dollars

`_credit_spread_selection_metrics` in `train_xgboost.py` always reads `expected_pnl` from the
DataFrame to report top-decile dollar metrics (`top_decile_actual_mean`, `top_decile_profit_factor`,
etc.) regardless of training target. This keeps all exit criteria thresholds calibrated in dollars
even when training on `return_on_risk`.

### Golden dataset

Base dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched`  
1.44M rows, 39 ETF underlyings, features_v005, labels_v002.

Champion training dataset (balanced): `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k`  
500K rows, derived from base via sqrt-frequency hierarchical sampling (max 12% per underlying).  
Located at: `artifacts/datasets/candidate_rows/dataset_version=<name>`

## Current model artifacts (V006c — champion)

V006c ranker is the registered champion in `artifacts/model_registry.json` and is loaded by the
live agent automatically. Trained via 2-pass Optuna optimisation (100 HP trials + 64 feature
trials) on the balanced 500K dataset. Passes all 33 exit criteria (V006b passed 16/33).

| Artifact | File | Notes |
|----------|------|-------|
| XGBoost ranker (champion) | `xgboost_v006c_500r_dp28.json` | PF 2.47 (+26%), RoR 0.167 (+10%), WR 78.6%; 38 features |
| Large-loss classifier | `large_loss_classifier_v006c.json` | WF AUC 0.8519 (+0.4%), precision +5.7%, F1 +4.9%; 33 features |
| Stop-loss classifier | `stop_loss_classifier_v006b.json` | AUC 0.804, recall 99.7% (not yet re-optimised) |

V006c key HP changes vs V006b: `reg_lambda` 10→37.5 (heavy L2), `colsample_bytree` 0.85→0.42
(aggressive column dropout), `eta` 0.05→0.020, `downside_penalty` 2.5→2.83, `huber_delta` 1.0→2.20.

V006c drops 23 features (5 groups) vs V006b: `underlying_price` (directional returns add bias),
`market_regime` (SPY-level features confound ETF-specific ranking), `vix_features` (macro fear
index hurts individual option quality assessment), `vol_momentum` (redundant with absolute vol),
`credit_efficiency` (credit_to_width already captures this).

Canonical retrain command (V006c):
```
PYTHONPATH=. .venv/bin/python -m ml.models.train_xgboost \
  --input  artifacts/datasets/candidate_rows/candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k \
  --output artifacts/models/xgboost_v006c_500r_dp28.json \
  --target return_on_risk --target-scale 0.10 --target-clip 5.0 \
  --num-boost-round 500 --val-fraction 0.0 --early-stopping-rounds 0 --embargo-days 30 \
  --eta 0.02033541341546644 --max-depth 3 --min-child-weight 5.467924645206163 \
  --subsample 0.6688247220130267 --colsample-bytree 0.41677825864481716 \
  --reg-lambda 37.4687664351599 --reg-alpha 0.1564206251204104 \
  --downside-penalty 2.829276302091746 --huber-delta 2.2027261783930134 \
  --exclude-features underlying_close,underlying_return_1d,underlying_return_3d,underlying_return_5d,underlying_return_20d,underlying_range_pct,underlying_sma_20_distance_pct,underlying_above_sma_20,underlying_volume,underlying_volatility_ratio_5d_20d,underlying_vol_vs_market,vol_acceleration,market_return_5d,market_return_20d,market_realized_vol_5d,market_realized_vol_20d,market_sma_20_distance_pct,market_above_sma_20,market_volatility_ratio_5d_20d,vix_regime,vix_return_5d,vix_realized_vol_5d,credit_per_day_per_risk
```

Rollback (ranker): `xgboost_v006b_500r_dp25.json` (PF 1.96, passes 16/33 exit criteria).

V006c large-loss classifier uses 33 features (from 60). Keeps: underlying_price, option_entry,
vix_features, credit_efficiency. Drops: greeks, iv_surface, event_risk, underlying_vol,
option_activity, market_regime, vol_momentum. HP unchanged from V006b (HP search failed due to
--reg-alpha CLI bug, now fixed — re-run `optimize_large_loss_classifier --mode hp` for further gains).

Previous artifacts (V006, trained on full 1.44M enriched dataset — kept as reference):

| Artifact | File |
|----------|------|
| XGBoost ranker | `xgboost_v006_500r_dp25.json` |
| Large-loss classifier | `large_loss_classifier_v006.json` |
| Stop-loss classifier | `stop_loss_classifier_v006.json` |
