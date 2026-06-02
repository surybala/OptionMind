# OptionMind ML Roadmap

Last updated: 2026-05-27

This roadmap turns the north-star spec into buildable milestones.

**Current status: Phase 6 active (paper/shadow inference running with v006b champion).**
Phases 0–6 are complete. Phase 7 (limited live deployment) is the next milestone.

---

## Phase 0: Data Source Audit ✓

Goal: learn what data we can actually retrieve and where provider gaps exist.

Deliverables:

- Alpaca audit script for historical stock bars, option contracts, option bars/trades, current option chains, and opening-window data.
- JSON and Markdown audit reports.
- Gap list for data that Alpaca does not provide or does not provide historically.
- Candidate list of complementary providers.

---

## Phase 1: Provider Interfaces ✓

Goal: make the data layer provider-pluggable so Alpaca is a source, not a lock-in.

Interfaces implemented:

- `MarketDataProvider`
- `OptionContractProvider`
- `OptionChainProvider`
- `OptionPriceProvider`
- `VolatilityDataProvider`
- `EventDataProvider`

---

## Phase 2: Historical Candidate Dataset ✓

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

Current golden dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_20wide_enriched`
1.44M rows, 39 ETF underlyings, features_v005, labels_v002.

Balanced training dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k`
500K rows, sqrt-frequency hierarchical sampling, max 12% per underlying.

---

## Phase 3: Labeling Engine ✓

Goal: turn historical price paths into P&L-centered targets.

Labels implemented:

- `expected_pnl`
- `profit_label`
- `stop_loss_hit`
- `large_loss_label`
- `max_adverse_excursion`
- `max_favorable_excursion`
- `days_to_exit`
- `exit_reason`
- `return_on_risk` (derived: `expected_pnl / max_loss`)

Source: `ml/labels/short_option.py`, `ShortOptionLabelConfig`, `label_short_option_path`.
Each row records `label_version` so future label definitions can be compared without mixing training targets.

Supporting tooling: `ml/datasets/audit_candidate_dataset.py` reports dataset quality before training.

---

## Phase 4: Baseline and XGBoost Models ✓

Goal: train inspectable models with rigorous time-ordered evaluation.

Implemented:

- `ml/models/train_baseline.py` — transparent least-squares baseline (reference only).
- `ml/models/train_xgboost.py` — canonical XGBoost ranker with `return_on_risk` target.
- `ml/models/train_large_loss_classifier.py` — binary classifier for `large_loss_label` and `stop_loss_hit`.
- `ml/models/evaluate_exit_criteria.py` — hard quality gates for the ranker.
- `ml/models/evaluate_risk_adjusted_ranking.py` — combined ranker + classifier evaluation.

Current champion v006b metrics (top-10% holdout, 12,636 trades):
- PF: 1.96 | Win rate: 75.6% | Mean PnL: $56 | Mean RoR: 15.2%
- Large-loss classifier: AUC 0.848, recall 98.2%
- Stop-loss classifier: AUC 0.804, recall 99.7%

---

## Phase 5: Model Registry And Champion Selection ✓

Goal: create a repeatable model lifecycle.

Implemented:

- `ml/models/registry.py` — register, promote, load champion.
- `artifacts/model_registry.json` — single source of truth for the champion ranker.
- v006b XGBoost ranker promoted as champion (2026-05-27).

Each model version records: training data range, feature set version, label definition,
model type and parameters, validation metrics, backtest metrics, promotion decision.

---

## Phase 6: Paper And Shadow Inference ✓ (Active)

Goal: score real market candidates without risking capital first.

Implemented flow:

1. `LivePaperInferenceProvider` in `src/model_scanner.py` generates PCS/CCS spread candidates from live option chains.
2. Champion ranker scores candidates by predicted return-on-risk.
3. Large-loss classifier vetoes candidates with `p(large_loss) > 0.70`.
4. `PortfolioRiskService.filter_picks()` applies gamma stress caps and concentration limits.
5. Trade picks are logged and executed as paper trades via Alpaca.
6. Realized outcomes feed back into the dataset pipeline.

The champion artifact is loaded automatically from `artifacts/model_registry.json`.
The large-loss classifier path is configured in `config.json` under `ml_scanner.large_loss_classifier_path`.

---

## Phase 7: Limited Live Deployment

Goal: carefully introduce live trades with deterministic controls still active.

Controls required before going live:

- Small allocation
- Strict buying-power checks
- Max daily loss
- Max position size
- Max concentration
- Stop-loss monitor
- Kill switch
- Rollback to previous champion

Prerequisites:

- At least 30 trading days or 200 selected paper trades passing the shadow gate.
- Paper top-selection profit factor >= 1.25.
- Paper large-loss rate <= 15%.
- Slippage and fill assumptions within backtest budget.

---

## Immediate Next Task

Complete the paper/shadow gate (Phase 6) by accumulating ≥ 30 days or 200 paper trades and
verifying profit factor ≥ 1.25 and large-loss rate ≤ 15% before considering Phase 7 rollout.

Parallel build now underway: an `intraday_risk_rows` training corpus seeded
from `candidate_rows` and backfilled with Massive minute bars so the future
live risk monitor can train a dedicated exit-hazard model instead of reusing
the entry ranker.
