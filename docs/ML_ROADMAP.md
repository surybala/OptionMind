# OptionMind ML Roadmap

Last updated: 2026-06-15

This roadmap turns the north-star spec into buildable milestones.

**Current status: the live runtime stack is wired around `xgboost_v007b_dte21_quant` + `large_loss_classifier_v008` + `stop_loss_classifier_v008`, with `intraday_risk_monitor_stop30m_v004` guarding open positions.**
Phases 0–6 are complete. Current work is rollout hardening and keeping offline evaluation aligned with the live stack.

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

Balanced training dataset: `candidate_rows_massive_broad_etfs_pcs_ccs_20220526_20260425_v006_balanced_cap12_500k_dte21`
500K rows, DTE<=21, sqrt-frequency hierarchical sampling, max 12% per underlying.

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
- `ml/models/train_intraday_risk_monitor.py` — dedicated open-position exit-risk model for `stop_loss_hit_30m`.

Current live stack metrics:
- Ranker `xgboost_v007b_dte21_quant`: holdout RoR `0.4717`, PF `1.733`, win rate `68.9%`, mean PnL `$46.39`, walk-forward PF min `1.874`
- Large-loss classifier `large_loss_classifier_v008`: AUC `0.843878`, recall `0.910883`, precision `0.38261`, walk-forward AUC `0.852065`
- Stop-loss classifier `stop_loss_classifier_v008`: AUC `0.800183`, recall `0.995387`, precision `0.365765`, walk-forward AUC `0.822999`
- Open-position risk monitor `intraday_risk_monitor_stop30m_v004`: AUC `0.930315`, recall `0.691576`, close rate `0.063957`, false-close rate `0.061815`

---

## Phase 5: Model Registry And Champion Selection ✓

Goal: create a repeatable model lifecycle.

Implemented:

- `ml/models/registry.py` — register, promote, load champion entry artifacts.
- `artifacts/model_registry.json` — single source of truth for the live entry stack.
- `artifacts/risk_model_registry.json` — single source of truth for the live open-position exit model.
- Current champions: `xgboost_v007b_dte21_quant` (2026-06-04), `large_loss_classifier_v008` and `stop_loss_classifier_v008` (2026-06-10), `intraday_risk_monitor_stop30m_v004` (2026-06-10).

Each model version records: training data range, feature set version, label definition,
model type and parameters, validation metrics, backtest metrics, promotion decision.

---

## Phase 6: Paper And Shadow Inference ✓

Goal: score real market candidates without risking capital first.

Implemented flow:

1. `LivePaperInferenceProvider` in `src/model_scanner.py` generates PCS/CCS spread candidates from live option chains.
2. Champion ranker scores candidates by predicted return-on-risk.
3. Large-loss classifier vetoes candidates with `p(large_loss) > 0.60`.
4. Stop-loss classifier vetoes candidates with `p(stop_loss_hit) > 0.30`.
5. `PortfolioRiskService.filter_picks()` applies gamma stress caps and concentration limits.
6. Trade picks are logged and executed via the runtime mode configured for the agent.
7. Realized outcomes feed back into the dataset pipeline.

The entry artifacts are loaded automatically from `artifacts/model_registry.json`.
The large-loss and stop-loss classifier paths are configured in `config.json` under `ml_scanner.*`.

---

## Phase 7: Limited Live Deployment (Hardening In Progress)

Goal: carefully introduce live trades with deterministic controls still active.

Controls required before going live:

- Small allocation
- Strict buying-power checks
- Max daily loss
- Max position size
- Max concentration
- Realtime stop-loss / exit-risk monitor
- Kill switch
- Rollback to previous champion

Prerequisites:

- At least 30 trading days or 200 selected paper trades passing the shadow gate.
- Paper top-selection profit factor >= 1.25.
- Paper large-loss rate <= 15%.
- Slippage and fill assumptions within backtest budget.

---

## Immediate Next Tasks

- Keep the live documentation and config aligned around the `v007b` / `v008` / `v004` stack and current thresholds (`0.60`, `0.30`, `0.08`, `2` confirmations).
- Continue hardening evaluation so promotion decisions reflect the real live funnel, including entry vetoes, portfolio controls, and the separate open-position exit-risk layer.
- Use the `intraday_risk_rows` training flow to iterate on challengers for `intraday_risk_monitor_stop30m_v004` without regressing realtime feature compatibility.
