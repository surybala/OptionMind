"""
PositionMonitor
===============

Checks every open position in the database against a stop-loss rule:

    If the current cost-to-close the position exceeds
    (1 + stop_loss_multiplier) × entry_premium, close it immediately.

Default stop_loss_multiplier = 1.5, which means:
    Loss > 1.5× premium collected  →  close now.
    Equivalently: current mark > 2.5× premium  →  exit.

Configuration (under 'risk_parameters' in config.json)
-------------------------------------------------------
  stop_loss_multiplier : float, default 1.5

Usage
-----
  monitor = PositionMonitor(db, executor, config)
  closed  = monitor.run(dry_run=True)   # call once per agent run

The monitor calls executor.execute_close_position() for each triggered
position, then marks the trade CLOSED in the database.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Optional

from src.greeks import position_risk_score

_log = logging.getLogger('optionwheel')


class PositionMonitor:
    """Stop-loss guardian for open option positions."""

    def __init__(self, db, executor, config: dict):
        self.db       = db
        self.executor = executor
        self.config   = config          # stored for Alpaca data client access
        risk = config.get('risk_parameters', {})
        self.stop_loss_multiplier = float(risk.get('stop_loss_multiplier', 2.0))
        self._new_position_grace_minutes = float(risk.get('new_position_grace_minutes', 0.0))

        # Gamma/theta dynamic risk config
        gr = risk.get('gamma_risk', {})
        self._gamma_risk_enabled      = bool(gr.get('enabled', True))
        # Close when gamma_theta_ratio > threshold AND other conditions met
        self._gr_ratio_threshold      = float(gr.get('gamma_theta_ratio_threshold', 1.5))
        # Don't trigger unless |short_delta| >= this (position is moving ITM)
        self._gr_min_delta            = float(gr.get('min_delta_to_trigger', 0.15))
        # Don't trigger unless we've captured at least this fraction of max profit
        self._gr_min_profit_pct       = float(gr.get('min_profit_captured_pct', 0.25))
        # Urgent: close regardless of profit captured if delta is this high
        self._gr_urgent_delta         = float(gr.get('urgent_delta_threshold', 0.35))

        # ── Profit-take rule ──────────────────────────────────────────────────
        self._pt_enabled  = bool(risk.get('profit_take_enabled', True))
        self._pt_pct      = float(risk.get('profit_take_pct', 0.75))

        # ── Risk rule instances ───────────────────────────────────────────────
        from src.risk_rules import RulePipeline, StopLossRule, ProfitTakeRule
        _max_loss_pct = risk.get('stop_loss_max_loss_pct')
        _max_loss_pct = float(_max_loss_pct) if _max_loss_pct is not None else None
        self._stop_loss_rule  = StopLossRule(self.stop_loss_multiplier,
                                             max_loss_pct=_max_loss_pct)
        self._profit_take_rule = ProfitTakeRule(self._pt_enabled, self._pt_pct)
        self._rule_pipeline   = RulePipeline([self._stop_loss_rule, self._profit_take_rule])

        # ── Market data adapter (HFT / non-HFT isolation) ────────────────────
        from src.market_data import DataAdapter
        from src.alpaca_data import make_alpaca_data_client
        self._data = DataAdapter(
            hist_cache      = {},
            chain_cache     = {},
            client_getter   = lambda: make_alpaca_data_client(config),
            hft_mode_getter = lambda: config.get('hft_mode', False),
            hft_config      = config.get('hft', {}),
        )

        # ── Shared risk enrichment service ────────────────────────────────────
        from src.risk_service import PositionRiskService
        self._risk_service = PositionRiskService(self._data, config)
        from src.risk_ml import MlExitRiskService
        self._ml_exit_risk = MlExitRiskService(config)
        self._ml_exit_confirmations: dict[int, int] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def reconcile_pending_closes(self) -> None:
        """
        Fast-path check for PENDING_CLOSE rows at the start of each risk cycle.

        The full reconciliation (all four checks) runs in a separate slower cycle
        via ``PositionReconciler`` in monitor.py.  This method only warns about
        stuck rows so the risk cycle isn't blocked on Alpaca position queries.
        Rows are resolved automatically by the next reconciliation cycle.
        """
        stuck = self.db.get_pending_close_positions()
        if not stuck:
            return
        _log.warning(
            "[PositionMonitor] %d position(s) in PENDING_CLOSE — will be resolved "
            "by the next reconciliation cycle.",
            len(stuck),
        )
        for pos in stuck:
            _log.warning(
                "  PENDING_CLOSE  id=%-4s  %s %-6s  expiry=%s  premium=%.2f",
                pos['id'], pos['type'], pos['symbol'], pos['expiry'], pos.get('premium', 0),
            )

    def run(self, dry_run: bool = True) -> list[dict]:
        """
        Scan all open positions and close any that breach a risk rule.

        Automatically selects HFT (Alpaca snapshots) or non-HFT (yfinance)
        data path based on ``config['hft_mode']``.

        In HFT mode, all option snapshots and spot prices are pre-fetched in
        2 bulk Alpaca calls before the position loop, reducing per-cycle API
        usage from ``2 × N`` sequential calls to exactly 2.

        Returns a list of position dicts that were closed (empty if none).
        """
        self.reconcile_pending_closes()
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return []
        hft = self._data.is_hft()

        _prefetch = None
        if hft:
            try:
                _prefetch = self._data.batch_prefetch_hft(
                    open_positions,
                    self._get_position_leg_specs,
                    self._build_osi_symbol,
                )
            except RuntimeError as exc:
                _log.warning(
                    "[HFT] batch prefetch failed — falling back to per-position fetch: %s",
                    exc,
                )

        return self._run_loop(
            open_positions,
            lambda pos: self._check_position(pos, dry_run, _prefetch),
            'PositionMonitor/HFT' if hft else 'PositionMonitor',
            catch_runtime=hft,
        )

    def run_hft(self, dry_run: bool = True) -> list[dict]:
        """
        HFT variant of :meth:`run`.

        Equivalent to ``run()`` but always catches per-position Alpaca
        failures (``RuntimeError``) so that one bad symbol doesn't abort
        the whole monitoring cycle.

        Prefer calling :meth:`run` directly — it auto-selects HFT mode
        based on ``config['hft_mode']``.
        """
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return []
        return self._run_loop(
            open_positions,
            lambda pos: self._check_position(pos, dry_run),
            'PositionMonitor/HFT',
            catch_runtime=True,
        )

    # ── Shared loop helper ────────────────────────────────────────────────────

    def _run_loop(self, open_positions, check_fn, label, catch_runtime=False) -> list:
        print(
            f"\n[{label}] Checking {len(open_positions)} open position(s) "
            f"(stop-loss = {self.stop_loss_multiplier:.1f}× premium) ..."
        )
        closed = []
        for pos in open_positions:
            try:
                result = check_fn(pos)
                if result:
                    closed.append(result)
            except RuntimeError as exc:
                if not catch_runtime:
                    raise
                _log.error(
                    "[HFT] position %s (%s/%s) skipped after Alpaca retries exhausted: %s",
                    pos.get('id', '?'), pos.get('symbol', '?'), pos.get('type', '?'), exc,
                )
        if closed:
            print(f"[{label}] ⚠  Closed {len(closed)} position(s).\n")
        else:
            print(f"[{label}] ✓  No triggers fired — all positions within limits.\n")
        return closed

    # ── Expiry settlement ─────────────────────────────────────────────────────

    def settle_expired(self) -> list[dict]:
        """
        Find all positions that passed their expiry date without being
        stopped out, compute their final P&L, and mark them CLOSED.

        This is purely a bookkeeping operation — no orders are placed.
        It runs regardless of dry_run because it only updates the local DB.

        P&L logic (per contract = × 100):
          Expired worthless  →  full premium collected
          Expired ITM        →  premium minus intrinsic value of the spread
          Unknown price      →  $0 P&L, still marked CLOSED

        Returns a list of position dicts that were settled.
        """
        expired = self.db.get_expired_unsettled_positions()
        if not expired:
            return []

        print(
            f"\n[PositionMonitor] Settling {len(expired)} expired position(s) ..."
        )

        settled = []
        for pos in expired:
            pnl = self._compute_expiry_pnl(pos)

            symbol  = pos['symbol']
            strat   = pos['type']
            expiry  = pos['expiry']
            premium = float(pos.get('premium', 0) or 0)
            outcome = 'worthless' if pnl is not None and pnl > 0 else \
                      'ITM loss'  if pnl is not None and pnl < 0 else 'unknown'

            if pnl is None:
                pnl = 0.0   # can't price → record zero, don't leave orphaned

            print(
                f"  {strat:6} {symbol:<6} expired {expiry}  "
                f"entry=${premium:.2f}  pnl=${pnl:+.2f}  [{outcome}]"
            )

            self.db.close_position(
                pos['id'],
                pnl,
                'EXPIRED',
                pnl_source='EXPIRED' if pnl != 0.0 else 'EXPIRED_ESTIMATE',
                pnl_verified=pnl != 0.0,
            )
            pos['close_pnl'] = pnl
            settled.append(pos)

        total_pnl = sum(p['close_pnl'] for p in settled)
        print(
            f"[PositionMonitor] Settled {len(settled)} position(s).  "
            f"Total P&L: ${total_pnl:+,.2f}\n"
        )
        return settled

    def _compute_expiry_pnl(self, pos: dict) -> Optional[float]:
        """
        Fetch the closing price on the expiry date (or nearest prior trading
        day) and calculate the option spread's final P&L per contract.

        Returns None if the price cannot be determined.
        """
        symbol  = pos['symbol']
        expiry  = pos['expiry']
        premium = float(pos.get('premium', 0) or 0)
        strat   = pos['type']
        legs    = self._parse_legs(pos)

        try:
            exp_date = datetime.date.fromisoformat(expiry)
            start = (exp_date - datetime.timedelta(days=5)).isoformat()
            end   = (exp_date + datetime.timedelta(days=1)).isoformat()
            close = self._data.get_historical_close(symbol, start, end)
            if close is None:
                return None
        except Exception:
            return None

        try:
            from src.risk_rules.intrinsic_value import compute_cost_to_close
            # For single-leg strategies the 'short_strike' may live on the
            # position root when the legs dict was stored without it.
            resolved = dict(legs)
            if not resolved.get('short_strike'):
                resolved['short_strike'] = pos.get('strike')

            cost = compute_cost_to_close(close, strat, resolved)
            if cost is None:
                return None

            contracts = int(pos.get('contracts') or 1)
            return round((premium - cost) * 100 * contracts, 2)

        except Exception:
            return None

    # ── Per-position logic ────────────────────────────────────────────────────

    def _check_position(self, pos: dict, dry_run: bool, prefetch=None) -> Optional[dict]:
        """
        Return closed-position dict if any exit trigger fires, else None.

        Two independent triggers:
          1. Stop-loss     — current mark > stop_loss_multiplier × entry_premium
          2. Gamma risk    — rising gamma/theta ratio + delta moving ITM
                             (aims to lock in remaining profit early)

        Delegates entirely to :class:`DataAdapter` which selects the correct
        data path (Alpaca snapshots for HFT, yfinance/Alpaca chain for non-HFT).

        Raises
        ------
        RuntimeError
            **HFT only** — propagates from the adapter when Alpaca fails after
            all retries.  Caught and logged by :meth:`_run_loop`.
        """
        entry_premium = pos.get('premium', 0) or 0
        if entry_premium <= 0:
            return None
        if self._within_new_position_grace(pos):
            return None

        # Adapter dispatches HFT vs non-HFT internally.
        # HFT failure raises RuntimeError (propagates to _run_loop).
        # Non-HFT failure returns put_map=None.
        # When prefetch is provided (batch mode), no API calls are made here.
        chain = self._data.get_position_chain(
            pos, self._risk_service._get_position_leg_specs,
            self._risk_service._build_osi_symbol,
            prefetch=prefetch,
        )
        if chain.put_map is None:
            return None

        # Use conservative (ask) prices so the stop-loss trigger accounts for
        # the actual cost to close, not just mid — avoids triggering on a mark
        # that can't be filled, then failing to execute at that price.
        current_mark = self._risk_service.compute_mark(pos, chain, conservative=True)
        if current_mark is None:
            if chain.has_broker_greeks:
                raise RuntimeError(
                    f"HFT: cannot compute mark for pos {pos.get('id')} "
                    f"({pos.get('type')}) — put_map={list(chain.put_map)}, "
                    f"call_map={list(chain.call_map)}"
                )
            return None

        label = '[HFT] ' if chain.has_broker_greeks else ''
        return self._apply_triggers(
            pos, dry_run, entry_premium, current_mark, chain.spot,
            lambda ep, pnl: self._check_gamma_risk_unified(pos, chain, chain.spot, ep, pnl),
            metrics_fn=lambda: self._risk_service._get_greek_risk_data_unified(pos, chain, chain.spot),
            chain=chain,
            label_prefix=label,
        )

    def _check_position_hft(self, pos: dict, dry_run: bool) -> Optional[dict]:
        """
        HFT variant of :meth:`_check_position`.

        Backward-compatible entry-point kept so that tests can mock this
        method independently.  Delegates directly to :meth:`_check_position`
        which dispatches via the :class:`DataAdapter` when HFT mode is active.

        Raises ``RuntimeError`` on Alpaca failure — caller (``run_hft``)
        catches and logs it.
        """
        return self._check_position(pos, dry_run)

    def _within_new_position_grace(self, pos: dict) -> bool:
        """Skip close triggers for freshly opened positions while quotes settle."""
        grace_minutes = self._new_position_grace_minutes
        if grace_minutes <= 0:
            return False

        ts_raw = (
            pos.get('filled_at')
            or pos.get('status_updated_at')
            or pos.get('timestamp')
            or pos.get('created_at')
        )
        if not ts_raw:
            return False

        try:
            opened_at = datetime.datetime.fromisoformat(str(ts_raw))
        except ValueError:
            return False

        age = datetime.datetime.now(opened_at.tzinfo) - opened_at
        if age < datetime.timedelta(minutes=grace_minutes):
            _log.debug(
                "[PositionMonitor] Skipping fresh %s %s (id=%s) during %.1f minute grace window "
                "(age %.1f seconds).",
                pos.get('type', '?'),
                pos.get('symbol', '?'),
                pos.get('id', '?'),
                grace_minutes,
                age.total_seconds(),
            )
            return True
        return False

    def _execute_close(
        self, pos: dict, current_mark: float, pnl_dollars: float,
        reason_tag: str, dry_run: bool,
    ) -> Optional[dict]:
        """
        Two-phase close: mark PENDING_CLOSE → submit order → CLOSED or roll back.

        In dry-run mode the DB is updated directly (no broker call, no two-phase
        needed) so existing tests and paper-trading behaviour are unchanged.

        Returns the enriched pos dict on success, or None if the broker order
        failed (position remains EXECUTED and will be retried next cycle).
        """
        trade_id = pos['id']

        if dry_run:
            self.db.close_position(trade_id, pnl_dollars, 'DRY_RUN_CLOSE')
            pos['close_pnl']      = pnl_dollars
            pos['close_order_id'] = 'DRY_RUN_CLOSE'
            pos['reason_tag']     = reason_tag
            self._ml_exit_confirmations.pop(trade_id, None)
            return pos

        from src.position_lifecycle import PositionLifecycleService
        result = PositionLifecycleService(self.db, self.executor).close_position(
            pos,
            limit_price=current_mark,
            pnl=pnl_dollars,
            dry_run=False,
            reason=reason_tag,
        )
        if not result.success:
            _log.warning(
                "[PositionMonitor] Close order failed for %s %s (id=%s): %s",
                pos.get('type'), pos.get('symbol'), trade_id, result.error,
            )
            return None

        pos['close_pnl']      = pnl_dollars
        pos['close_order_id'] = result.order_id
        pos['reason_tag']     = reason_tag
        self._ml_exit_confirmations.pop(trade_id, None)
        return pos

    # ── Shared trigger evaluation helper ─────────────────────────────────────

    def _apply_triggers(
        self, pos: dict, dry_run: bool,
        entry_premium: float, current_mark: float, spot: Optional[float],
        gamma_risk_fn,
        metrics_fn=None,
        chain=None,
        label_prefix: str = '',
    ) -> Optional[dict]:
        pnl_per_share = entry_premium - current_mark
        loss          = -pnl_per_share
        trig          = self.stop_loss_multiplier * entry_premium
        cached_metrics = None
        metrics_loaded = False

        def _load_metrics():
            nonlocal cached_metrics, metrics_loaded
            if not metrics_loaded:
                cached_metrics = metrics_fn() if metrics_fn is not None else None
                metrics_loaded = True
            return cached_metrics

        symbol = pos['symbol']; strat = pos['type']; expiry = pos['expiry']
        status_str = (
            f"  {label_prefix}{strat:6} {symbol:<6} {expiry}  "
            f"entry=${entry_premium:.2f}  mark=${current_mark:.2f}  "
            f"loss=${loss:+.2f}"
        )

        contracts = int(pos.get('contracts') or 1)
        # Trigger 1: ML exit-risk model (primary proactive drawdown guard)
        if self._ml_exit_risk.is_active():
            metrics = _load_metrics()
            risk = metrics[1] if metrics is not None else None
            score_payload = self._ml_exit_risk.score_position(
                pos,
                current_mark=current_mark,
                spot=spot,
                risk=risk,
                chain=chain,
            )
            if score_payload is not None:
                pos.update(score_payload)
                confirmation_count = self._track_ml_confirmation(pos, score_payload)
                pos['ml_exit_risk_confirmation_count'] = confirmation_count
                pos['ml_exit_risk_confirmations_required'] = (
                    self._ml_exit_risk.confirmations_required
                )
                if score_payload.get('ml_exit_risk_should_trigger'):
                    required = self._ml_exit_risk.confirmations_required
                    if confirmation_count >= required:
                        pnl_dollars = round(pnl_per_share * 100 * contracts, 2)
                        tag = "[DRY RUN]" if dry_run else "[LIVE]"
                        print(
                            f"{status_str}  ml_score={score_payload['ml_exit_risk_score']:.3f}"
                            f"  confirmations={confirmation_count}/{required}"
                            f"  → ML-RISK TRIGGERED {tag}"
                        )
                        pos['entry_premium'] = entry_premium
                        pos['current_mark'] = current_mark
                        pos['pnl_per_share'] = pnl_per_share
                        if metrics is not None:
                            dte, risk = metrics
                            pos['dte'] = dte
                            pos['ratio'] = risk['gamma_theta_ratio']
                            pos['short_delta'] = abs(risk['net_short_delta'])
                            pos['risk_score'] = risk['risk_score']
                        return self._execute_close(
                            pos,
                            current_mark,
                            pnl_dollars,
                            self._ml_exit_risk.reason_tag,
                            dry_run,
                        )

        # Trigger 2: profit-take (happy-path exit — lock in captured premium)
        pt_signal = self._profit_take_rule.evaluate(entry_premium, current_mark, pnl_per_share, spot, pos)
        if pt_signal:
            pnl_dollars  = round(pnl_per_share * 100 * contracts, 2)
            captured_pct = round(pnl_per_share / entry_premium * 100, 1)
            tag = "[DRY RUN]" if dry_run else "[LIVE]"
            print(f"{status_str}  profit_captured={captured_pct}%  → PROFIT-TAKE TRIGGERED {tag}")
            pos['entry_premium']      = entry_premium
            pos['current_mark']       = current_mark
            pos['pnl_per_share']      = pnl_per_share
            pos['profit_captured_pct'] = captured_pct
            pos['profit_take_pct']    = self._pt_pct
            return self._execute_close(pos, current_mark, pnl_dollars, 'PROFIT_TAKE', dry_run)

        # Trigger 3: deterministic stop-loss fallback
        sl_signal = self._stop_loss_rule.evaluate(entry_premium, current_mark, pnl_per_share, spot, pos)
        if sl_signal:
            pnl_dollars = round(pnl_per_share * 100 * contracts, 2)
            tag = "[DRY RUN]" if dry_run else "[LIVE]"
            print(f"{status_str}  → STOP-LOSS TRIGGERED {tag}")
            pos['entry_premium'] = entry_premium
            pos['current_mark']  = current_mark
            pos['pnl_per_share'] = pnl_per_share
            # Enrich with risk metrics so the position-closed email always
            # shows the "Risk Metrics at Close" section with real values.
            if metrics_fn is not None:
                raw = _load_metrics()
                if raw is not None:
                    dte, risk = raw
                    pos['dte']         = dte
                    pos['ratio']       = risk['gamma_theta_ratio']
                    pos['short_delta'] = abs(risk['net_short_delta'])
                    pos['risk_score']  = risk['risk_score']
            return self._execute_close(pos, current_mark, pnl_dollars, 'STOP_LOSS', dry_run)

        # Trigger 4: gamma risk (legacy; disabled in live config)
        if self._gamma_risk_enabled and spot is not None:
            gr_result = gamma_risk_fn(entry_premium, pnl_per_share)
            if gr_result is not None:
                reason, extras, gr_metrics = gr_result
                pnl_dollars = round(pnl_per_share * 100 * contracts, 2)
                tag = "[DRY RUN]" if dry_run else "[LIVE]"
                print(f"{status_str}  {extras}  → GAMMA-RISK TRIGGERED {tag} ({reason})")
                pos['entry_premium'] = entry_premium
                pos['current_mark']  = current_mark
                pos['pnl_per_share'] = pnl_per_share
                pos['ratio']         = gr_metrics.get('ratio')
                pos['short_delta']   = gr_metrics.get('short_delta')
                pos['risk_score']    = gr_metrics.get('risk_score')
                pos['dte']           = gr_metrics.get('dte')
                return self._execute_close(pos, current_mark, pnl_dollars, 'GAMMA_RISK', dry_run)

        print(f"{status_str}  → OK")
        return None

    def _track_ml_confirmation(self, pos: dict, score_payload: dict) -> int:
        trade_id = int(pos.get('id') or 0)
        if trade_id <= 0:
            return 0
        if score_payload.get('ml_exit_risk_should_trigger'):
            count = self._ml_exit_confirmations.get(trade_id, 0) + 1
            self._ml_exit_confirmations[trade_id] = count
            return count
        self._ml_exit_confirmations.pop(trade_id, None)
        return 0

    # ── Gamma risk check ──────────────────────────────────────────────────────

    def _eval_gamma_trigger(
        self, risk: dict, dte: int, entry_premium: float, pnl_per_share: float
    ) -> Optional[tuple]:
        from src.risk_rules.gamma_risk import eval_gamma_trigger
        return eval_gamma_trigger(risk, dte, entry_premium, pnl_per_share,
                                  self._gr_ratio_threshold, self._gr_min_delta,
                                  self._gr_min_profit_pct, self._gr_urgent_delta)

    # ── Unified gamma-risk helpers ────────────────────────────────────────────

    def _get_greek_risk_data_unified(
        self, pos: dict, chain, spot: Optional[float]
    ) -> Optional[tuple]:
        """
        Compute greek risk data from either broker greeks (HFT) or IV (non-HFT).

        Returns ``(dte, risk_dict)`` or ``None``.
        """
        from src.greeks import position_risk_score_from_greeks as _prs_g
        try:
            dte = (datetime.date.fromisoformat(pos['expiry']) - datetime.date.today()).days
            if dte <= 0:
                return None
            if chain.has_broker_greeks:
                greeks_legs = self._build_greeks_legs_from_snapshots(
                    pos, chain.leg_specs, chain.osi_map, chain.snapshots
                )
                if not greeks_legs:
                    _log.warning(
                        "[HFT] pos %s (%s/%s): all legs missing broker greeks — "
                        "gamma-risk check skipped; price-based stop-loss still active",
                        pos.get('id', '?'), pos.get('symbol', '?'), pos.get('type', '?'),
                    )
                    return None
                risk = _prs_g(greeks_legs)
            else:
                greeks_legs = self._build_greeks_legs(pos, chain.put_map, chain.call_map)
                if not greeks_legs:
                    return None
                risk = position_risk_score(spot, greeks_legs, dte)
            return dte, risk
        except Exception as exc:
            _log.error(
                "[PositionMonitor] Greeks error for %s %s: %s",
                pos.get('symbol', '?'), pos.get('expiry', '?'), exc, exc_info=True,
            )
        return None

    def _check_gamma_risk_unified(
        self, pos: dict, chain, spot: Optional[float],
        entry_premium: float, pnl_per_share: float,
    ) -> Optional[tuple]:
        """Unified gamma-risk check (HFT broker greeks or IV-based)."""
        result = self._risk_service._get_greek_risk_data_unified(pos, chain, spot)
        if result is None:
            return None
        dte, risk = result
        return self._eval_gamma_trigger(risk, dte, entry_premium, pnl_per_share)

    def _build_greeks_legs(
        self, pos: dict, put_map: dict, call_map: dict
    ) -> list[dict]:
        """
        Build the legs list consumed by position_risk_score().
        Each entry: {'strike', 'iv', 'option_type', 'position'}
        Reuses _get_position_leg_specs() for strategy dispatch; only adds IV lookup.
        """
        def _iv(strike: float, opt_type: str) -> Optional[float]:
            m   = put_map if opt_type == 'put' else call_map
            row = m.get(float(strike))
            if row is None:
                return None
            raw = (row.get('impliedVolatility') if hasattr(row, 'get')
                   else getattr(row, 'impliedVolatility', None))
            v = float(raw) if raw is not None else 0.0
            return v if v > 0 else None

        legs = []
        for strike, opt_type, position_side in self._get_position_leg_specs(pos):
            iv = _iv(strike, opt_type)
            if iv is not None:
                legs.append({
                    'strike':      strike,
                    'iv':          iv,
                    'option_type': opt_type,
                    'position':    position_side,
                })
            else:
                _log.warning(
                    "[PositionMonitor] IV missing for pos %s %s strike=%.2f %s "
                    "— leg excluded from greek risk calculation",
                    pos.get('symbol', '?'), pos.get('type', '?'), strike, opt_type,
                )
        return legs

    # ── HFT helpers (Alpaca-only, no yfinance) ────────────────────────────────

    @staticmethod
    def _build_osi_symbol(symbol: str, expiry: str, strike: float, opt_type: str) -> str:
        """
        Construct an OSI option symbol string.

        Format: ``{ROOT}{YYMMDD}{C|P}{STRIKE*1000 zero-padded to 8 digits}``
        Example: AAPL240119P00200000 = AAPL put @ $200, expiry 2024-01-19
        """
        yy = expiry[2:4]; mm = expiry[5:7]; dd = expiry[8:10]
        cp = 'C' if opt_type.lower() == 'call' else 'P'
        strike_int = int(round(float(strike) * 1000))
        return f"{symbol}{yy}{mm}{dd}{cp}{strike_int:08d}"

    def _get_position_leg_specs(self, pos: dict) -> list[tuple[float, str, str]]:
        """
        Return ``[(strike, opt_type, position_side), …]`` for every leg in *pos*.

        Delegates to :func:`src.risk_rules.leg_specs.get_position_leg_specs`.
        """
        from src.risk_rules.leg_specs import get_position_leg_specs
        return get_position_leg_specs(pos)

    def _fetch_chain_data_hft(
        self, pos: dict
    ) -> tuple:
        """
        HFT variant of ``_fetch_chain_data()``.

        Fetches Alpaca snapshots **only for the specific OSI contracts this
        position holds** — far more efficient than pulling the full chain.
        Also fetches spot price via Alpaca.

        Returns ``(spot, put_map, call_map, snapshots, osi_map, leg_specs)``.

        Raises ``RuntimeError`` on any Alpaca failure after retries —
        **no yfinance fallback**.

        Delegates to :meth:`DataAdapter._fetch_position_chain_hft` which
        centralises all HFT-specific Alpaca logic.
        """
        result = self._data._fetch_position_chain_hft(
            pos, self._get_position_leg_specs, self._build_osi_symbol
        )
        return (
            result.spot, result.put_map, result.call_map,
            result.snapshots, result.osi_map, result.leg_specs,
        )

    def _build_greeks_legs_from_snapshots(
        self,
        pos: dict,
        leg_specs: list,
        osi_map: dict,
        snapshots: dict,
    ) -> list[dict]:
        """
        Build ``legs_with_greeks`` from Alpaca snapshot rows.

        Returns a list of dicts compatible with
        ``position_risk_score_from_greeks()``:
          [{'delta': …, 'gamma': …, 'theta': …, 'position': 'short'|'long'}, …]

        Logs a warning and skips any leg whose snapshot lacks greeks data.
        """
        legs = []
        for strike, opt_type, position_side in leg_specs:
            osi = osi_map.get((strike, opt_type))
            if osi is None:
                continue
            row = snapshots.get(osi)
            if row is None:
                _log.warning(
                    "[HFT] snapshot not returned by Alpaca for %s (pos %s) — "
                    "leg skipped from gamma check",
                    osi, pos.get('id', '?'),
                )
                continue

            delta = row.get('delta')
            gamma = row.get('gamma')
            theta = row.get('theta')

            if delta is None or gamma is None or theta is None:
                _log.warning(
                    "[HFT] broker greeks are None for %s (pos %s) — "
                    "likely outside market hours or contract not priced by Alpaca; "
                    "leg skipped from gamma check",
                    osi, pos.get('id', '?'),
                )
                continue

            legs.append({
                'delta':    float(delta),
                'gamma':    float(gamma),
                'theta':    float(theta),
                'position': position_side,
            })
        return legs

    # ── Option pricing ────────────────────────────────────────────────────────

    def _get_current_mark(self, pos: dict, conservative: bool = False) -> Optional[float]:
        """
        Compute the net cost to close the position.

        conservative=False (default) — mid prices for both legs; suitable for
            unrealized P&L display and stop-loss monitoring.
        conservative=True — realistic transaction prices:
            • SHORT option legs (buy-to-close) → ask price
            • LONG  option legs (sell-to-close) → bid price
            Use this when computing realized P&L at close time.

        For credit spreads: cost_to_close = short_leg_price - long_leg_price.
        Entry premium collected minus cost_to_close = current P&L per share.
        """
        chain = self._data.get_position_chain(
            pos, self._risk_service._get_position_leg_specs,
            self._risk_service._build_osi_symbol,
        )
        if chain.put_map is None:
            return None
        return self._risk_service.compute_mark(pos, chain, conservative)

    @staticmethod
    def _compute_mark_from_maps(strat, legs, pos, put_map, call_map, conservative=False):
        from src.risk_rules.mark import compute_mark_from_maps
        return compute_mark_from_maps(strat, legs, pos, put_map, call_map, conservative)

    # ── Risk snapshot ─────────────────────────────────────────────────────────

    @staticmethod
    def _enrich_pnl(pos: dict, entry_premium: float, current_mark: float, sl_multiplier: float) -> None:
        pnl_per_share   = entry_premium - current_mark
        # Stop fires when (mark - entry) > multiplier × entry, i.e. mark > entry × (1 + multiplier).
        # Proximity = fraction of that allowed loss already used; hits 100% exactly when stop fires.
        stop_threshold  = (1.0 + sl_multiplier) * entry_premium   # mark level that triggers stop
        stop_proximity  = (current_mark - entry_premium) / (sl_multiplier * entry_premium)
        profit_captured = pnl_per_share / entry_premium
        contracts = int(pos.get('contracts') or 1)
        pos['current_mark']        = round(current_mark, 2)
        pos['pnl_per_share']       = round(pnl_per_share, 4)
        pos['pnl_dollars']         = round(pnl_per_share * 100 * contracts, 2)
        pos['stop_threshold']      = round(stop_threshold, 2)
        pos['stop_proximity_pct']  = round(stop_proximity * 100, 1)
        pos['profit_captured_pct'] = round(profit_captured * 100, 1)

    @staticmethod
    def _enrich_greeks(pos: dict, risk: dict, dte: int) -> None:
        pos['gamma_theta_ratio'] = round(risk['gamma_theta_ratio'], 3)
        pos['net_short_delta']   = round(abs(risk['net_short_delta']), 3)
        pos['risk_score']        = round(risk['risk_score'], 3)
        pos['dte']               = dte

    def get_risk_snapshot(self) -> list[dict]:
        """
        Return all open positions enriched with current mark-to-market, P&L,
        greeks, and risk-level classification — without closing anything.

        Each returned dict has (where available):
          current_mark, pnl_per_share, pnl_dollars,
          stop_threshold, stop_proximity_pct, profit_captured_pct,
          gamma_theta_ratio, net_short_delta, risk_score, dte, spot,
          has_broker_greeks, risk_level  ('SAFE' | 'WATCH' | 'CAUTION' | 'CRITICAL')

        Delegates to :class:`~src.risk_service.PositionRiskService` so that
        the dashboard and the daemon always produce identical metrics.

        In HFT mode (``hft_mode: true`` in config) data comes entirely from
        Alpaca snapshots using broker-supplied greeks.  Failures are logged
        at WARNING and the position is still included with partial data.
        """
        from src.risk_ml import classify_ml_exit_risk_level

        open_positions = self.db.get_open_positions()
        results        = []

        for pos in open_positions:
            try:
                enriched = self._risk_service.enrich_position(dict(pos))
                self._ml_exit_risk.annotate_position(enriched)
                enriched['risk_level'] = classify_ml_exit_risk_level(
                    enriched.get('ml_exit_risk_score'),
                    enriched.get('ml_exit_risk_threshold'),
                    guard_reason=enriched.get('ml_exit_risk_guard_reason'),
                )
                results.append(enriched)
            except RuntimeError as exc:
                _log.warning(
                    "[HFT] risk_snapshot: skipping pos %s (%s/%s) — %s",
                    pos.get('id'), pos.get('symbol'), pos.get('type'), exc,
                )
                p = dict(pos)
                p['spot']      = None
                p['risk_level'] = 'WATCH' if self._ml_exit_risk.is_active() else 'SAFE'
                results.append(p)

        _order = {'CRITICAL': 0, 'CAUTION': 1, 'WATCH': 2, 'SAFE': 3}
        results.sort(key=lambda p: _order.get(p.get('risk_level', 'SAFE'), 3))
        return results

    @staticmethod
    def _parse_legs(pos: dict) -> dict:
        """Deserialise ``pos['legs']`` to a plain dict.

        Delegates to :func:`src.risk_rules.leg_specs.parse_legs`.
        """
        from src.risk_rules.leg_specs import parse_legs
        return parse_legs(pos)


# ── Risk classification ────────────────────────────────────────────────────────

def _classify_risk_level(pos: dict, stop_loss_multiplier: float) -> str:
    from src.risk_rules.classify import classify_risk_level
    return classify_risk_level(pos, stop_loss_multiplier)


def _within_market_hours(
    open_t: str, close_t: str, tz_name: str, weekdays_only: bool
) -> bool:
    """Return True if *now* falls within the configured market window."""
    tz = None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:
        try:
            import pytz
            tz = pytz.timezone(tz_name)
        except Exception:
            pass  # last resort: system local time

    now = datetime.datetime.now(tz)
    if weekdays_only and now.weekday() >= 5:   # Saturday=5, Sunday=6
        return False

    hh_o, mm_o = (int(p) for p in open_t.split(':'))
    hh_c, mm_c = (int(p) for p in close_t.split(':'))
    t_open  = now.replace(hour=hh_o, minute=mm_o, second=0, microsecond=0)
    t_close = now.replace(hour=hh_c, minute=mm_c, second=0, microsecond=0)
    return t_open <= now < t_close
