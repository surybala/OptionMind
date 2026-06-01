"""
PortfolioRiskService
====================

Account-level short-gamma guardrail for pre-trade sizing.

The per-position monitor decides when an individual open position should be
closed.  This service answers a different question before opening anything:
"If we add this pick to the existing book, does the whole portfolio become too
fragile to an index gap plus IV expansion?"
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional

from src.greeks import bs_greeks
from src.risk_service import PositionRiskService

_log = logging.getLogger('optionwheel')


@dataclass
class GreekExposure:
    symbol: str
    expiry: str
    dte: int
    spot: float
    contracts: int
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    source: str
    ref: str


class PortfolioRiskService:
    """Evaluate portfolio-level gamma/vega stress before adding trades."""

    def __init__(
        self,
        config: dict,
        position_risk_service: Optional[PositionRiskService] = None,
    ) -> None:
        self.config = config
        self._cfg = config.get('risk_parameters', {}).get('portfolio_gamma_risk', {})
        self._position_risk = position_risk_service or PositionRiskService.from_config(config)

    def enabled(self) -> bool:
        return bool(self._cfg.get('enabled', True))

    def filter_picks(
        self,
        picks: list[dict],
        positions: list[dict],
        account_capital: Optional[float],
        rejection_sink: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Return picks resized/rejected by account-level gamma stress limits.

        Picks are processed in score order and each accepted pick is added to
        the running exposure before evaluating the next pick.
        """
        if not self.enabled() or not picks:
            return picks
        account_capital = self._coerce_capital(account_capital)
        if account_capital <= 0:
            _log.warning("Portfolio gamma gate skipped: account capital is not configured.")
            return picks

        base_exposures = self._position_exposures(positions)
        if positions and self._fail_closed() and len(base_exposures) < len(positions):
            _log.warning(
                "Portfolio gamma gate: usable Greeks for %d/%d open/pending-close "
                "position(s); rejecting new picks because fail_closed=true.",
                len(base_exposures), len(positions),
            )
            return []

        accepted: list[dict] = []
        running = list(base_exposures)
        rejected = reduced = 0
        for pick in sorted(picks, key=lambda x: x.get('score', 0.0), reverse=True):
            requested_qty = max(1, int(pick.get('quantity') or 1))
            requested_qty = self._cap_complex_spread_quantity(
                pick, requested_qty, running, account_capital
            )
            best_qty = 0
            best_summary = None
            last_summary = None
            for qty in range(requested_qty, 0, -1):
                candidate = self._pick_exposure(pick, qty)
                if candidate is None:
                    best_qty = qty if not self._fail_closed() else 0
                    best_summary = self._summarize(running, account_capital)
                    break
                summary = self._summarize(running + [candidate], account_capital)
                last_summary = summary
                if not summary['violations']:
                    best_qty = qty
                    best_summary = summary
                    break
            if best_qty <= 0:
                rejected += 1
                if rejection_sink is not None:
                    payload = dict(pick)
                    payload["risk_gate"] = "Portfolio gamma risk"
                    payload["reject_reason"] = self._violation_text(last_summary or best_summary)
                    payload["portfolio_violations"] = list((last_summary or best_summary or {}).get("violations", []))
                    payload["portfolio_violation_codes"] = list((last_summary or best_summary or {}).get("violation_codes", []))
                    rejection_sink.append(payload)
                _log.info(
                    "Portfolio gamma gate: rejected %s %s; %s",
                    pick.get('strategy'), pick.get('symbol'),
                    self._violation_text(last_summary or best_summary),
                )
                continue

            if best_qty < requested_qty:
                reduced += 1
                _log.info(
                    "Portfolio gamma gate: reduced %s %s quantity %d -> %d; %s",
                    pick.get('strategy'), pick.get('symbol'), requested_qty, best_qty,
                    self._summary_text(best_summary),
                )
            pick['quantity'] = best_qty
            self._annotate_pick(pick, best_summary)
            accepted.append(pick)
            candidate = self._pick_exposure(pick, best_qty)
            if candidate is not None:
                running.append(candidate)

        final_summary = self._summarize(running, account_capital)
        _log.info(
            "Portfolio gamma gate: kept %d/%d pick(s), reduced %d, rejected %d. %s",
            len(accepted), len(picks), reduced, rejected, self._summary_text(final_summary),
        )
        return accepted

    def _cap_complex_spread_quantity(
        self,
        pick: dict,
        requested_qty: int,
        running: list[GreekExposure],
        account_capital: float,
    ) -> int:
        cfg = self._cfg.get('complex_spread_quantity_cap', {})
        if not cfg.get('enabled', True):
            return requested_qty

        strat = str(pick.get('strategy') or '').upper()
        strategies = {str(s).upper() for s in cfg.get('strategies', ['IC', 'IFLY'])}
        if strat not in strategies or requested_qty <= 1:
            return requested_qty

        max_qty = max(1, int(cfg.get('max_quantity', 1)))
        threshold = float(cfg.get('symbol_stress_threshold_pct', 0.50))
        limits = self._limits(account_capital)
        symbol_limit = float(limits.get('symbol_stress') or 0.0)
        if not limits.get('symbol_stress_enabled') or symbol_limit <= 0:
            return min(requested_qty, max_qty)

        candidate = self._pick_exposure(pick, 1)
        if candidate is None:
            return requested_qty

        symbol_exposures = [
            exposure for exposure in running
            if exposure.symbol == candidate.symbol
        ] + [candidate]
        symbol_loss = self._worst_loss(
            symbol_exposures,
            self._shock_values(),
            float(self._cfg.get('iv_shock_points', 10.0)) / 100.0,
        )
        if symbol_loss >= symbol_limit * threshold:
            _log.info(
                "Portfolio gamma gate: capped %s %s quantity %d -> %d; "
                "symbol stress $%.0f >= %.0f%% of $%.0f cap.",
                strat, pick.get('symbol'), requested_qty, max_qty,
                symbol_loss, threshold * 100.0, symbol_limit,
            )
            return min(requested_qty, max_qty)
        return requested_qty

    def summarize_positions(
        self,
        positions: list[dict],
        account_capital: Optional[float],
    ) -> dict:
        """Expose a portfolio summary for dashboards/tests."""
        return self._summarize(
            self._position_exposures(positions),
            self._coerce_capital(account_capital),
        )

    def _position_exposures(self, positions: list[dict]) -> list[GreekExposure]:
        if not positions:
            return []
        prefetch = None
        data = getattr(self._position_risk, '_data', None)
        if data is not None and getattr(data, 'is_hft', lambda: False)():
            try:
                prefetch = data.batch_prefetch_hft(
                    positions,
                    PositionRiskService._get_position_leg_specs,
                    PositionRiskService._build_osi_symbol,
                )
            except Exception as exc:
                _log.warning("Portfolio gamma gate: HFT prefetch failed: %s", exc)
                if self._fail_closed():
                    return []

        exposures: list[GreekExposure] = []
        for pos in positions:
            try:
                enriched = self._position_risk.enrich_position(pos, prefetch=prefetch)
            except Exception as exc:
                _log.warning(
                    "Portfolio gamma gate: risk enrichment failed for %s %s: %s",
                    pos.get('type'), pos.get('symbol'), exc,
                )
                continue
            exposure = self._exposure_from_enriched_position(enriched)
            if exposure is not None:
                exposures.append(exposure)
        return exposures

    def _exposure_from_enriched_position(self, pos: dict) -> Optional[GreekExposure]:
        try:
            spot = float(pos.get('spot') or 0)
            gamma = float(pos.get('net_gamma'))
            theta = float(pos.get('net_theta'))
        except (TypeError, ValueError):
            return None
        if spot <= 0:
            return None
        return GreekExposure(
            symbol=str(pos.get('symbol', '?')),
            expiry=str(pos.get('expiry', '')),
            dte=self._dte(pos.get('expiry')),
            spot=spot,
            contracts=max(1, int(pos.get('contracts') or 1)),
            net_delta=float(pos.get('net_delta') or 0.0),
            net_gamma=gamma,
            net_theta=theta,
            net_vega=float(pos.get('net_vega') or 0.0),
            source='position',
            ref=str(pos.get('id') or pos.get('symbol') or '?'),
        )

    def _pick_exposure(self, pick: dict, quantity: int) -> Optional[GreekExposure]:
        try:
            spot = float(pick.get('current_price') or 0)
        except (TypeError, ValueError):
            spot = 0.0
        dte = self._dte(pick.get('expiry'))
        if spot <= 0 or dte <= 0:
            return None

        legs = self._pick_greek_legs(pick)
        if not legs:
            return None
        net_delta = net_gamma = net_theta = net_vega = 0.0
        for leg in legs:
            g = bs_greeks(
                spot,
                float(leg['strike']),
                float(leg['iv']),
                dte,
                str(leg['option_type']),
            )
            sign = -1.0 if leg['position'] == 'short' else 1.0
            net_delta += sign * g['delta']
            net_gamma += sign * g['gamma']
            net_theta += sign * g['theta']
            net_vega += sign * g['vega']
        return GreekExposure(
            symbol=str(pick.get('symbol', '?')),
            expiry=str(pick.get('expiry', '')),
            dte=dte,
            spot=spot,
            contracts=max(1, int(quantity)),
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_theta=net_theta,
            net_vega=net_vega,
            source='pick',
            ref=f"{pick.get('strategy')} {pick.get('symbol')}",
        )

    def _pick_greek_legs(self, pick: dict) -> list[dict]:
        strat = str(pick.get('strategy') or '').upper()
        default_iv = float(self._cfg.get('default_iv', 0.25))

        def _iv(*names):
            for name in names:
                value = pick.get(name)
                if value not in (None, ''):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        return value
            return default_iv

        legs: list[dict] = []

        def _add(strike, opt_type: str, side: str, iv: float) -> None:
            if strike is not None:
                legs.append({
                    'strike': float(strike),
                    'option_type': opt_type,
                    'position': side,
                    'iv': iv,
                })

        if strat == 'PCS':
            _add(pick.get('short_strike') or pick.get('short_put'), 'put', 'short',
                 _iv('short_iv', 'short_put_iv', 'iv'))
            _add(pick.get('long_strike') or pick.get('long_put'), 'put', 'long',
                 _iv('long_iv', 'long_put_iv', 'iv'))
        elif strat == 'CCS':
            _add(pick.get('short_strike') or pick.get('short_call'), 'call', 'short',
                 _iv('short_iv', 'short_call_iv', 'iv'))
            _add(pick.get('long_strike') or pick.get('long_call'), 'call', 'long',
                 _iv('long_iv', 'long_call_iv', 'iv'))
        elif strat in ('IC', 'IFLY'):
            _add(pick.get('short_put'), 'put', 'short', _iv('short_put_iv', 'iv'))
            _add(pick.get('long_put'), 'put', 'long', _iv('long_put_iv', 'iv'))
            _add(pick.get('short_call'), 'call', 'short', _iv('short_call_iv', 'iv'))
            _add(pick.get('long_call'), 'call', 'long', _iv('long_call_iv', 'iv'))
        elif strat == 'CSP':
            _add(pick.get('short_strike') or pick.get('short_put'), 'put', 'short',
                 _iv('short_iv', 'short_put_iv', 'iv'))
        elif strat == 'CC':
            _add(pick.get('short_strike') or pick.get('short_call'), 'call', 'short',
                 _iv('short_iv', 'short_call_iv', 'iv'))
        elif strat == 'STRANGLE':
            _add(pick.get('short_put'), 'put', 'short', _iv('short_put_iv', 'iv'))
            _add(pick.get('short_call'), 'call', 'short', _iv('short_call_iv', 'iv'))
        return legs

    def _summarize(self, exposures: list[GreekExposure], account_capital: float) -> dict:
        shocks = self._shock_values()
        iv_shock = float(self._cfg.get('iv_shock_points', 10.0)) / 100.0
        scenario_losses = {}
        for shock in shocks:
            pnl = 0.0
            for exposure in exposures:
                d_s = exposure.spot * (shock / 100.0)
                pnl += (
                    exposure.net_delta * d_s
                    + 0.5 * exposure.net_gamma * d_s * d_s
                    + exposure.net_vega * iv_shock
                ) * 100.0 * exposure.contracts
            scenario_losses[f"{shock:+g}%"] = round(max(0.0, -pnl), 2)

        worst_loss = max(scenario_losses.values(), default=0.0)
        one_pct_loss = max(
            loss for label, loss in scenario_losses.items()
            if label in ('+1%', '-1%')
        ) if scenario_losses else 0.0
        near_dte = int(self._cfg.get('near_expiry_dte', 2))
        near_loss = self._worst_loss(
            [e for e in exposures if e.dte <= near_dte], shocks, iv_shock
        )
        expiry_bucket_losses = {
            expiry: self._worst_loss(bucket, shocks, iv_shock)
            for expiry, bucket in self._by_expiry(exposures).items()
        }
        max_bucket_loss = max(expiry_bucket_losses.values(), default=0.0)
        symbol_bucket_losses = {
            symbol: self._worst_loss(bucket, shocks, iv_shock)
            for symbol, bucket in self._by_symbol(exposures).items()
        }
        max_symbol_loss = max(symbol_bucket_losses.values(), default=0.0)
        theta_dollars = round(sum(e.net_theta * 100.0 * e.contracts for e in exposures), 2)
        ratio = (
            float('inf')
            if theta_dollars <= 0 and one_pct_loss > 0
            else (one_pct_loss / theta_dollars if theta_dollars > 0 else 0.0)
        )

        limits = self._limits(account_capital)
        violations = []
        violation_codes: list[str] = []
        if worst_loss > limits['max_stress_loss']:
            violation_codes.append("worst_stress")
            violations.append(
                f"worst stress ${worst_loss:,.0f} > ${limits['max_stress_loss']:,.0f}"
            )
        if near_loss > limits['near_expiry_stress']:
            violation_codes.append("near_expiry_stress")
            violations.append(
                f"near-expiry stress ${near_loss:,.0f} > ${limits['near_expiry_stress']:,.0f}"
            )
        if limits['expiry_bucket_enabled'] and max_bucket_loss > limits['expiry_bucket']:
            violation_codes.append("expiry_bucket_stress")
            violations.append(
                f"expiry bucket stress ${max_bucket_loss:,.0f} > ${limits['expiry_bucket']:,.0f}"
            )
        if limits['symbol_stress_enabled'] and max_symbol_loss > limits['symbol_stress']:
            violation_codes.append("symbol_stress")
            violations.append(
                f"symbol stress ${max_symbol_loss:,.0f} > ${limits['symbol_stress']:,.0f}"
            )
        if (
            not limits['gamma_loss_to_theta_warning_only']
            and ratio > limits['gamma_loss_to_theta']
        ):
            violation_codes.append("gamma_loss_to_theta")
            violations.append(
                f"1% gamma loss/theta {ratio:.2f}x > {limits['gamma_loss_to_theta']:.2f}x"
            )

        return {
            'positions': len(exposures),
            'account_capital': account_capital,
            'scenario_losses': scenario_losses,
            'worst_stress_loss': round(worst_loss, 2),
            'one_pct_stress_loss': round(one_pct_loss, 2),
            'near_expiry_stress_loss': round(near_loss, 2),
            'max_expiry_bucket_loss': round(max_bucket_loss, 2),
            'max_symbol_stress_loss': round(max_symbol_loss, 2),
            'expiry_bucket_losses': expiry_bucket_losses,
            'symbol_bucket_losses': symbol_bucket_losses,
            'daily_theta': theta_dollars,
            'gamma_loss_to_daily_theta': (
                round(ratio, 4) if ratio != float('inf') else 'Infinity'
            ),
            'limits': limits,
            'violations': violations,
            'violation_codes': violation_codes,
        }

    def _worst_loss(
        self,
        exposures: list[GreekExposure],
        shocks: list[float],
        iv_shock: float,
    ) -> float:
        worst = 0.0
        for shock in shocks:
            pnl = 0.0
            for exposure in exposures:
                d_s = exposure.spot * (shock / 100.0)
                pnl += (
                    exposure.net_delta * d_s
                    + 0.5 * exposure.net_gamma * d_s * d_s
                    + exposure.net_vega * iv_shock
                ) * 100.0 * exposure.contracts
            worst = max(worst, max(0.0, -pnl))
        return round(worst, 2)

    @staticmethod
    def _by_expiry(exposures: list[GreekExposure]) -> dict[str, list[GreekExposure]]:
        buckets: dict[str, list[GreekExposure]] = {}
        for exposure in exposures:
            buckets.setdefault(exposure.expiry, []).append(exposure)
        return buckets

    @staticmethod
    def _by_symbol(exposures: list[GreekExposure]) -> dict[str, list[GreekExposure]]:
        buckets: dict[str, list[GreekExposure]] = {}
        for exposure in exposures:
            buckets.setdefault(exposure.symbol, []).append(exposure)
        return buckets

    def _limits(self, account_capital: float) -> dict:
        expiry_enabled = bool(self._cfg.get('expiry_bucket_cap_enabled', True))
        symbol_enabled = bool(self._cfg.get('symbol_stress_cap_enabled', True))
        gamma_theta_warning_only = bool(
            self._cfg.get('gamma_loss_to_theta_warning_only', False)
        )
        max_stress = max(
            account_capital * float(self._cfg.get('max_stress_loss_pct', 0.01)),
            float(self._cfg.get('min_stress_loss_dollars', 0.0) or 0.0),
        )
        near_stress = max(
            account_capital * float(self._cfg.get('max_near_expiry_stress_pct', 0.0025)),
            float(self._cfg.get('min_near_expiry_stress_dollars', 0.0) or 0.0),
        )
        symbol_stress = max(
            account_capital * float(self._cfg.get('max_symbol_stress_pct', 0.02)),
            float(self._cfg.get('min_symbol_stress_dollars', 0.0) or 0.0),
        )
        expiry_bucket = max(
            account_capital
            * float(self._cfg.get('max_expiry_bucket_pct', 0.40))
            * float(self._cfg.get('max_stress_loss_pct', 0.01)),
            float(self._cfg.get('min_expiry_bucket_dollars', 0.0) or 0.0),
        )
        return {
            'max_stress_loss': max_stress,
            'near_expiry_stress': near_stress,
            'expiry_bucket_enabled': expiry_enabled,
            'expiry_bucket': expiry_bucket if expiry_enabled else 0.0,
            'symbol_stress_enabled': symbol_enabled,
            'symbol_stress': symbol_stress if symbol_enabled else 0.0,
            'gamma_loss_to_theta': float(
                self._cfg.get('max_gamma_loss_to_daily_theta', 2.0)
            ),
            'gamma_loss_to_theta_warning_only': gamma_theta_warning_only,
        }

    def _shock_values(self) -> list[float]:
        raw = self._cfg.get('shock_moves_pct', [1, 2, 3, 5])
        values = sorted({abs(float(v)) for v in raw if float(v) > 0})
        out = []
        for value in values:
            out.extend([-value, value])
        return out or [-1.0, 1.0]

    def _coerce_capital(self, account_capital: Optional[float]) -> float:
        try:
            return float(account_capital or 0)
        except (TypeError, ValueError):
            return 0.0

    def _fail_closed(self) -> bool:
        return bool(self._cfg.get('fail_closed', True))

    @staticmethod
    def _dte(expiry) -> int:
        try:
            return (_dt.date.fromisoformat(str(expiry)) - _dt.date.today()).days
        except Exception:
            return 0

    @staticmethod
    def _summary_text(summary: Optional[dict]) -> str:
        if not summary:
            return "risk unavailable"
        ratio = summary['gamma_loss_to_daily_theta']
        ratio_s = "inf" if ratio == 'Infinity' else f"{float(ratio):.2f}x"
        return (
            f"worst=${summary['worst_stress_loss']:,.0f}, "
            f"symbol=${summary['max_symbol_stress_loss']:,.0f}, "
            f"1%=${summary['one_pct_stress_loss']:,.0f}, "
            f"near=${summary['near_expiry_stress_loss']:,.0f}, "
            f"theta=${summary['daily_theta']:,.0f}/day, "
            f"1%/theta={ratio_s}"
        )

    @classmethod
    def _violation_text(cls, summary: Optional[dict]) -> str:
        if not summary:
            return "risk unavailable"
        if not summary['violations']:
            return cls._summary_text(summary)
        return "; ".join(summary['violations'])

    @staticmethod
    def _annotate_pick(pick: dict, summary: Optional[dict]) -> None:
        if not summary:
            return
        pick['portfolio_worst_stress_loss'] = summary['worst_stress_loss']
        pick['portfolio_one_pct_stress_loss'] = summary['one_pct_stress_loss']
        pick['portfolio_max_symbol_stress_loss'] = summary['max_symbol_stress_loss']
        pick['portfolio_gamma_loss_to_daily_theta'] = summary['gamma_loss_to_daily_theta']
