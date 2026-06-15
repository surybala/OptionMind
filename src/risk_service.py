"""
PositionRiskService
===================
Single source of truth for per-position risk enrichment.

Both the position monitor daemon (PositionMonitor.get_risk_snapshot) and the
Flask dashboard (/api/risk-monitor) delegate to this service so that Greeks,
P&L metrics and risk-level classification are always computed identically.

HFT awareness
-------------
When ``hft_mode: true`` in config the DataAdapter returns
``chain.has_broker_greeks=True`` and broker-supplied delta/gamma/theta from
Alpaca snapshots are used directly when available. Missing snapshot Greeks
fall back to Black-Scholes estimates from snapshot implied volatility so the
portfolio gamma gate can keep running outside fully-priced broker conditions.
When ``hft_mode: false`` (default) Greek values are computed locally via
Black-Scholes from the implied volatility in the yfinance/Alpaca option chain.

Usage
-----
    # From config only (dashboard, scripts):
    svc = PositionRiskService.from_config(config)

    # Re-using an existing DataAdapter (PositionMonitor):
    svc = PositionRiskService(data_adapter, config)

    # Enrich a single position:
    enriched = svc.enrich_position(pos_dict)
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.market_data import DataAdapter

_log = logging.getLogger('optionwheel')


class PositionRiskService:
    """
    Enriches open-position dicts with live market data and risk metrics.

    Parameters
    ----------
    data_adapter :
        Configured :class:`~src.market_data.DataAdapter`.  Owns the
        HFT / non-HFT dispatch; the service never calls Alpaca or yfinance
        directly.
    config : dict
        Full application config dict.  Reads ``risk_parameters``.
    """

    def __init__(self, data_adapter: 'DataAdapter', config: dict) -> None:
        self._data = data_adapter
        self.config = config
        risk = config.get('risk_parameters', {})
        self.stop_loss_multiplier = float(risk.get('stop_loss_multiplier', 2.0))
        self._gamma_risk_enabled = bool((risk.get('gamma_risk', {}) or {}).get('enabled', True))

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> 'PositionRiskService':
        """
        Build a service from config alone (no existing DataAdapter needed).

        Creates a fresh :class:`~src.market_data.DataAdapter` via
        :meth:`DataAdapter.from_config`.
        """
        from src.market_data import DataAdapter
        return cls(DataAdapter.from_config(config), config)

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich_position(self, pos: dict, prefetch=None) -> dict:
        """
        Return *pos* (a copy) enriched with live market data and risk metrics.

        HFT-aware: uses Alpaca broker Greeks when ``hft_mode=True``, otherwise
        computes Greeks locally via Black-Scholes from implied volatility.

        Fields added where data is available
        -------------------------------------
        current_mark, pnl_per_share, pnl_dollars, stop_threshold,
        stop_proximity_pct, profit_captured_pct,
        gamma_theta_ratio, net_short_delta, risk_score, net_theta, net_gamma,
        net_delta, net_vega, dte, spot, has_broker_greeks, risk_level

        Raises
        ------
        RuntimeError
            HFT mode only — propagated from the adapter when Alpaca fails
            after all retries.  Callers should catch and handle gracefully.
        """
        pos = dict(pos)
        entry_premium = float(pos.get('premium', 0) or 0)

        # ── Fetch chain (HFT: Alpaca snapshots; non-HFT: yfinance/Alpaca chain)
        chain = self._data.get_position_chain(
            pos, self._get_position_leg_specs, self._build_osi_symbol,
            prefetch=prefetch,
        )

        # ── P&L enrichment ────────────────────────────────────────────────────
        if chain.put_map is not None and entry_premium > 0:
            current_mark = self._compute_mark_from_maps(
                pos['type'], self._parse_legs(pos), pos,
                chain.put_map, chain.call_map, conservative=False,
            )
            if current_mark is not None:
                self._enrich_pnl(pos, entry_premium, current_mark,
                                 self.stop_loss_multiplier)

        # ── Greeks enrichment (best-effort) ───────────────────────────────────
        if chain.spot is not None and chain.put_map is not None:
            try:
                result = self._get_greek_risk_data_unified(pos, chain, chain.spot)
                if result is not None:
                    dte, risk = result
                    self._enrich_greeks(pos, risk, dte)
            except Exception as exc:
                _log.debug(
                    "[risk_service] greeks failed for pos %s (%s): %s",
                    pos.get('id'), pos.get('symbol'), exc,
                )

        pos['spot']             = chain.spot
        pos['has_broker_greeks'] = chain.has_broker_greeks
        pos['risk_level']       = self._classify_risk_level(pos)
        return pos

    def compute_mark(
        self, pos: dict, chain, conservative: bool = False
    ) -> Optional[float]:
        """
        Compute the net cost-to-close from a pre-fetched chain result.

        conservative=False — mid prices (unrealized P&L display).
        conservative=True  — ask for short legs / bid for long legs
                             (realistic cost at close; use for stop-loss check).
        """
        if chain.put_map is None:
            return None
        return self._compute_mark_from_maps(
            pos['type'], self._parse_legs(pos), pos,
            chain.put_map, chain.call_map, conservative,
        )

    # ── Static helpers (strategy dispatch) ───────────────────────────────────

    @staticmethod
    def _get_position_leg_specs(pos: dict) -> list[tuple[float, str, str]]:
        """Return ``[(strike, opt_type, position_side), …]`` for *pos*."""
        from src.risk_rules.leg_specs import get_position_leg_specs
        return get_position_leg_specs(pos)

    @staticmethod
    def _build_osi_symbol(
        symbol: str, expiry: str, strike: float, opt_type: str
    ) -> str:
        """Construct an OSI option symbol string."""
        yy = expiry[2:4]; mm = expiry[5:7]; dd = expiry[8:10]
        cp = 'C' if opt_type.lower() == 'call' else 'P'
        strike_int = int(round(float(strike) * 1000))
        return f"{symbol}{yy}{mm}{dd}{cp}{strike_int:08d}"

    @staticmethod
    def _parse_legs(pos: dict) -> dict:
        """Deserialise ``pos['legs']`` to a plain dict."""
        from src.risk_rules.leg_specs import parse_legs
        return parse_legs(pos)

    # ── Mark computation ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_mark_from_maps(
        strat, legs, pos, put_map, call_map, conservative=False
    ) -> Optional[float]:
        from src.risk_rules.mark import compute_mark_from_maps
        return compute_mark_from_maps(strat, legs, pos, put_map, call_map, conservative)

    # ── P&L enrichment ────────────────────────────────────────────────────────

    @staticmethod
    def _enrich_pnl(
        pos: dict, entry_premium: float, current_mark: float,
        sl_multiplier: float,
    ) -> None:
        pnl_per_share   = entry_premium - current_mark
        stop_threshold  = (1.0 + sl_multiplier) * entry_premium
        stop_proximity  = (current_mark - entry_premium) / (sl_multiplier * entry_premium)
        profit_captured = pnl_per_share / entry_premium
        contracts       = int(pos.get('contracts') or 1)
        pos['current_mark']        = round(current_mark, 2)
        pos['pnl_per_share']       = round(pnl_per_share, 4)
        pos['pnl_dollars']         = round(pnl_per_share * 100 * contracts, 2)
        pos['stop_threshold']      = round(stop_threshold, 2)
        pos['stop_proximity_pct']  = round(stop_proximity * 100, 1)
        pos['profit_captured_pct'] = round(profit_captured * 100, 1)

    # ── Greeks enrichment ─────────────────────────────────────────────────────

    @staticmethod
    def _enrich_greeks(pos: dict, risk: dict, dte: int) -> None:
        pos['gamma_theta_ratio'] = round(risk['gamma_theta_ratio'], 3)
        pos['net_delta']         = round(risk.get('net_delta', 0.0), 6)
        pos['net_short_delta']   = round(abs(risk['net_short_delta']), 3)
        pos['risk_score']        = round(risk['risk_score'], 3)
        pos['net_theta']         = round(risk.get('net_theta', 0.0), 6)
        pos['net_gamma']         = round(risk.get('net_gamma', 0.0), 8)
        pos['net_vega']          = round(risk.get('net_vega', 0.0), 6)
        pos['dte']               = dte

    # ── Greek computation (HFT-aware) ─────────────────────────────────────────

    def _get_greek_risk_data_unified(
        self, pos: dict, chain, spot: Optional[float]
    ) -> Optional[tuple]:
        """
        Compute Greek risk data from either broker Greeks (HFT) or IV (non-HFT).

        Returns ``(dte, risk_dict)`` or ``None``.
        """
        from src.greeks import (
            position_risk_score,
            position_risk_score_from_greeks as _prs_g,
        )
        try:
            dte = (
                datetime.date.fromisoformat(pos['expiry'])
                - datetime.date.today()
            ).days
            if dte <= 0:
                return None

            if chain.has_broker_greeks:
                greeks_legs = self._build_greeks_legs_from_snapshots(
                    pos, chain.leg_specs, chain.osi_map, chain.snapshots, spot, dte
                )
                if not greeks_legs:
                    if self._gamma_risk_enabled:
                        _log.warning(
                            "[HFT] pos %s (%s/%s): all legs missing broker Greeks and "
                            "IV fallback inputs — gamma-risk check skipped; "
                            "price-based stop-loss still active",
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
                "[risk_service] Greeks error for pos %s (%s %s): %s",
                pos.get('id', '?'), pos.get('symbol', '?'),
                pos.get('expiry', '?'), exc, exc_info=True,
            )
        return None

    def _build_greeks_legs(
        self, pos: dict, put_map: dict, call_map: dict
    ) -> list[dict]:
        """
        Build the legs list consumed by ``position_risk_score()`` (non-HFT path).
        Each entry: ``{'strike', 'iv', 'option_type', 'position'}``.
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
                    "[risk_service] IV missing for pos %s %s strike=%.2f %s "
                    "— leg excluded from greek risk calculation",
                    pos.get('symbol', '?'), pos.get('type', '?'), strike, opt_type,
                )
        return legs

    def _build_greeks_legs_from_snapshots(
        self,
        pos: dict,
        leg_specs: list,
        osi_map: dict,
        snapshots: dict,
        spot: Optional[float],
        dte: int,
    ) -> list[dict]:
        """
        Build ``legs_with_greeks`` from Alpaca snapshot rows (HFT path).

        Returns a list compatible with ``position_risk_score_from_greeks()``:
        ``[{'delta', 'gamma', 'theta', 'position'}, …]``.
        Uses broker Greeks when present, otherwise falls back to IV-based
        Black-Scholes estimates from the snapshot row.
        """
        from src.greeks import bs_greeks

        def _estimate_from_snapshot(row: dict, strike: float, opt_type: str) -> Optional[dict]:
            raw = (
                row.get('impliedVolatility')
                if hasattr(row, 'get')
                else getattr(row, 'impliedVolatility', None)
            )
            try:
                iv = float(raw) if raw is not None else 0.0
            except (TypeError, ValueError):
                iv = 0.0
            if iv <= 0 or spot is None or spot <= 0 or dte <= 0:
                return None
            return bs_greeks(float(spot), float(strike), iv, float(dte), opt_type)

        legs = []
        for strike, opt_type, position_side in leg_specs:
            osi = osi_map.get((strike, opt_type))
            if osi is None:
                continue
            row = snapshots.get(osi)
            if row is None:
                if self._gamma_risk_enabled:
                    _log.warning(
                        "[HFT] snapshot not returned by Alpaca for %s (pos %s) — "
                        "leg skipped from gamma check",
                        osi, pos.get('id', '?'),
                    )
                continue

            delta = row.get('delta')
            gamma = row.get('gamma')
            theta = row.get('theta')
            vega = row.get('vega')

            if delta is not None and gamma is not None and theta is not None:
                legs.append({
                    'delta':    float(delta),
                    'gamma':    float(gamma),
                    'theta':    float(theta),
                    'vega':     float(vega or 0.0),
                    'position': position_side,
                })
                continue

            estimated = _estimate_from_snapshot(row, strike, opt_type)
            if estimated is not None:
                _log.debug(
                    "[HFT] broker Greeks missing for %s (pos %s) — using IV-based estimate",
                    osi, pos.get('id', '?'),
                )
                legs.append({
                    'delta':    float(estimated['delta']),
                    'gamma':    float(estimated['gamma']),
                    'theta':    float(estimated['theta']),
                    'vega':     float(estimated['vega']),
                    'position': position_side,
                })
                continue

            if self._gamma_risk_enabled:
                _log.warning(
                    "[HFT] broker Greeks missing and IV fallback unavailable for %s "
                    "(pos %s) — leg skipped from gamma check",
                    osi, pos.get('id', '?'),
                )
        return legs

    # ── Risk classification ───────────────────────────────────────────────────

    def _classify_risk_level(self, pos: dict) -> str:
        from src.risk_rules.classify import classify_risk_level
        return classify_risk_level(pos, self.stop_loss_multiplier)
