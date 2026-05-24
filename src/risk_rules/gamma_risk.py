import logging
from typing import Optional, Tuple
from .base import CloseSignal

_log = logging.getLogger('optionwheel')


def eval_gamma_trigger(risk: dict, dte: int, entry_premium: float,
                       pnl_per_share: float, ratio_threshold: float,
                       min_delta: float, min_profit_pct: float,
                       urgent_delta: float) -> Optional[Tuple]:
    """
    Returns (reason_str, extras_str, metrics_dict) if gamma trigger fires, else None.
    Extracted from PositionMonitor._eval_gamma_trigger().
    """
    ratio       = risk['gamma_theta_ratio']
    short_delta = abs(risk['net_short_delta'])
    score       = risk['risk_score']

    extras = (
        f"\u03b3/\u03b8={ratio:.2f}  \u0394short={short_delta:.3f}  "
        f"risk_score={score:.2f}"
    )

    profit_captured_pct = pnl_per_share / entry_premium if entry_premium > 0 else 0.0
    urgent     = short_delta >= urgent_delta
    enough_pnl = profit_captured_pct >= min_profit_pct

    # Near-expiry guard: when DTE <= 1, theta → 0 and the ratio becomes
    # artificially inflated (sentinel = gamma × 1000). Skip the ratio-based
    # check entirely; only exit if delta is urgently ITM (urgent override).
    if dte <= 1 and not urgent:
        return None

    if (ratio >= ratio_threshold
            and short_delta >= min_delta
            and (enough_pnl or urgent)):
        reason = (
            f"ratio={ratio:.2f}>={ratio_threshold:.2f}, "
            f"delta={short_delta:.3f}"
        )
        if urgent and not enough_pnl:
            reason += " [urgent-delta override]"
        metrics = {
            'ratio':       ratio,
            'short_delta': short_delta,
            'risk_score':  score,
            'dte':         dte,
        }
        return reason, extras, metrics
    return None


class GammaRiskRule:
    """Close position when gamma risk exceeds safe threshold relative to theta income."""
    name = 'GAMMA_RISK'

    def __init__(self, enabled: bool, ratio_threshold: float,
                 min_delta: float, min_profit_pct: float,
                 urgent_delta: float):
        self._enabled = enabled
        self._ratio_threshold = ratio_threshold
        self._min_delta = min_delta
        self._min_profit_pct = min_profit_pct
        self._urgent_delta = urgent_delta

    def evaluate(self, entry_premium: float, current_mark: float,
                 pnl_per_share: float, spot: Optional[float],
                 pos: dict, **kwargs) -> Optional[CloseSignal]:
        if not self._enabled:
            return None
        # The actual gamma check requires put_map/call_map/greeks which are passed as kwargs
        result = kwargs.get('gamma_check_result')
        if result is None:
            return None
        reason_str, extras_str, metrics = result
        return CloseSignal(
            reason_tag='GAMMA_RISK',
            reason_str=reason_str,
            extras_str=extras_str,
            metrics=metrics,
        )
