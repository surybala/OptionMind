import logging
from typing import Optional
from .base import CloseSignal

_log = logging.getLogger('optionwheel')


class ProfitTakeRule:
    """Close position when a target fraction of max profit has been captured.

    Fires when: pnl_per_share >= profit_take_pct * entry_premium
    e.g. profit_take_pct=0.80 closes once 80% of the collected premium is locked in.
    Set profit_take_pct=0.0 or enabled=False to disable.
    """
    name = 'PROFIT_TAKE'

    def __init__(self, enabled: bool, profit_take_pct: float):
        self._enabled = enabled
        self._pct = profit_take_pct

    def evaluate(self, entry_premium: float, current_mark: float,
                 pnl_per_share: float, spot: Optional[float],
                 pos: dict, **kwargs) -> Optional[CloseSignal]:
        if not self._enabled or self._pct <= 0 or entry_premium <= 0:
            return None
        target = self._pct * entry_premium
        if pnl_per_share < target:
            return None
        captured_pct = pnl_per_share / entry_premium
        reason_str = (
            f"profit_captured={captured_pct:.1%} >= target={self._pct:.1%} "
            f"(pnl={pnl_per_share:.2f} >= {target:.2f})"
        )
        extras_str = f"entry={entry_premium:.2f}, mark={current_mark:.2f}, pnl={pnl_per_share:.2f}"
        return CloseSignal(
            reason_tag='PROFIT_TAKE',
            reason_str=reason_str,
            extras_str=extras_str,
            metrics={
                'pnl_per_share':  pnl_per_share,
                'captured_pct':   round(captured_pct, 4),
                'profit_take_pct': self._pct,
            },
        )
