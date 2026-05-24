from typing import Optional, List
from .base import CloseSignal, RiskRule


class RulePipeline:
    """Applies risk rules in order; returns the first triggered CloseSignal or None."""
    def __init__(self, rules):
        self._rules = rules

    def evaluate_all(self, entry_premium: float, current_mark: float,
                     pnl_per_share: float, spot, pos: dict,
                     **kwargs) -> Optional[CloseSignal]:
        for rule in self._rules:
            result = rule.evaluate(entry_premium, current_mark,
                                   pnl_per_share, spot, pos, **kwargs)
            if result is not None:
                return result
        return None
