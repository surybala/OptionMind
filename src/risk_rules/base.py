import dataclasses
from typing import Optional, Protocol, runtime_checkable


@dataclasses.dataclass
class CloseSignal:
    """Returned by a RiskRule when it triggers a position close."""
    reason_tag: str    # 'STOP_LOSS' | 'GAMMA_RISK' | 'PROFIT_TAKE'
    reason_str: str    # human-readable details for log/email
    extras_str: str    # display string
    metrics: dict      # enrichment fields: ratio, short_delta, risk_score, dte


@runtime_checkable
class RiskRule(Protocol):
    name: str
    def evaluate(self, entry_premium: float, current_mark: float,
                 pnl_per_share: float, spot: Optional[float],
                 pos: dict, **kwargs) -> Optional[CloseSignal]: ...
