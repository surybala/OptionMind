import dataclasses
from typing import Protocol, runtime_checkable


@dataclasses.dataclass
class FilterContext:
    current_price: float
    short_strike: float
    option_type: str       # 'put' | 'call'
    prob_otm: float
    atr: float
    min_otm_put: float
    min_otm_call: float
    atr_enabled: bool
    atr_multiplier: float
    max_delta: float
    min_prob: float


@runtime_checkable
class StrikeFilter(Protocol):
    name: str
    def passes(self, ctx: 'FilterContext') -> bool: ...
