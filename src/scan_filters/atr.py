import math
from .base import StrikeFilter, FilterContext


class AtrDistanceFilter:
    """Reject short strikes closer to spot than multiplier x ATR."""
    name = 'atr_distance'
    def passes(self, ctx: FilterContext) -> bool:
        if not ctx.atr_enabled or ctx.atr <= 0:
            return True
        return abs(ctx.current_price - ctx.short_strike) >= ctx.atr_multiplier * ctx.atr
