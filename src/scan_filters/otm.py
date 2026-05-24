from .base import StrikeFilter, FilterContext


class OtmDirectionFilter:
    """Reject strikes that are in-the-money (wrong side of spot)."""
    name = 'otm_direction'
    def passes(self, ctx: FilterContext) -> bool:
        if ctx.option_type == 'put':
            return ctx.short_strike < ctx.current_price
        return ctx.short_strike > ctx.current_price


class SanityBoundsFilter:
    """Reject puts below 30% of spot or calls above 200% of spot (data errors)."""
    name = 'sanity_bounds'
    def passes(self, ctx: FilterContext) -> bool:
        if ctx.option_type == 'put':
            return ctx.short_strike > ctx.current_price * 0.30
        return ctx.short_strike < ctx.current_price * 2.0


class MinOtmPctFilter:
    """Reject short strikes too close to spot (less than min_otm_pct away)."""
    name = 'min_otm_pct'
    def passes(self, ctx: FilterContext) -> bool:
        if ctx.option_type == 'put' and ctx.min_otm_put > 0:
            return ctx.short_strike <= ctx.current_price * (1.0 - ctx.min_otm_put)
        if ctx.option_type == 'call' and ctx.min_otm_call > 0:
            return ctx.short_strike >= ctx.current_price * (1.0 + ctx.min_otm_call)
        return True
