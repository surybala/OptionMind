from .base import StrikeFilter, FilterContext


class ProbabilityFilter:
    """Reject strikes where delta (1-prob_otm) exceeds max_delta, or prob_otm < min_prob."""
    name = 'probability'
    def passes(self, ctx: FilterContext) -> bool:
        if (1.0 - ctx.prob_otm) > ctx.max_delta:
            return False
        if ctx.prob_otm < ctx.min_prob:
            return False
        return True
