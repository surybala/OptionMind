from .base import FilterContext, StrikeFilter
from .otm import OtmDirectionFilter, SanityBoundsFilter, MinOtmPctFilter
from .atr import AtrDistanceFilter
from .probability import ProbabilityFilter
from .liquidity import apply_liquidity_filter
from .earnings import should_skip_expiry


class FilterChain:
    """Applies a sequence of StrikeFilters; returns False on first rejection."""
    def __init__(self, filters):
        self._filters = filters

    def passes(self, ctx: FilterContext) -> bool:
        return all(f.passes(ctx) for f in self._filters)


__all__ = [
    'FilterContext', 'StrikeFilter', 'FilterChain',
    'OtmDirectionFilter', 'SanityBoundsFilter', 'MinOtmPctFilter',
    'AtrDistanceFilter', 'ProbabilityFilter',
    'apply_liquidity_filter', 'should_skip_expiry',
]
