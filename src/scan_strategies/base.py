from abc import ABC, abstractmethod
from typing import List, Dict, Any


class StrategyScanner(ABC):
    """Base class for all option strategy scanners."""

    def __init__(self, params: dict, min_prob: float,
                 min_otm_put: float, min_otm_call: float,
                 atr_enabled: bool, atr_multiplier: float,
                 prob_otm_fn, row_oi_vol_fn,
                 dynamic_width_cfg: dict = None):
        self._params           = params
        self._min_prob         = min_prob
        self._min_otm_put      = min_otm_put
        self._min_otm_call     = min_otm_call
        self._atr_enabled      = atr_enabled
        self._atr_multiplier   = atr_multiplier
        self._prob_otm         = prob_otm_fn
        self._row_oi_vol       = row_oi_vol_fn
        self._dynamic_width_cfg = dynamic_width_cfg or {}

    @abstractmethod
    def scan(self, symbol, current_price, expiry, days,
             *chain_args, atr=0.0, **kwargs) -> List[Dict[str, Any]]: ...

    @staticmethod
    def _score(premium: float, prob_win: float, width: float = 1.0) -> float:
        """Yield-normalised score: (credit / width) × prob².

        Dividing by width removes the bias toward near-the-money spreads that
        collect more absolute dollars at the same strike interval — a 10% yield
        spread at 10% OTM scores identically to a 10% yield spread at 5% OTM,
        so the probability term is the sole safety differentiator.
        """
        effective_width = max(width, 0.01)
        return round((premium / effective_width) * (prob_win ** 2), 4)

    def _width_for_price(self, price: float, fallback: float) -> float:
        """Return the strike width appropriate for *price* using dynamic tiers.

        If dynamic_width is disabled (or no tiers configured), returns *fallback*
        (the per-strategy strike_width from config).  Tiers are evaluated in
        order; the first tier whose max_price >= price wins.
        """
        cfg = self._dynamic_width_cfg
        if not cfg.get('enabled'):
            return fallback
        for tier in cfg.get('tiers', []):
            if price <= tier['max_price']:
                return float(tier['width'])
        tiers = cfg.get('tiers', [])
        return float(tiers[-1]['width']) if tiers else fallback
