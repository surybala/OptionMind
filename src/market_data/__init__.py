"""
market_data
===========

HFT / non-HFT data-source isolation layer.

The ``DataAdapter`` class consolidates every ``if _HFT_MODE:`` / ``if hft_mode:``
branch that was previously scattered through ``scanner.py`` and
``position_monitor.py``.  All callers receive the same interface; the adapter
reads the runtime mode flag *lazily* via a callable so ``@patch`` in tests
continues to work after the scanner object has already been constructed.

Public API
----------
    from src.market_data import DataAdapter, PositionChainResult

    adapter = DataAdapter(
        hist_cache      = _HIST_CACHE,
        chain_cache     = _CHAIN_CACHE,
        client_getter   = lambda: _ALPACA_CLIENT,   # read lazily
        hft_mode_getter = lambda: _HFT_MODE,        # read lazily
        hft_config      = _HFT_CONFIG,
    )
"""

from .base import PositionChainResult
from .adapter import DataAdapter

__all__ = ["DataAdapter", "PositionChainResult"]
