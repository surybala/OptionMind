"""ML scanner hook for OptionMind.

This is the only scanner entry point the app should use going forward.
The inherited deterministic scanner lives in ``src/scanner.py`` as legacy
reference code, but it should not be used as a fallback for live candidate
generation.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

_log = logging.getLogger("optionwheel")


class ModelScanner:
    """Bridge between the application and a future trained model."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.scanner_config = config.get("ml_scanner", {})
        self.model_provider = self._load_provider()

    def get_top_picks(self, ticker_list: list[str], n: int = 10) -> list[dict]:
        """Return model-ranked candidates, or no picks until a model exists."""
        if not self.scanner_config.get("enabled", False):
            _log.warning(
                "ML scanner is disabled or no model is configured. "
                "No deterministic scanner fallback will run."
            )
            return []

        if self.model_provider is None:
            _log.warning(
                "ML scanner is enabled but no provider was loaded. "
                "Set ml_scanner.provider to 'module:callable'."
            )
            return []

        candidates = self._call_provider(ticker_list, n)
        normalized = [self._normalize_candidate(candidate) for candidate in candidates]
        normalized = [candidate for candidate in normalized if candidate is not None]
        normalized.sort(key=lambda item: item.get("model_score", item.get("score", 0.0)), reverse=True)
        return normalized[:n]

    def _load_provider(self) -> Any | None:
        provider_path = self.scanner_config.get("provider")
        if not provider_path:
            return None
        if ":" not in provider_path:
            raise ValueError("ml_scanner.provider must use 'module:callable' format")

        module_name, attr_name = provider_path.split(":", 1)
        module = importlib.import_module(module_name)
        provider = getattr(module, attr_name)
        if isinstance(provider, type):
            return provider(self.config)
        return provider

    def _call_provider(self, ticker_list: list[str], n: int) -> list[dict]:
        provider = self.model_provider
        if hasattr(provider, "get_top_picks"):
            return list(provider.get_top_picks(ticker_list, n=n))
        if callable(provider):
            return list(provider(ticker_list=ticker_list, n=n, config=self.config))
        raise TypeError("ML scanner provider must be callable or expose get_top_picks()")

    def _normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        item = dict(candidate)
        item.setdefault("source", "ml_model")
        item.setdefault("score", item.get("model_score", 0.0))
        item.setdefault("prob_win", item.get("probability_of_profit", 0.0))
        item.setdefault("quantity", 1)
        item.setdefault(
            "mispricing_score_basis",
            "ML model score: expected utility / P&L-centered inference",
        )

        min_score = float(self.scanner_config.get("min_score", float("-inf")))
        try:
            score = float(item.get("model_score", item.get("score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < min_score:
            return None
        return item


OptionScanner = ModelScanner
