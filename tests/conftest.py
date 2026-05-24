"""
conftest.py — loaded by pytest before any test module is imported.

1. Import real numpy and pandas here so they are registered in sys.modules
   *before* any test module's ``sys.modules[...] = MagicMock()`` can replace
   them with MagicMock objects.

2. Mock yfinance (not a test dependency) so scanner imports do not attempt
   any live network calls.

Tests that rely on mock DataFrames (test_new_strategies.py, test_csp.py …)
build their own MagicMock objects and are unaffected by real pandas being
present in the module namespace.
"""
import sys
from unittest.mock import MagicMock

import numpy   # noqa: F401  — registers real numpy in sys.modules
import pandas  # noqa: F401  — registers real pandas in sys.modules

# Provide a stub for yfinance so the src imports below do not attempt
# any live HTTP calls.
sys.modules.setdefault('yfinance', MagicMock())
