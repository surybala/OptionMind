"""Outcome labeling engines for OptionMind ML datasets."""

from ml.labels.short_option import (
    ShortOptionLabel,
    ShortOptionLabelConfig,
    label_short_option_path,
)
from ml.labels.strategy import (
    CreditSpreadLabel,
    CreditSpreadLabelConfig,
    label_credit_spread_path,
)

__all__ = [
    "CreditSpreadLabel",
    "CreditSpreadLabelConfig",
    "ShortOptionLabel",
    "ShortOptionLabelConfig",
    "label_credit_spread_path",
    "label_short_option_path",
]
