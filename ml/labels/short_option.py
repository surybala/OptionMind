"""Standardized labels for historical short-option candidates.

The labeler depends only on normalized provider models. It deliberately keeps
the first label definition simple and inspectable so future model metrics can
state exactly what outcome rule they learned against.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from ml.providers.models import PriceBar


ExitReason = Literal["profit_take", "stop_loss", "horizon"]


@dataclass(frozen=True)
class ShortOptionLabelConfig:
    profit_take_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    large_loss_multiple: float = 1.0
    contract_multiplier: int = 100
    price_field: Literal["close"] = "close"
    label_version: str = "short_option_labels_v001"

    def __post_init__(self) -> None:
        if not 0.0 <= self.profit_take_pct < 1.0:
            raise ValueError("profit_take_pct must be in [0.0, 1.0)")
        if self.stop_loss_multiple <= 1.0:
            raise ValueError("stop_loss_multiple must be greater than 1.0")
        if self.large_loss_multiple <= 0.0:
            raise ValueError("large_loss_multiple must be positive")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")


@dataclass(frozen=True)
class ShortOptionLabel:
    label_version: str
    entry_price: float
    exit_price: float
    exit_timestamp: datetime
    exit_reason: ExitReason
    expected_pnl: float
    realized_pnl_per_contract: float
    profit_label: int
    stop_loss_hit: int
    large_loss_label: int
    max_adverse_excursion: float
    max_favorable_excursion: float
    days_to_exit: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def label_short_option_path(
    entry_bar: PriceBar,
    path: list[PriceBar],
    config: ShortOptionLabelConfig | None = None,
) -> ShortOptionLabel:
    """Label a short option candidate using a forward option-price path.

    The candidate is treated as a short option entered at ``entry_bar``. The
    first bar whose selected price reaches the stop-loss or profit-take rule
    determines the simulated exit; otherwise the final forward bar is used.
    """

    label_config = config or ShortOptionLabelConfig()
    entry_price = _bar_price(entry_bar, label_config.price_field)
    if entry_price < 0.0:
        raise ValueError("entry option price must be non-negative")

    forward_path = sorted(
        [bar for bar in path if bar.timestamp > entry_bar.timestamp],
        key=lambda bar: bar.timestamp,
    )

    profit_take_price = entry_price * (1.0 - label_config.profit_take_pct)
    stop_price = entry_price * label_config.stop_loss_multiple

    exit_bar = forward_path[-1] if forward_path else entry_bar
    exit_reason: ExitReason = "horizon"
    max_cost = entry_price
    min_cost = entry_price

    for bar in forward_path:
        cost = _bar_price(bar, label_config.price_field)
        if cost < 0.0:
            raise ValueError("option path prices must be non-negative")
        max_cost = max(max_cost, cost)
        min_cost = min(min_cost, cost)
        if cost >= stop_price:
            exit_bar = bar
            exit_reason = "stop_loss"
            break
        if cost <= profit_take_price:
            exit_bar = bar
            exit_reason = "profit_take"
            break

    exit_price = _bar_price(exit_bar, label_config.price_field)
    multiplier = label_config.contract_multiplier
    realized = round((entry_price - exit_price) * multiplier, 4)
    max_adverse = round(max(0.0, (max_cost - entry_price) * multiplier), 4)
    max_favorable = round(max(0.0, (entry_price - min_cost) * multiplier), 4)
    days_to_exit = round((exit_bar.timestamp - entry_bar.timestamp).total_seconds() / 86_400, 6)

    return ShortOptionLabel(
        label_version=label_config.label_version,
        entry_price=entry_price,
        exit_price=exit_price,
        exit_timestamp=exit_bar.timestamp,
        exit_reason=exit_reason,
        expected_pnl=realized,
        realized_pnl_per_contract=realized,
        profit_label=1 if realized > 0 else 0,
        stop_loss_hit=1 if exit_reason == "stop_loss" else 0,
        large_loss_label=1 if realized <= -(entry_price * multiplier * label_config.large_loss_multiple) else 0,
        max_adverse_excursion=max_adverse,
        max_favorable_excursion=max_favorable,
        days_to_exit=days_to_exit,
    )


def _bar_price(bar: PriceBar, price_field: Literal["close"]) -> float:
    return float(getattr(bar, price_field))
