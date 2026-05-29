"""Labels for executable option strategies such as PCS and CCS."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal

from ml.labels.short_option import ExitReason
from ml.providers.models import PriceBar


StrategyType = Literal["PCS", "CCS"]


@dataclass(frozen=True)
class CreditSpreadLabelConfig:
    profit_take_pct: float = 0.75
    stop_loss_multiple: float = 2.0
    stop_loss_max_loss_pct: float | None = 0.80
    large_loss_multiple: float = 1.0
    contract_multiplier: int = 100
    price_field: Literal["close"] = "close"
    label_version: str = "credit_spread_labels_v002"

    def __post_init__(self) -> None:
        if not 0.0 <= self.profit_take_pct < 1.0:
            raise ValueError("profit_take_pct must be in [0.0, 1.0)")
        if self.stop_loss_multiple <= 1.0:
            raise ValueError("stop_loss_multiple must be greater than 1.0")
        if self.stop_loss_max_loss_pct is not None and self.stop_loss_max_loss_pct <= 0.0:
            raise ValueError("stop_loss_max_loss_pct must be positive")
        if self.large_loss_multiple <= 0.0:
            raise ValueError("large_loss_multiple must be positive")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")


@dataclass(frozen=True)
class CreditSpreadLabel:
    label_version: str
    strategy: StrategyType
    spread_width: float
    entry_credit: float
    exit_debit: float
    max_profit: float
    max_loss: float
    return_on_risk: float | None
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


def label_credit_spread_path(
    *,
    strategy: StrategyType,
    short_entry_bar: PriceBar,
    long_entry_bar: PriceBar,
    short_path: list[PriceBar],
    long_path: list[PriceBar],
    spread_width: float | None = None,
    config: CreditSpreadLabelConfig | None = None,
) -> CreditSpreadLabel:
    """Label an executable PCS/CCS using forward prices for both legs.

    The strategy is entered for a net credit: sell the short option and buy the
    protective long option. Future P&L is measured using the net debit needed
    to close both legs at each timestamp.
    """
    if strategy not in {"PCS", "CCS"}:
        raise ValueError("strategy must be PCS or CCS")

    label_config = config or CreditSpreadLabelConfig()
    if short_entry_bar.timestamp != long_entry_bar.timestamp:
        raise ValueError("short and long entry bars must share a timestamp")

    entry_credit = _spread_debit(short_entry_bar, long_entry_bar, label_config.price_field)
    if entry_credit <= 0.0:
        raise ValueError("entry credit must be positive")
    resolved_width = _resolve_spread_width(spread_width, short_entry_bar.symbol, long_entry_bar.symbol)
    if resolved_width <= 0.0:
        raise ValueError("spread_width must be positive")
    if entry_credit >= resolved_width:
        raise ValueError("entry credit must be less than spread_width")

    entry_timestamp = short_entry_bar.timestamp
    forward_pairs = _aligned_forward_pairs(
        short_path,
        long_path,
        entry_timestamp,
        label_config.price_field,
    )

    profit_take_debit = entry_credit * (1.0 - label_config.profit_take_pct)
    max_loss_per_share = resolved_width - entry_credit
    if label_config.stop_loss_max_loss_pct is not None:
        stop_debit = entry_credit + max_loss_per_share * label_config.stop_loss_max_loss_pct
    else:
        stop_debit = entry_credit * label_config.stop_loss_multiple
    exit_timestamp = entry_timestamp
    exit_debit = entry_credit
    exit_reason: ExitReason = "horizon"
    max_debit = entry_credit
    min_debit = entry_credit

    for timestamp, debit in forward_pairs:
        if debit < 0.0:
            raise ValueError("forward spread debit must be non-negative")
        # Cap debit at spread width: a credit spread's exit cost can never exceed its width.
        # Zero-priced long-leg bars (stale quotes on illiquid options) can produce debit > width;
        # capping here prevents wildly inflated losses that exceed the theoretical maximum.
        debit = min(debit, resolved_width)
        max_debit = max(max_debit, debit)
        min_debit = min(min_debit, debit)
        exit_timestamp = timestamp
        exit_debit = debit
        if debit > stop_debit:
            exit_reason = "stop_loss"
            break
        if debit <= profit_take_debit:
            exit_reason = "profit_take"
            break

    multiplier = label_config.contract_multiplier
    realized = round((entry_credit - exit_debit) * multiplier, 4)
    max_profit = round(entry_credit * multiplier, 4)
    max_loss = round(max_loss_per_share * multiplier, 4)
    max_adverse = round(max(0.0, (max_debit - entry_credit) * multiplier), 4)
    max_favorable = round(max(0.0, (entry_credit - min_debit) * multiplier), 4)
    days_to_exit = round((exit_timestamp - entry_timestamp).total_seconds() / 86_400, 6)

    return CreditSpreadLabel(
        label_version=label_config.label_version,
        strategy=strategy,
        spread_width=resolved_width,
        entry_credit=entry_credit,
        exit_debit=exit_debit,
        max_profit=max_profit,
        max_loss=max_loss,
        return_on_risk=round(realized / max_loss, 8) if max_loss > 0 else None,
        exit_timestamp=exit_timestamp,
        exit_reason=exit_reason,
        expected_pnl=realized,
        realized_pnl_per_contract=realized,
        profit_label=1 if realized > 0 else 0,
        stop_loss_hit=1 if exit_reason == "stop_loss" else 0,
        large_loss_label=1 if realized <= -(max_loss * label_config.large_loss_multiple) else 0,
        max_adverse_excursion=max_adverse,
        max_favorable_excursion=max_favorable,
        days_to_exit=days_to_exit,
    )


def _aligned_forward_pairs(
    short_path: list[PriceBar],
    long_path: list[PriceBar],
    entry_timestamp: datetime,
    price_field: Literal["close"],
) -> list[tuple[datetime, float]]:
    long_by_time = {
        bar.timestamp: bar
        for bar in long_path
        if bar.timestamp > entry_timestamp
    }
    pairs: list[tuple[datetime, float]] = []
    for short_bar in sorted(short_path, key=lambda bar: bar.timestamp):
        if short_bar.timestamp <= entry_timestamp:
            continue
        long_bar = long_by_time.get(short_bar.timestamp)
        if long_bar is None:
            continue
        pairs.append((short_bar.timestamp, _spread_debit(short_bar, long_bar, price_field)))
    return pairs


def _spread_debit(short_bar: PriceBar, long_bar: PriceBar, price_field: Literal["close"]) -> float:
    short_price = float(getattr(short_bar, price_field))
    long_price = float(getattr(long_bar, price_field))
    if short_price < 0.0 or long_price < 0.0:
        raise ValueError("option prices must be non-negative")
    return round(short_price - long_price, 8)


def _resolve_spread_width(
    explicit_width: float | None,
    short_symbol: str,
    long_symbol: str,
) -> float:
    if explicit_width is not None:
        return round(float(explicit_width), 8)
    short_strike = _strike_from_osi(short_symbol)
    long_strike = _strike_from_osi(long_symbol)
    if short_strike is None or long_strike is None:
        return 0.0
    return round(abs(short_strike - long_strike), 8)


def _strike_from_osi(symbol: str) -> float | None:
    normalized = symbol[2:] if symbol.startswith("O:") else symbol
    if len(normalized) < 15:
        return None
    raw = normalized[-8:]
    if not raw.isdigit():
        return None
    return int(raw) / 1000.0
