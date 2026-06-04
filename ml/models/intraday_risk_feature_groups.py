"""Feature-group definitions for intraday risk monitor feature selection."""
from __future__ import annotations

ALWAYS_ON: dict[str, list[str]] = {
    "core_position": [
        "is_pcs",
        "is_ccs",
        "dte",
        "spread_width",
        "entry_credit",
        "max_loss",
        "stop_debit",
        "profit_take_debit",
        "current_debit",
        "pnl_per_contract",
        "profit_captured_pct",
        "stop_distance_pct",
        "minutes_since_entry",
        "minutes_to_expiry",
        "underlying_close",
        "current_debit_to_stop",
        "current_debit_to_profit_take",
        "debit_to_width",
        "loss_pct_of_max_loss",
        "credit_retained_pct",
    ],
}

TOGGLEABLE: dict[str, list[str]] = {
    "underlying_path": [
        "underlying_return_5m",
        "underlying_return_15m",
        "underlying_return_30m",
        "abs_underlying_return_5m",
        "abs_underlying_return_15m",
        "abs_underlying_return_30m",
    ],
    "underlying_volatility": [
        "underlying_realized_vol_15m",
        "underlying_realized_vol_30m",
        "underlying_vol_ratio_15m_30m",
    ],
    "leg_marks": [
        "short_leg_close",
        "long_leg_close",
        "short_leg_share_of_debit",
        "long_leg_share_of_debit",
    ],
    "leg_activity": [
        "short_leg_volume",
        "long_leg_volume",
        "short_leg_trade_count",
        "long_leg_trade_count",
        "leg_volume_imbalance",
        "leg_trade_count_imbalance",
    ],
    "market_regime": [
        "market_trend_uptrend",
        "market_trend_sideways",
        "market_trend_downtrend",
        "market_volatility_low",
        "market_volatility_medium",
        "market_volatility_high",
    ],
}

CHAMPION_GROUPS: dict[str, bool] = {group: True for group in TOGGLEABLE}
