"""Shared feature-group definitions used by both optimizer scripts.

ALWAYS_ON groups are never toggled by Optuna — the model cannot rank trades
without the core spread economics they describe.

TOGGLEABLE groups are candidate inclusions for Pass 2 (feature selection).
Optuna decides include (True) or exclude (False) for each group independently.
"""
from __future__ import annotations

ALWAYS_ON: dict[str, list[str]] = {
    "contract_structure": [
        "is_pcs", "is_ccs", "dte", "strike", "strike_distance_pct", "moneyness",
    ],
    "spread_structure": [
        "spread_width", "entry_credit", "max_profit", "max_loss", "credit_to_width",
        "long_option_entry_price", "long_option_entry_volume",
        "long_option_entry_trade_count", "long_option_entry_vwap",
    ],
}

TOGGLEABLE: dict[str, list[str]] = {
    "underlying_price": [
        "underlying_close", "underlying_return_1d", "underlying_return_3d",
        "underlying_return_5d", "underlying_return_20d", "underlying_range_pct",
        "underlying_sma_20_distance_pct", "underlying_above_sma_20", "underlying_volume",
    ],
    "underlying_vol": [
        "underlying_realized_vol_5d", "underlying_realized_vol_10d",
        "underlying_realized_vol_20d", "underlying_skew_5d",
    ],
    "vol_momentum": [
        "underlying_volatility_ratio_5d_20d", "underlying_vol_vs_market", "vol_acceleration",
    ],
    "market_regime": [
        "market_return_5d", "market_return_20d", "market_realized_vol_5d",
        "market_realized_vol_20d", "market_sma_20_distance_pct",
        "market_above_sma_20", "market_volatility_ratio_5d_20d",
    ],
    "option_entry": [
        "option_entry_price", "option_entry_range_pct", "option_entry_volume",
        "option_entry_trade_count", "option_entry_vwap",
    ],
    "option_activity": [
        "option_volume_5d_avg", "option_trade_count_5d_avg", "option_activity_spike",
    ],
    "greeks": [
        "implied_volatility", "option_delta", "option_gamma", "option_theta", "option_vega",
    ],
    "iv_surface": [
        "iv_vs_hv5d", "iv_vs_hv20d", "iv_skew_wing",
    ],
    "vix_features": [
        "vix_regime", "vix_return_5d", "vix_realized_vol_5d",
    ],
    "event_risk": [
        "days_to_earnings", "days_to_ex_dividend", "days_to_fomc", "days_to_macro_event",
    ],
    "credit_efficiency": [
        "credit_per_day_per_risk",
    ],
}

CHAMPION_GROUPS: dict[str, bool] = {g: True for g in TOGGLEABLE}
