"""Prototype historical option-candidate dataset builder.

This module is intentionally provider-agnostic. It consumes normalized
provider protocols and emits training-row shaped records that can later be
written to parquet/csv and used by feature, label, and model code.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from ml.labels import ShortOptionLabelConfig, label_short_option_path
from ml.providers.models import DividendEvent, EarningsEvent, EconomicEvent, OptionContract, PriceBar
from ml.providers.protocols import DividendDataProvider, EconomicCalendarProvider, EventDataProvider, MarketDataProvider, OptionContractProvider, OptionPriceProvider


@dataclass(frozen=True)
class CandidateDatasetConfig:
    underlyings: list[str]
    entry_start: datetime
    entry_end: datetime
    min_dte: int = 7
    max_dte: int = 45
    contract_status: str = "inactive"
    option_limit: int | None = 100
    max_contracts_per_underlying: int = 25
    max_rows_per_underlying: int | None = None
    max_abs_strike_distance_pct: float | None = 0.30
    min_forward_bars: int = 2
    sample_every_n_bars: int = 1
    stock_lookback_days: int = 60
    market_regime_symbol: str = "SPY"
    forward_days: int = 30
    profit_take_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    option_timeframe: str = "1Day"
    stock_timeframe: str = "1Day"
    large_loss_multiple: float = 1.0
    label_version: str = "short_option_labels_v001"
    build_window_days: int = 45
    vix_symbol: str = "VIX"
    risk_free_rate: float = 0.045
    option_lookback_days: int = 10


@dataclass(frozen=True)
class CandidateDatasetRow:
    entry_timestamp: datetime
    underlying: str
    option_symbol: str
    option_type: str | None
    strike: float | None
    expiration: date | None
    dte: int | None
    source: str

    underlying_close: float | None
    underlying_return_1d: float | None
    underlying_return_5d: float | None
    underlying_return_20d: float | None
    underlying_range_pct: float | None
    underlying_realized_vol_5d: float | None
    underlying_realized_vol_20d: float | None
    underlying_sma_20_distance_pct: float | None
    underlying_above_sma_20: int | None
    underlying_volatility_ratio_5d_20d: float | None
    underlying_volume: float | None
    strike_distance_pct: float | None
    moneyness: float | None

    market_regime_symbol: str | None
    market_return_5d: float | None
    market_return_20d: float | None
    market_realized_vol_5d: float | None
    market_realized_vol_20d: float | None
    market_sma_20_distance_pct: float | None
    market_above_sma_20: int | None
    market_volatility_ratio_5d_20d: float | None
    market_trend_regime: str | None
    market_volatility_regime: str | None

    option_entry_open: float
    option_entry_high: float
    option_entry_low: float
    option_entry_price: float
    option_entry_range_pct: float | None
    option_entry_volume: float | None
    option_entry_trade_count: int | None
    option_entry_vwap: float | None
    option_exit_price: float
    exit_timestamp: datetime
    exit_reason: str

    expected_pnl: float
    realized_pnl_per_contract: float
    profit_label: int
    stop_loss_hit: int
    large_loss_label: int
    max_adverse_excursion: float
    max_favorable_excursion: float
    days_to_exit: float
    label_version: str
    missing_fields: tuple[str, ...] = field(default_factory=tuple)

    # Additional underlying features
    underlying_realized_vol_10d: float | None = None
    underlying_return_3d: float | None = None
    underlying_skew_5d: float | None = None

    # Black-Scholes Greeks and implied volatility
    implied_volatility: float | None = None
    option_delta: float | None = None
    option_gamma: float | None = None
    option_theta: float | None = None
    option_vega: float | None = None
    iv_vs_hv5d: float | None = None
    iv_vs_hv20d: float | None = None

    # Option historical context (pre-entry lookback)
    option_volume_5d_avg: float | None = None
    option_trade_count_5d_avg: float | None = None

    # VIX market regime
    vix_close: float | None = None
    vix_return_5d: float | None = None
    vix_realized_vol_5d: float | None = None
    vix_above_20: int | None = None
    vix_above_30: int | None = None

    # Event risk — earnings (from EventDataProvider)
    days_to_earnings: int | None = None
    has_earnings_in_forward_days: int | None = None

    # Ex-dividend risk (from DividendDataProvider)
    days_to_ex_dividend: int | None = None
    has_dividend_in_forward_days: int | None = None

    # Macro event risk — FOMC / CPI / NFP (from EconomicCalendarProvider + FOMC calendar)
    days_to_fomc: int | None = None
    has_fomc_in_forward_days: int | None = None
    days_to_macro_event: int | None = None
    has_macro_event_in_forward_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoricalCandidateDatasetBuilder:
    """Build prototype candidate rows from provider protocols."""

    def __init__(
        self,
        market_provider: MarketDataProvider,
        contract_provider: OptionContractProvider,
        price_provider: OptionPriceProvider,
        event_provider: EventDataProvider | None = None,
        dividend_provider: DividendDataProvider | None = None,
        economic_provider: EconomicCalendarProvider | None = None,
    ) -> None:
        self.market_provider = market_provider
        self.contract_provider = contract_provider
        self.price_provider = price_provider
        self.event_provider = event_provider
        self.dividend_provider = dividend_provider
        self.economic_provider = economic_provider

    def build(self, config: CandidateDatasetConfig) -> list[CandidateDatasetRow]:
        rows: list[CandidateDatasetRow] = []
        underlying_row_counts: dict[str, int] = {u: 0 for u in config.underlyings}

        stock_start = config.entry_start - timedelta(days=config.stock_lookback_days)
        stock_symbols = _stock_symbols(config)
        stock_bars = self.market_provider.get_stock_bars(
            stock_symbols,
            stock_start,
            config.entry_end,
            config.stock_timeframe,
        )
        market_history = stock_bars.get(config.market_regime_symbol.upper(), [])
        vix_bars = stock_bars.get(config.vix_symbol.upper(), [])

        event_horizon = (config.entry_end + timedelta(days=config.forward_days)).date()

        # Fetch earnings calendar for the full entry window (stub returns {} if not wired).
        earnings_events: dict[str, list[EarningsEvent]] = {}
        if self.event_provider is not None:
            earnings_events = self.event_provider.get_earnings_calendar(
                config.underlyings,
                config.entry_start.date(),
                event_horizon,
            )

        # Fetch ex-dividend events (empty dict if provider not wired).
        dividend_events: dict[str, list[DividendEvent]] = {}
        if self.dividend_provider is not None:
            dividend_events = self.dividend_provider.get_dividends(
                config.underlyings,
                config.entry_start.date(),
                event_horizon,
            )

        # Fetch economic calendar events (FOMC always included via fomc_events helper).
        # Search 90 days ahead for FOMC/macro so days_to_* always resolves even
        # when the nearest event falls just outside the 30-day forward window.
        macro_search_end = (config.entry_end + timedelta(days=90)).date()
        from ml.providers.calendar import fomc_events as _fomc_events
        fomc_list: list[EconomicEvent] = _fomc_events(config.entry_start.date(), macro_search_end)
        macro_events: list[EconomicEvent] = list(fomc_list)
        if self.economic_provider is not None:
            macro_events.extend(
                self.economic_provider.get_economic_calendar(
                    config.entry_start.date(),
                    macro_search_end,
                )
            )
        # Deduplicate by (name, date) and sort chronologically
        macro_events = sorted(
            {(e.event_name, e.event_date): e for e in macro_events}.values(),
            key=lambda e: e.event_date,
        )

        for window_start, window_end, is_last_window in _date_windows(
            config.entry_start, config.entry_end, config.build_window_days
        ):
            # Stop early if all underlyings have hit their row cap.
            if config.max_rows_per_underlying is not None and all(
                underlying_row_counts[u] >= config.max_rows_per_underlying for u in config.underlyings
            ):
                break

            expiration_gte = _datetime_to_date(window_start) + timedelta(days=config.min_dte)
            expiration_lte = _datetime_to_date(window_end) + timedelta(days=config.max_dte)

            for underlying in config.underlyings:
                if (
                    config.max_rows_per_underlying is not None
                    and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                ):
                    continue

                contracts = self.contract_provider.get_option_contracts(
                    [underlying],
                    expiration_gte=expiration_gte,
                    expiration_lte=expiration_lte,
                    status=config.contract_status,
                    limit=config.option_limit,
                )
                underlying_history = stock_bars.get(underlying, [])
                reference_features = _underlying_features(underlying_history, window_start)
                contracts = _candidate_contracts(
                    contracts,
                    underlying,
                    config,
                    reference_close=reference_features["underlying_close"],
                )
                if not contracts:
                    continue

                # Extend lookback start so _option_lookback_features has pre-entry bars.
                option_fetch_start = window_start - timedelta(days=config.option_lookback_days)
                option_bars = self.price_provider.get_option_bars(
                    [contract.symbol for contract in contracts],
                    option_fetch_start,
                    window_end + timedelta(days=config.forward_days),
                    config.option_timeframe,
                    limit=None,
                )

                for contract in contracts:
                    path = sorted(option_bars.get(contract.symbol, []), key=lambda bar: bar.timestamp)
                    # Use exclusive upper bound except on the last window to avoid
                    # emitting duplicate rows when windows share a boundary timestamp.
                    if is_last_window:
                        window_bars = [b for b in path if window_start <= b.timestamp <= config.entry_end]
                    else:
                        window_bars = [b for b in path if window_start <= b.timestamp < window_end]
                    for entry_bar in window_bars[:: config.sample_every_n_bars]:
                        entry_dte = _dte(entry_bar.timestamp, contract.expiration)
                        if entry_dte is None or entry_dte < config.min_dte or entry_dte > config.max_dte:
                            continue
                        future_path = [
                            bar
                            for bar in path
                            if entry_bar.timestamp <= bar.timestamp <= entry_bar.timestamp + timedelta(days=config.forward_days)
                        ]
                        if len(future_path) < config.min_forward_bars:
                            continue
                        underlying_features = _underlying_features(underlying_history, entry_bar.timestamp)
                        market_features = _market_regime_features(
                            market_history,
                            entry_bar.timestamp,
                            config.market_regime_symbol,
                        )
                        greeks_features = _option_greeks_features(
                            entry_bar,
                            underlying_features["underlying_close"],
                            contract.strike,
                            contract.option_type,
                            entry_dte,
                            config.risk_free_rate,
                            underlying_features["underlying_realized_vol_5d"],
                            underlying_features["underlying_realized_vol_20d"],
                        )
                        lookback_features = _option_lookback_features(path, entry_bar.timestamp)
                        vix_feat = _vix_features(vix_bars, entry_bar.timestamp)
                        event_feat = _event_features(
                            earnings_events.get(underlying.upper(), []),
                            entry_bar.timestamp,
                            config.forward_days,
                        )
                        dividend_feat = _dividend_features(
                            dividend_events.get(underlying.upper(), []),
                            entry_bar.timestamp,
                            config.forward_days,
                        )
                        macro_feat = _macro_event_features(
                            fomc_list,
                            macro_events,
                            entry_bar.timestamp,
                            config.forward_days,
                        )
                        label = label_short_option_path(
                            entry_bar,
                            future_path,
                            ShortOptionLabelConfig(
                                profit_take_pct=config.profit_take_pct,
                                stop_loss_multiple=config.stop_loss_multiple,
                                large_loss_multiple=config.large_loss_multiple,
                                label_version=config.label_version,
                            ),
                        )
                        rows.append(
                            _row_from_label(
                                contract,
                                underlying,
                                entry_bar,
                                entry_dte,
                                underlying_features,
                                market_features,
                                greeks_features,
                                lookback_features,
                                vix_feat,
                                event_feat,
                                dividend_feat,
                                macro_feat,
                                label,
                            )
                        )
                        underlying_row_counts[underlying] += 1
                        if (
                            config.max_rows_per_underlying is not None
                            and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                        ):
                            break
                    if (
                        config.max_rows_per_underlying is not None
                        and underlying_row_counts[underlying] >= config.max_rows_per_underlying
                    ):
                        break
        return rows


def _candidate_contracts(
    contracts: list[OptionContract],
    underlying: str,
    config: CandidateDatasetConfig,
    reference_close: float | None = None,
) -> list[OptionContract]:
    filtered: list[OptionContract] = [
        contract
        for contract in contracts
        if contract.underlying.upper() == underlying.upper()
        and contract.expiration is not None
        and contract.strike is not None
        and contract.option_type in {"call", "put"}
    ]
    if reference_close and config.max_abs_strike_distance_pct is not None:
        filtered = [
            contract
            for contract in filtered
            if abs((float(contract.strike) - reference_close) / reference_close) <= config.max_abs_strike_distance_pct
        ]
    if reference_close:
        filtered = sorted(filtered, key=lambda c: abs(float(c.strike or 0.0) - reference_close))

    buckets: dict[tuple[date, str], list[OptionContract]] = {}
    for contract in filtered:
        buckets.setdefault((contract.expiration, contract.option_type), []).append(contract)

    ordered: list[OptionContract] = []
    bucket_values = list(buckets.values())
    while bucket_values and len(ordered) < config.max_contracts_per_underlying:
        next_values: list[list[OptionContract]] = []
        for bucket in bucket_values:
            if bucket and len(ordered) < config.max_contracts_per_underlying:
                ordered.append(bucket.pop(0))
            if bucket:
                next_values.append(bucket)
        bucket_values = next_values
    return ordered


def _stock_symbols(config: CandidateDatasetConfig) -> list[str]:
    symbols: list[str] = []
    for symbol in [*config.underlyings, config.market_regime_symbol, config.vix_symbol]:
        normalized = symbol.upper()
        if normalized not in symbols:
            symbols.append(normalized)
    return symbols


def _date_windows(
    start: datetime,
    end: datetime,
    window_days: int,
) -> list[tuple[datetime, datetime, bool]]:
    """Non-overlapping entry windows covering [start, end].

    Returns (window_start, window_end, is_last) triples. Windows are contiguous
    with no overlap. The last window's is_last flag signals callers to use an
    inclusive upper-bound filter so the final entry_end timestamp is included.
    """
    windows: list[tuple[datetime, datetime, bool]] = []
    current = start
    while current < end:
        window_end = min(current + timedelta(days=window_days), end)
        is_last = window_end >= end
        windows.append((current, window_end, is_last))
        if is_last:
            break
        current = window_end
    return windows


def _row_from_label(
    contract: OptionContract,
    underlying: str,
    entry_bar: PriceBar,
    entry_dte: int,
    underlying_features: dict[str, Any],
    market_features: dict[str, Any],
    greeks_features: dict[str, Any],
    lookback_features: dict[str, Any],
    vix_features: dict[str, Any],
    event_features: dict[str, Any],
    dividend_features: dict[str, Any],
    macro_features: dict[str, Any],
    label: Any,
) -> CandidateDatasetRow:
    all_features = {
        **underlying_features, **market_features, **greeks_features,
        **lookback_features, **vix_features, **event_features,
        **dividend_features, **macro_features,
    }
    missing = _missing_fields(contract, all_features)
    return CandidateDatasetRow(
        entry_timestamp=entry_bar.timestamp,
        underlying=underlying,
        option_symbol=contract.symbol,
        option_type=contract.option_type,
        strike=contract.strike,
        expiration=contract.expiration,
        dte=entry_dte,
        source=contract.source,
        underlying_close=underlying_features["underlying_close"],
        underlying_return_1d=underlying_features["underlying_return_1d"],
        underlying_return_5d=underlying_features["underlying_return_5d"],
        underlying_return_20d=underlying_features["underlying_return_20d"],
        underlying_range_pct=underlying_features["underlying_range_pct"],
        underlying_realized_vol_5d=underlying_features["underlying_realized_vol_5d"],
        underlying_realized_vol_20d=underlying_features["underlying_realized_vol_20d"],
        underlying_sma_20_distance_pct=underlying_features["underlying_sma_20_distance_pct"],
        underlying_above_sma_20=underlying_features["underlying_above_sma_20"],
        underlying_volatility_ratio_5d_20d=underlying_features["underlying_volatility_ratio_5d_20d"],
        underlying_volume=underlying_features["underlying_volume"],
        strike_distance_pct=_strike_distance_pct(contract.strike, underlying_features["underlying_close"]),
        moneyness=_moneyness(contract.strike, underlying_features["underlying_close"]),
        market_regime_symbol=market_features["market_regime_symbol"],
        market_return_5d=market_features["market_return_5d"],
        market_return_20d=market_features["market_return_20d"],
        market_realized_vol_5d=market_features["market_realized_vol_5d"],
        market_realized_vol_20d=market_features["market_realized_vol_20d"],
        market_sma_20_distance_pct=market_features["market_sma_20_distance_pct"],
        market_above_sma_20=market_features["market_above_sma_20"],
        market_volatility_ratio_5d_20d=market_features["market_volatility_ratio_5d_20d"],
        market_trend_regime=market_features["market_trend_regime"],
        market_volatility_regime=market_features["market_volatility_regime"],
        option_entry_open=float(entry_bar.open),
        option_entry_high=float(entry_bar.high),
        option_entry_low=float(entry_bar.low),
        option_entry_price=label.entry_price,
        option_entry_range_pct=_range_pct(entry_bar.high, entry_bar.low, entry_bar.close),
        option_entry_volume=entry_bar.volume,
        option_entry_trade_count=entry_bar.trade_count,
        option_entry_vwap=entry_bar.vwap,
        option_exit_price=label.exit_price,
        exit_timestamp=label.exit_timestamp,
        exit_reason=label.exit_reason,
        expected_pnl=label.expected_pnl,
        realized_pnl_per_contract=label.realized_pnl_per_contract,
        profit_label=label.profit_label,
        stop_loss_hit=label.stop_loss_hit,
        large_loss_label=label.large_loss_label,
        max_adverse_excursion=label.max_adverse_excursion,
        max_favorable_excursion=label.max_favorable_excursion,
        days_to_exit=label.days_to_exit,
        label_version=label.label_version,
        missing_fields=tuple(missing),
        # Additional underlying features
        underlying_realized_vol_10d=underlying_features.get("underlying_realized_vol_10d"),
        underlying_return_3d=underlying_features.get("underlying_return_3d"),
        underlying_skew_5d=underlying_features.get("underlying_skew_5d"),
        # Black-Scholes Greeks and IV
        implied_volatility=greeks_features.get("implied_volatility"),
        option_delta=greeks_features.get("option_delta"),
        option_gamma=greeks_features.get("option_gamma"),
        option_theta=greeks_features.get("option_theta"),
        option_vega=greeks_features.get("option_vega"),
        iv_vs_hv5d=greeks_features.get("iv_vs_hv5d"),
        iv_vs_hv20d=greeks_features.get("iv_vs_hv20d"),
        # Option lookback
        option_volume_5d_avg=lookback_features.get("option_volume_5d_avg"),
        option_trade_count_5d_avg=lookback_features.get("option_trade_count_5d_avg"),
        # VIX regime
        vix_close=vix_features.get("vix_close"),
        vix_return_5d=vix_features.get("vix_return_5d"),
        vix_realized_vol_5d=vix_features.get("vix_realized_vol_5d"),
        vix_above_20=vix_features.get("vix_above_20"),
        vix_above_30=vix_features.get("vix_above_30"),
        # Event risk — earnings
        days_to_earnings=event_features.get("days_to_earnings"),
        has_earnings_in_forward_days=event_features.get("has_earnings_in_forward_days"),
        # Ex-dividend risk
        days_to_ex_dividend=dividend_features.get("days_to_ex_dividend"),
        has_dividend_in_forward_days=dividend_features.get("has_dividend_in_forward_days"),
        # Macro event risk
        days_to_fomc=macro_features.get("days_to_fomc"),
        has_fomc_in_forward_days=macro_features.get("has_fomc_in_forward_days"),
        days_to_macro_event=macro_features.get("days_to_macro_event"),
        has_macro_event_in_forward_days=macro_features.get("has_macro_event_in_forward_days"),
    )


def _underlying_features(bars: list[PriceBar], entry_timestamp: datetime) -> dict[str, Any]:
    history = [bar for bar in sorted(bars, key=lambda b: b.timestamp) if bar.timestamp <= entry_timestamp]
    if not history:
        return {
            "underlying_close": None,
            "underlying_return_1d": None,
            "underlying_return_3d": None,
            "underlying_return_5d": None,
            "underlying_return_20d": None,
            "underlying_range_pct": None,
            "underlying_realized_vol_5d": None,
            "underlying_realized_vol_10d": None,
            "underlying_realized_vol_20d": None,
            "underlying_sma_20_distance_pct": None,
            "underlying_above_sma_20": None,
            "underlying_volatility_ratio_5d_20d": None,
            "underlying_volume": None,
            "underlying_skew_5d": None,
        }

    current = history[-1]
    previous = history[-2] if len(history) >= 2 else None
    prev_close = previous.close if previous else None
    return_1d = ((current.close / prev_close) - 1.0) if prev_close else None
    return_3d = _window_return(history, periods=3)
    return_5d = _window_return(history, periods=5)
    return_20d = _window_return(history, periods=20)
    returns = _close_returns(history)
    realized_vol_5d = _realized_vol(returns[-5:])
    realized_vol_10d = _realized_vol(returns[-10:])
    realized_vol_20d = _realized_vol(returns[-20:])
    skew_5d = _skewness(returns[-5:])
    sma_20_distance_pct = _sma_distance_pct(history, periods=20)
    return {
        "underlying_close": current.close,
        "underlying_return_1d": round(return_1d, 8) if return_1d is not None else None,
        "underlying_return_3d": round(return_3d, 8) if return_3d is not None else None,
        "underlying_return_5d": round(return_5d, 8) if return_5d is not None else None,
        "underlying_return_20d": round(return_20d, 8) if return_20d is not None else None,
        "underlying_range_pct": _range_pct(current.high, current.low, current.close),
        "underlying_realized_vol_5d": realized_vol_5d,
        "underlying_realized_vol_10d": realized_vol_10d,
        "underlying_realized_vol_20d": realized_vol_20d,
        "underlying_sma_20_distance_pct": sma_20_distance_pct,
        "underlying_above_sma_20": _above_sma(sma_20_distance_pct),
        "underlying_volatility_ratio_5d_20d": _volatility_ratio(realized_vol_5d, realized_vol_20d),
        "underlying_volume": current.volume,
        "underlying_skew_5d": skew_5d,
    }


def _market_regime_features(
    bars: list[PriceBar],
    entry_timestamp: datetime,
    market_regime_symbol: str,
) -> dict[str, Any]:
    features = _underlying_features(bars, entry_timestamp)
    market_sma_20_distance_pct = features["underlying_sma_20_distance_pct"]
    market_realized_vol_20d = features["underlying_realized_vol_20d"]
    return {
        "market_regime_symbol": market_regime_symbol.upper(),
        "market_return_5d": features["underlying_return_5d"],
        "market_return_20d": features["underlying_return_20d"],
        "market_realized_vol_5d": features["underlying_realized_vol_5d"],
        "market_realized_vol_20d": market_realized_vol_20d,
        "market_sma_20_distance_pct": market_sma_20_distance_pct,
        "market_above_sma_20": features["underlying_above_sma_20"],
        "market_volatility_ratio_5d_20d": features["underlying_volatility_ratio_5d_20d"],
        "market_trend_regime": _trend_regime(market_sma_20_distance_pct),
        "market_volatility_regime": _volatility_regime(market_realized_vol_20d),
    }


# ---------------------------------------------------------------------------
# Black-Scholes helpers (no external dependencies — pure math)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Black-Scholes European option price (call or put)."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (∂Price/∂sigma)."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return S * _norm_pdf(d1) * sqrt_T


def _implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> float | None:
    """Newton-Raphson implied volatility solver. Returns None if unconverged."""
    if T <= 0.0 or S <= 0.0 or K <= 0.0 or market_price <= 0.0:
        return None
    sigma = 0.25
    for _ in range(max_iterations):
        try:
            price = _bs_price(S, K, T, r, sigma, option_type)
            vega = _bs_vega(S, K, T, r, sigma)
        except (ValueError, ZeroDivisionError):
            return None
        if abs(vega) < 1e-10:
            return None
        diff = price - market_price
        if abs(diff) < tolerance:
            return round(max(0.001, min(sigma, 10.0)), 8)
        sigma = max(0.001, min(sigma - diff / vega, 10.0))
    return None


def _black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> dict[str, float | None]:
    """Compute delta, gamma, theta ($/day), vega (per 1% IV move)."""
    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        pdf_d1 = _norm_pdf(d1)
        discount = math.exp(-r * T)
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        if option_type == "call":
            delta = _norm_cdf(d1)
            theta = (-S * pdf_d1 * sigma / (2.0 * sqrt_T) - r * K * discount * _norm_cdf(d2)) / 365.0
        else:
            delta = _norm_cdf(d1) - 1.0
            theta = (-S * pdf_d1 * sigma / (2.0 * sqrt_T) + r * K * discount * _norm_cdf(-d2)) / 365.0
        vega_per_pct = S * pdf_d1 * sqrt_T / 100.0
        return {
            "delta": round(delta, 8),
            "gamma": round(gamma, 8),
            "theta": round(theta, 8),
            "vega": round(vega_per_pct, 8),
        }
    except (ValueError, ZeroDivisionError):
        return {"delta": None, "gamma": None, "theta": None, "vega": None}


def _option_greeks_features(
    entry_bar: PriceBar,
    underlying_close: float | None,
    strike: float | None,
    option_type: str | None,
    dte: int | None,
    risk_free_rate: float,
    realized_vol_5d: float | None,
    realized_vol_20d: float | None,
) -> dict[str, Any]:
    """Compute IV, Greeks, and IV/HV ratios for a candidate row."""
    _empty: dict[str, Any] = {
        "implied_volatility": None,
        "option_delta": None,
        "option_gamma": None,
        "option_theta": None,
        "option_vega": None,
        "iv_vs_hv5d": None,
        "iv_vs_hv20d": None,
    }
    if underlying_close is None or strike is None or option_type is None or dte is None or dte <= 0:
        return _empty
    T = dte / 365.0
    iv = _implied_volatility(float(entry_bar.close), float(underlying_close), float(strike), T, risk_free_rate, option_type)
    if iv is None:
        return _empty
    greeks = _black_scholes_greeks(float(underlying_close), float(strike), T, risk_free_rate, iv, option_type)
    iv_vs_hv5d = round(iv / realized_vol_5d, 8) if realized_vol_5d else None
    iv_vs_hv20d = round(iv / realized_vol_20d, 8) if realized_vol_20d else None
    return {
        "implied_volatility": iv,
        "option_delta": greeks["delta"],
        "option_gamma": greeks["gamma"],
        "option_theta": greeks["theta"],
        "option_vega": greeks["vega"],
        "iv_vs_hv5d": iv_vs_hv5d,
        "iv_vs_hv20d": iv_vs_hv20d,
    }


def _option_lookback_features(
    path: list[PriceBar],
    entry_timestamp: datetime,
    lookback_days: int = 5,
) -> dict[str, Any]:
    """Average volume and trade count from pre-entry option bars (last lookback_days)."""
    cutoff = entry_timestamp - timedelta(days=lookback_days)
    pre_entry = [b for b in path if cutoff <= b.timestamp < entry_timestamp]
    if not pre_entry:
        return {"option_volume_5d_avg": None, "option_trade_count_5d_avg": None}
    volumes = [b.volume for b in pre_entry if b.volume is not None]
    counts = [b.trade_count for b in pre_entry if b.trade_count is not None]
    return {
        "option_volume_5d_avg": round(sum(volumes) / len(volumes), 4) if volumes else None,
        "option_trade_count_5d_avg": round(sum(counts) / len(counts), 4) if counts else None,
    }


def _vix_features(
    vix_bars: list[PriceBar],
    entry_timestamp: datetime,
) -> dict[str, Any]:
    """VIX level, return, vol-of-vol, and threshold regime features."""
    history = [bar for bar in sorted(vix_bars, key=lambda b: b.timestamp) if bar.timestamp <= entry_timestamp]
    if not history:
        return {
            "vix_close": None,
            "vix_return_5d": None,
            "vix_realized_vol_5d": None,
            "vix_above_20": None,
            "vix_above_30": None,
        }
    vix_close = history[-1].close
    vix_return_5d = _window_return(history, periods=5)
    returns = _close_returns(history)
    vix_realized_vol_5d = _realized_vol(returns[-5:])
    return {
        "vix_close": vix_close,
        "vix_return_5d": round(vix_return_5d, 8) if vix_return_5d is not None else None,
        "vix_realized_vol_5d": vix_realized_vol_5d,
        "vix_above_20": int(vix_close >= 20.0),
        "vix_above_30": int(vix_close >= 30.0),
    }


def _event_features(
    earnings_events: list[EarningsEvent],
    entry_timestamp: datetime,
    forward_days: int,
) -> dict[str, Any]:
    """Days to next earnings and in-window flag."""
    entry_date = entry_timestamp.date()
    horizon = entry_date + timedelta(days=forward_days)
    upcoming = [e for e in earnings_events if e.report_date > entry_date]
    if not upcoming:
        return {"days_to_earnings": None, "has_earnings_in_forward_days": 0}
    nearest = min(upcoming, key=lambda e: e.report_date)
    return {
        "days_to_earnings": (nearest.report_date - entry_date).days,
        "has_earnings_in_forward_days": int(nearest.report_date <= horizon),
    }


def _dividend_features(
    dividend_events: list[DividendEvent],
    entry_timestamp: datetime,
    forward_days: int,
) -> dict[str, Any]:
    """Days to next ex-dividend date and in-window flag."""
    entry_date = entry_timestamp.date()
    horizon = entry_date + timedelta(days=forward_days)
    upcoming = [e for e in dividend_events if e.ex_date > entry_date]
    if not upcoming:
        return {"days_to_ex_dividend": None, "has_dividend_in_forward_days": 0}
    nearest = min(upcoming, key=lambda e: e.ex_date)
    return {
        "days_to_ex_dividend": (nearest.ex_date - entry_date).days,
        "has_dividend_in_forward_days": int(nearest.ex_date <= horizon),
    }


def _macro_event_features(
    fomc_events: list[EconomicEvent],
    all_macro_events: list[EconomicEvent],
    entry_timestamp: datetime,
    forward_days: int,
) -> dict[str, Any]:
    """Days to next FOMC and next CPI/NFP/GDP event, plus in-window flags.

    fomc_events: FOMC-only list (hardcoded calendar).
    all_macro_events: merged list of FOMC + FMP economic events.
    """
    entry_date = entry_timestamp.date()
    horizon = entry_date + timedelta(days=forward_days)

    upcoming_fomc = [e for e in fomc_events if e.event_date > entry_date]
    nearest_fomc = min(upcoming_fomc, key=lambda e: e.event_date) if upcoming_fomc else None

    upcoming_macro = [e for e in all_macro_events if e.event_date > entry_date]
    nearest_macro = min(upcoming_macro, key=lambda e: e.event_date) if upcoming_macro else None

    return {
        "days_to_fomc": (nearest_fomc.event_date - entry_date).days if nearest_fomc else None,
        "has_fomc_in_forward_days": int(nearest_fomc is not None and nearest_fomc.event_date <= horizon),
        "days_to_macro_event": (nearest_macro.event_date - entry_date).days if nearest_macro else None,
        "has_macro_event_in_forward_days": int(nearest_macro is not None and nearest_macro.event_date <= horizon),
    }


def _skewness(returns: list[float]) -> float | None:
    """Moment-based skewness of a return series. Returns None if < 3 observations."""
    n = len(returns)
    if n < 3:
        return None
    mean = sum(returns) / n
    variance = sum((x - mean) ** 2 for x in returns) / (n - 1)
    std = variance ** 0.5
    if std < 1e-10:
        return None
    third_moment = sum((x - mean) ** 3 for x in returns) / n
    return round(third_moment / (std ** 3), 8)


# ---------------------------------------------------------------------------
# Feature computation helpers (original)
# ---------------------------------------------------------------------------

def _window_return(history: list[PriceBar], periods: int) -> float | None:
    if len(history) <= periods:
        return None
    current = history[-1].close
    previous = history[-(periods + 1)].close
    if not previous:
        return None
    return (current / previous) - 1.0


def _close_returns(history: list[PriceBar]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(history, history[1:]):
        if previous.close:
            returns.append((current.close / previous.close) - 1.0)
    return returns


def _realized_vol(returns: list[float]) -> float | None:
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return round((variance ** 0.5) * (252 ** 0.5), 8)


def _sma_distance_pct(history: list[PriceBar], periods: int) -> float | None:
    if len(history) < periods:
        return None
    closes = [bar.close for bar in history[-periods:]]
    sma = sum(closes) / periods
    if not sma:
        return None
    return round((history[-1].close / sma) - 1.0, 8)


def _above_sma(sma_distance_pct: float | None) -> int | None:
    if sma_distance_pct is None:
        return None
    return int(sma_distance_pct > 0.0)


def _volatility_ratio(short_vol: float | None, long_vol: float | None) -> float | None:
    if short_vol is None or not long_vol:
        return None
    return round(short_vol / long_vol, 8)


def _trend_regime(sma_distance_pct: float | None) -> str | None:
    if sma_distance_pct is None:
        return None
    if sma_distance_pct >= 0.02:
        return "uptrend"
    if sma_distance_pct <= -0.02:
        return "downtrend"
    return "sideways"


def _volatility_regime(realized_vol_20d: float | None) -> str | None:
    if realized_vol_20d is None:
        return None
    if realized_vol_20d >= 0.30:
        return "high"
    if realized_vol_20d <= 0.15:
        return "low"
    return "normal"


def _range_pct(high: float | None, low: float | None, close: float | None) -> float | None:
    if high is None or low is None or not close:
        return None
    return round((float(high) - float(low)) / float(close), 8)


def _strike_distance_pct(strike: float | None, underlying_close: float | None) -> float | None:
    if strike is None or not underlying_close:
        return None
    return round((float(strike) - float(underlying_close)) / float(underlying_close), 8)


def _moneyness(strike: float | None, underlying_close: float | None) -> float | None:
    if strike is None or not underlying_close:
        return None
    return round(float(underlying_close) / float(strike), 8)


def _first_bar_at_or_after(path: list[PriceBar], timestamp: datetime) -> PriceBar | None:
    for bar in path:
        if bar.timestamp >= timestamp:
            return bar
    return None


def _dte(entry_timestamp: datetime, expiration: date | None) -> int | None:
    if expiration is None:
        return None
    return (expiration - entry_timestamp.date()).days


def _missing_fields(contract: OptionContract, features: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if contract.expiration is None:
        missing.append("expiration")
    if contract.strike is None:
        missing.append("strike")
    if contract.option_type is None:
        missing.append("option_type")
    for key, value in features.items():
        if value is None:
            missing.append(key)
    return missing


def _datetime_to_date(value: datetime) -> date:
    return value.date()


def market_open_utc(day: date) -> datetime:
    """Convenience helper for tests and scripts using regular U.S. market open.

    This is intentionally simple for the prototype. Later we should make market
    calendars explicit so holidays and daylight-saving shifts are handled by a
    real calendar provider.
    """
    return datetime.combine(day, time(13, 30), tzinfo=UTC)
