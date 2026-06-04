"""Build intraday spread-state rows for live risk model training."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import math
import pandas as pd

from ml.datasets.audit_candidate_dataset import load_dataset
from ml.labels import CreditSpreadLabelConfig, label_credit_spread_path
from ml.providers.models import PriceBar


@dataclass(frozen=True)
class IntradayRiskDatasetConfig:
    option_timeframe: str = "1Min"
    stock_timeframe: str = "1Min"
    lookback_minutes: int = 30
    sample_every_n_candidates: int = 1
    sample_every_n_minutes: int = 5
    max_candidates: int | None = None
    max_forward_days: int = 30
    min_state_rows_per_candidate: int = 3
    max_workers: int = 8
    profit_take_pct: float = 0.75
    stop_loss_multiple: float = 2.0
    stop_loss_max_loss_pct: float | None = 0.80
    large_loss_multiple: float = 1.0


@dataclass(frozen=True)
class IntradayRiskRow:
    entry_timestamp: datetime
    state_timestamp: datetime
    state_date: date
    underlying: str
    strategy: str
    source: str
    market_regime_symbol: str | None
    market_trend_regime: str | None
    market_volatility_regime: str | None
    expiration: date | None
    dte: int | None
    short_option_symbol: str
    long_option_symbol: str
    spread_width: float
    entry_credit: float
    max_loss: float
    stop_debit: float
    profit_take_debit: float
    current_debit: float
    pnl_per_contract: float
    profit_captured_pct: float | None
    stop_distance_pct: float | None
    minutes_since_entry: float
    minutes_to_expiry: float | None
    minutes_to_exit: float
    underlying_close: float
    underlying_return_5m: float | None
    underlying_return_15m: float | None
    underlying_return_30m: float | None
    underlying_realized_vol_15m: float | None
    underlying_realized_vol_30m: float | None
    short_leg_close: float
    long_leg_close: float
    short_leg_volume: float | None
    long_leg_volume: float | None
    short_leg_trade_count: int | None
    long_leg_trade_count: int | None
    future_worst_debit_5m: float | None
    future_worst_debit_15m: float | None
    future_worst_debit_30m: float | None
    future_best_debit_5m: float | None
    future_best_debit_15m: float | None
    future_best_debit_30m: float | None
    adverse_move_5m: float | None
    adverse_move_15m: float | None
    adverse_move_30m: float | None
    favorable_move_5m: float | None
    favorable_move_15m: float | None
    favorable_move_30m: float | None
    stop_loss_hit_5m: int
    stop_loss_hit_15m: int
    stop_loss_hit_30m: int
    profit_take_hit_5m: int
    profit_take_hit_15m: int
    profit_take_hit_30m: int
    intraday_exit_timestamp: datetime
    intraday_exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IntradayRiskDatasetBuilder:
    """Build minute-level spread states from a seed candidate dataset."""

    def __init__(self, market_provider, option_price_provider) -> None:
        self.market_provider = market_provider
        self.option_price_provider = option_price_provider

    def build(
        self,
        seed_path,
        config: IntradayRiskDatasetConfig,
        *,
        entry_start: datetime | None = None,
        entry_end: datetime | None = None,
        strategy_types: tuple[str, ...] = ("PCS", "CCS"),
    ) -> list[IntradayRiskRow]:
        seed_df = load_dataset(seed_path if isinstance(seed_path, Path) else Path(seed_path))
        seeds = _seed_candidates(
            seed_df,
            entry_start=entry_start,
            entry_end=entry_end,
            strategy_types=strategy_types,
            sample_every_n_candidates=config.sample_every_n_candidates,
            max_candidates=config.max_candidates,
        )
        if not seeds:
            return []

        rows: list[IntradayRiskRow] = []
        total = len(seeds)
        print(
            f"Building intraday risk dataset from {total} seed candidate(s) "
            f"with {config.max_workers} worker thread(s) ...",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_map = {
                executor.submit(self._process_seed, seed, config): seed
                for seed in seeds
            }
            completed = 0
            for future in as_completed(future_map):
                completed += 1
                seed = future_map[future]
                try:
                    built = future.result()
                except Exception as exc:
                    print(
                        f"  [{completed}/{total}] {seed['underlying']} {seed['entry_timestamp'].isoformat()} FAILED: {exc}",
                        flush=True,
                    )
                    continue
                rows.extend(built)
                print(
                    f"  [{completed}/{total}] {seed['underlying']} {seed['entry_timestamp'].isoformat()} "
                    f"-> {len(built)} rows (running total: {len(rows)})",
                    flush=True,
                )
        return rows

    def _process_seed(
        self,
        seed: dict[str, Any],
        config: IntradayRiskDatasetConfig,
    ) -> list[IntradayRiskRow]:
        entry_timestamp = seed["entry_timestamp"]
        exit_limit = min(
            seed.get("exit_timestamp") or (entry_timestamp + timedelta(days=config.max_forward_days)),
            entry_timestamp + timedelta(days=config.max_forward_days),
        )
        max_horizon = timedelta(minutes=30)
        fetch_start = entry_timestamp - timedelta(minutes=config.lookback_minutes)
        fetch_end = exit_limit + max_horizon

        short_symbol = seed["short_option_symbol"]
        long_symbol = seed["long_option_symbol"]
        underlying = seed["underlying"]
        option_bars = self.option_price_provider.get_option_bars(
            [short_symbol, long_symbol],
            fetch_start,
            fetch_end,
            config.option_timeframe,
            limit=None,
        )
        stock_bars = self.market_provider.get_stock_bars(
            [underlying],
            fetch_start,
            fetch_end,
            config.stock_timeframe,
        )
        short_path = sorted(option_bars.get(short_symbol, []), key=lambda bar: bar.timestamp)
        long_path = sorted(option_bars.get(long_symbol, []), key=lambda bar: bar.timestamp)
        underlying_path = sorted(stock_bars.get(underlying, []), key=lambda bar: bar.timestamp)
        if not short_path or not long_path or not underlying_path:
            return []

        aligned_states = _aligned_state_path(
            short_path,
            long_path,
            underlying_path,
            start=fetch_start,
            end=fetch_end,
        )
        if not aligned_states:
            return []
        eligible_states = [state for state in aligned_states if entry_timestamp <= state[0] <= exit_limit + max_horizon]
        if not eligible_states:
            return []
        entry_timestamp = eligible_states[0][0]
        short_entry_bar = eligible_states[0][1]
        long_entry_bar = eligible_states[0][2]
        aligned_short_path = [state[1] for state in eligible_states]
        aligned_long_path = [state[2] for state in eligible_states]
        aligned_stock_path = [state[3] for state in aligned_states]

        spread_width = float(seed["spread_width"])
        label = label_credit_spread_path(
            strategy=seed["strategy"],
            short_entry_bar=short_entry_bar,
            long_entry_bar=long_entry_bar,
            short_path=aligned_short_path,
            long_path=aligned_long_path,
            spread_width=spread_width,
            config=CreditSpreadLabelConfig(
                profit_take_pct=config.profit_take_pct,
                stop_loss_multiple=config.stop_loss_multiple,
                stop_loss_max_loss_pct=config.stop_loss_max_loss_pct,
                large_loss_multiple=config.large_loss_multiple,
            ),
        )
        stop_debit = _stop_debit_for_spread(
            float(label.entry_credit),
            spread_width,
            config.stop_loss_multiple,
            config.stop_loss_max_loss_pct,
        )

        active_states = [state for state in eligible_states if entry_timestamp <= state[0] <= label.exit_timestamp]
        if len(active_states) < config.min_state_rows_per_candidate:
            return []
        state_timestamps = [state[0] for state in active_states]
        short_by_time = {state[0]: state[1] for state in active_states}
        long_by_time = {state[0]: state[2] for state in active_states}
        stock_by_time = {state[0]: state[3] for state in active_states}

        rows: list[IntradayRiskRow] = []
        step = max(1, int(config.sample_every_n_minutes))
        for timestamp in state_timestamps[::step]:
            current_short = short_by_time[timestamp]
            current_long = long_by_time[timestamp]
            current_stock = stock_by_time[timestamp]
            current_debit = _spread_debit(current_short.close, current_long.close, spread_width)
            future = _future_window_metrics(
                state_timestamps,
                short_by_time,
                long_by_time,
                timestamp,
                spread_width,
                stop_debit=stop_debit,
                profit_take_debit=label.entry_credit * (1.0 - config.profit_take_pct),
            )
            minutes_since_entry = round((timestamp - entry_timestamp).total_seconds() / 60.0, 4)
            minutes_to_expiry = _minutes_to_expiry(timestamp, seed.get("expiration"))
            max_loss = float(label.max_loss)
            profit_captured_pct = (
                round((label.entry_credit - current_debit) / label.entry_credit * 100.0, 4)
                if label.entry_credit > 0
                else None
            )
            stop_distance_pct = (
                round((future["stop_debit"] - current_debit) / future["stop_debit"] * 100.0, 4)
                if future["stop_debit"] > 0
                else None
            )
            rows.append(
                IntradayRiskRow(
                    entry_timestamp=entry_timestamp,
                    state_timestamp=timestamp,
                    state_date=timestamp.date(),
                    underlying=underlying,
                    strategy=seed["strategy"],
                    source=str(seed.get("source") or "massive"),
                    market_regime_symbol=_optional_string(seed.get("market_regime_symbol")),
                    market_trend_regime=_optional_string(seed.get("market_trend_regime")),
                    market_volatility_regime=_optional_string(seed.get("market_volatility_regime")),
                    expiration=seed.get("expiration"),
                    dte=_dte(timestamp, seed.get("expiration")),
                    short_option_symbol=short_symbol,
                    long_option_symbol=long_symbol,
                    spread_width=spread_width,
                    entry_credit=round(float(label.entry_credit), 8),
                    max_loss=max_loss,
                    stop_debit=future["stop_debit"],
                    profit_take_debit=future["profit_take_debit"],
                    current_debit=current_debit,
                    pnl_per_contract=round((label.entry_credit - current_debit) * 100.0, 4),
                    profit_captured_pct=profit_captured_pct,
                    stop_distance_pct=stop_distance_pct,
                    minutes_since_entry=minutes_since_entry,
                    minutes_to_expiry=minutes_to_expiry,
                    minutes_to_exit=round((label.exit_timestamp - timestamp).total_seconds() / 60.0, 4),
                    underlying_close=float(current_stock.close),
                    underlying_return_5m=_trailing_return(aligned_stock_path, timestamp, 5),
                    underlying_return_15m=_trailing_return(aligned_stock_path, timestamp, 15),
                    underlying_return_30m=_trailing_return(aligned_stock_path, timestamp, 30),
                    underlying_realized_vol_15m=_trailing_realized_vol(aligned_stock_path, timestamp, 15),
                    underlying_realized_vol_30m=_trailing_realized_vol(aligned_stock_path, timestamp, 30),
                    short_leg_close=float(current_short.close),
                    long_leg_close=float(current_long.close),
                    short_leg_volume=current_short.volume,
                    long_leg_volume=current_long.volume,
                    short_leg_trade_count=current_short.trade_count,
                    long_leg_trade_count=current_long.trade_count,
                    future_worst_debit_5m=future["worst_5m"],
                    future_worst_debit_15m=future["worst_15m"],
                    future_worst_debit_30m=future["worst_30m"],
                    future_best_debit_5m=future["best_5m"],
                    future_best_debit_15m=future["best_15m"],
                    future_best_debit_30m=future["best_30m"],
                    adverse_move_5m=_future_delta(future["worst_5m"], current_debit),
                    adverse_move_15m=_future_delta(future["worst_15m"], current_debit),
                    adverse_move_30m=_future_delta(future["worst_30m"], current_debit),
                    favorable_move_5m=_future_delta(current_debit, future["best_5m"]),
                    favorable_move_15m=_future_delta(current_debit, future["best_15m"]),
                    favorable_move_30m=_future_delta(current_debit, future["best_30m"]),
                    stop_loss_hit_5m=future["stop_loss_hit_5m"],
                    stop_loss_hit_15m=future["stop_loss_hit_15m"],
                    stop_loss_hit_30m=future["stop_loss_hit_30m"],
                    profit_take_hit_5m=future["profit_take_hit_5m"],
                    profit_take_hit_15m=future["profit_take_hit_15m"],
                    profit_take_hit_30m=future["profit_take_hit_30m"],
                    intraday_exit_timestamp=label.exit_timestamp,
                    intraday_exit_reason=label.exit_reason,
                )
            )
        return rows


def _aligned_state_path(
    short_path: list[PriceBar],
    long_path: list[PriceBar],
    stock_path: list[PriceBar],
    *,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, PriceBar, PriceBar, PriceBar]]:
    short_sorted = sorted((bar for bar in short_path if bar.timestamp <= end), key=lambda bar: bar.timestamp)
    long_sorted = sorted((bar for bar in long_path if bar.timestamp <= end), key=lambda bar: bar.timestamp)
    stock_sorted = sorted((bar for bar in stock_path if bar.timestamp <= end), key=lambda bar: bar.timestamp)
    timeline = sorted(
        {
            bar.timestamp
            for bar in [*short_sorted, *long_sorted, *stock_sorted]
            if start <= bar.timestamp <= end
        }
    )
    states: list[tuple[datetime, PriceBar, PriceBar, PriceBar]] = []
    short_index = 0
    long_index = 0
    stock_index = 0
    current_short: PriceBar | None = None
    current_long: PriceBar | None = None
    current_stock: PriceBar | None = None
    for timestamp in timeline:
        while short_index < len(short_sorted) and short_sorted[short_index].timestamp <= timestamp:
            current_short = short_sorted[short_index]
            short_index += 1
        while long_index < len(long_sorted) and long_sorted[long_index].timestamp <= timestamp:
            current_long = long_sorted[long_index]
            long_index += 1
        while stock_index < len(stock_sorted) and stock_sorted[stock_index].timestamp <= timestamp:
            current_stock = stock_sorted[stock_index]
            stock_index += 1
        if current_short is None or current_long is None or current_stock is None:
            continue
        states.append(
            (
                timestamp,
                _clone_bar_with_timestamp(current_short, timestamp),
                _clone_bar_with_timestamp(current_long, timestamp),
                _clone_bar_with_timestamp(current_stock, timestamp),
            )
        )
    return states


def _clone_bar_with_timestamp(bar: PriceBar, timestamp: datetime) -> PriceBar:
    return PriceBar(
        symbol=bar.symbol,
        timestamp=timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        trade_count=bar.trade_count,
        vwap=bar.vwap,
        source=bar.source,
    )


def _seed_candidates(
    df: pd.DataFrame,
    *,
    entry_start: datetime | None,
    entry_end: datetime | None,
    strategy_types: tuple[str, ...],
    sample_every_n_candidates: int,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    frame = df.copy()
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True, errors="coerce")
    frame["exit_timestamp"] = pd.to_datetime(frame.get("exit_timestamp"), utc=True, errors="coerce")
    allowed = {item.upper() for item in strategy_types}
    frame = frame[frame["strategy"].isin(allowed)]
    frame = frame.dropna(
        subset=[
            "entry_timestamp",
            "underlying",
            "strategy",
            "short_option_symbol",
            "long_option_symbol",
            "spread_width",
        ]
    )
    if entry_start is not None:
        frame = frame[frame["entry_timestamp"] >= pd.Timestamp(entry_start)]
    if entry_end is not None:
        frame = frame[frame["entry_timestamp"] <= pd.Timestamp(entry_end)]
    frame = frame.sort_values("entry_timestamp")
    if sample_every_n_candidates > 1:
        frame = frame.iloc[::sample_every_n_candidates]
    if max_candidates is not None:
        frame = frame.head(max_candidates)
    seeds: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        row["entry_timestamp"] = _coerce_timestamp(row["entry_timestamp"])
        row["exit_timestamp"] = _coerce_timestamp(row.get("exit_timestamp"))
        row["expiration"] = _coerce_date(row.get("expiration"))
        seeds.append(row)
    return seeds


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts.to_pydatetime()


def _coerce_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def _optional_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _future_window_metrics(
    timestamps: list[datetime],
    short_by_time: dict[datetime, PriceBar],
    long_by_time: dict[datetime, PriceBar],
    current_timestamp: datetime,
    spread_width: float,
    *,
    stop_debit: float,
    profit_take_debit: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "stop_debit": round(stop_debit, 8),
        "profit_take_debit": round(profit_take_debit, 8),
    }
    for minutes in (5, 15, 30):
        cutoff = current_timestamp + timedelta(minutes=minutes)
        debits = [
            _spread_debit(short_by_time[timestamp].close, long_by_time[timestamp].close, spread_width)
            for timestamp in timestamps
            if current_timestamp < timestamp <= cutoff
        ]
        key = f"{minutes}m"
        metrics[f"worst_{key}"] = round(max(debits), 8) if debits else None
        metrics[f"best_{key}"] = round(min(debits), 8) if debits else None
        metrics[f"stop_loss_hit_{key}"] = int(any(value >= stop_debit for value in debits))
        metrics[f"profit_take_hit_{key}"] = int(any(value <= profit_take_debit for value in debits))
    return metrics


def _spread_debit(short_close: float, long_close: float, spread_width: float) -> float:
    return round(max(0.0, min(float(spread_width), float(short_close) - float(long_close))), 8)


def _stop_debit_for_spread(
    entry_credit: float,
    spread_width: float,
    stop_loss_multiple: float,
    stop_loss_max_loss_pct: float | None,
) -> float:
    if stop_loss_max_loss_pct is not None:
        return round(entry_credit + (spread_width - entry_credit) * stop_loss_max_loss_pct, 8)
    return round(entry_credit * stop_loss_multiple, 8)


def _trailing_return(path: list[PriceBar], timestamp: datetime, lookback_minutes: int) -> float | None:
    current = next((bar for bar in path if bar.timestamp == timestamp), None)
    past_cutoff = timestamp - timedelta(minutes=lookback_minutes)
    past = None
    for bar in path:
        if bar.timestamp <= past_cutoff:
            past = bar
        if bar.timestamp >= timestamp:
            break
    if current is None or past is None or past.close <= 0:
        return None
    return round((current.close / past.close) - 1.0, 8)


def _trailing_realized_vol(path: list[PriceBar], timestamp: datetime, lookback_minutes: int) -> float | None:
    cutoff = timestamp - timedelta(minutes=lookback_minutes)
    closes = [bar.close for bar in path if cutoff <= bar.timestamp <= timestamp and bar.close > 0]
    if len(closes) < 3:
        return None
    returns = [math.log(closes[idx] / closes[idx - 1]) for idx in range(1, len(closes))]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return round(math.sqrt(variance) * math.sqrt(252.0 * 390.0), 8)


def _future_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 8)


def _minutes_to_expiry(timestamp: datetime, expiration: date | None) -> float | None:
    if expiration is None:
        return None
    expiry_timestamp = datetime.combine(expiration, datetime.min.time(), tzinfo=UTC) + timedelta(hours=20)
    return round((expiry_timestamp - timestamp).total_seconds() / 60.0, 4)


def _dte(timestamp: datetime, expiration: date | None) -> int | None:
    if expiration is None:
        return None
    return (expiration - timestamp.date()).days
