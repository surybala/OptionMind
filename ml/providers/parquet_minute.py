"""Local parquet-backed minute-bar provider built from ingested flat files."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ml.providers.models import PriceBar


@dataclass(frozen=True)
class ParquetMinuteBarProvider:
    """Serve minute bars from a local `massive_flatfiles` parquet dataset."""

    dataset_root: Path | str

    def get_stock_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        return self._get_bars("stocks", symbols, start, end, timeframe)

    def get_option_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
        limit: int | None = None,
    ) -> dict[str, list[PriceBar]]:
        result = self._get_bars("options", symbols, start, end, timeframe)
        if limit is None:
            return result
        return {symbol: bars[:limit] for symbol, bars in result.items()}

    def _get_bars(
        self,
        asset_class: str,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        if timeframe.strip().lower() not in {"1min", "minute", "min", "1m"}:
            raise ValueError(f"ParquetMinuteBarProvider only supports minute bars, got {timeframe}")
        root = Path(self.dataset_root)
        normalized_symbols = {_normalize_ticker(asset_class, symbol) for symbol in symbols}
        files = _partition_files(root, asset_class, normalized_symbols, start.date(), end.date())
        if not files:
            return {symbol: [] for symbol in normalized_symbols}
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        frame["window_start"] = pd.to_datetime(frame["window_start"], utc=True, errors="coerce")
        frame = frame[
            (frame["ticker"].isin(normalized_symbols))
            & (frame["window_start"] >= start)
            & (frame["window_start"] <= end)
        ].sort_values(["ticker", "window_start"])
        by_symbol: dict[str, list[PriceBar]] = {symbol: [] for symbol in normalized_symbols}
        for row in frame.to_dict(orient="records"):
            ticker = str(row["ticker"])
            by_symbol.setdefault(ticker, []).append(
                PriceBar(
                    symbol=ticker,
                    timestamp=_coerce_timestamp(row["window_start"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]) if row.get("volume") is not None else None,
                    trade_count=int(row["transactions"]) if row.get("transactions") is not None else None,
                    vwap=None,
                    source=str(row.get("source") or "massive_flatfiles"),
                )
            )
        return by_symbol


def _normalize_ticker(asset_class: str, symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if asset_class == "options" and raw.startswith("O:"):
        return raw[2:]
    return raw


def _coerce_timestamp(value) -> datetime:
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _partition_files(
    root: Path,
    asset_class: str,
    tickers: set[str],
    start_date: date,
    end_date: date,
) -> list[Path]:
    files: list[Path] = []
    current = start_date
    while current <= end_date:
        day_str = current.isoformat()
        for ticker in tickers:
            part_dir = (
                root
                / f"asset_class={asset_class}"
                / f"ticker={ticker}"
                / f"window_date={day_str}"
            )
            if part_dir.exists():
                files.extend(sorted(part_dir.glob("part-*.parquet")))
        current += timedelta(days=1)
    return files
