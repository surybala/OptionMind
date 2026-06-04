"""Local parquet-backed minute-bar provider built from ingested flat files."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
import re

import pandas as pd

from ml.providers.models import OptionContract, PriceBar
from src.osi import parse_osi


_OPTION_TICKER_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


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

    def get_option_contracts(
        self,
        underlyings: list[str],
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
        status: str = "active",
        limit: int | None = None,
    ) -> list[OptionContract]:
        contracts: list[OptionContract] = []
        for underlying in underlyings:
            contracts.extend(
                _contract
                for _contract in self._contracts_for_underlying(underlying)
                if _contract.expiration is not None
                and (expiration_gte is None or _contract.expiration >= expiration_gte)
                and (expiration_lte is None or _contract.expiration <= expiration_lte)
                and _contract.underlying == underlying.upper()
                and _contract.status in _allowed_statuses(status)
            )
        contracts.sort(key=_contract_sort_key)
        if limit is None:
            return contracts
        return contracts[:limit]

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

    def _contracts_for_underlying(self, underlying: str) -> list[OptionContract]:
        return list(_load_contracts_from_parquet(str(self.dataset_root), underlying.upper()))


@lru_cache(maxsize=256)
def _load_contracts_from_parquet(dataset_root: str, underlying: str) -> tuple[OptionContract, ...]:
    root = Path(dataset_root) / "asset_class=options" / f"underlying={underlying}"
    if not root.exists():
        return ()

    parquet_files = sorted(root.glob("window_date=*/part-*.parquet"))
    if not parquet_files:
        return ()

    contracts: dict[str, OptionContract] = {}
    for path in parquet_files:
        frame = pd.read_parquet(path, columns=["ticker"])
        for ticker in frame["ticker"].dropna().astype(str).unique():
            normalized = _normalize_ticker("options", ticker)
            if normalized in contracts:
                continue
            parsed = parse_osi(normalized)
            if parsed is None:
                continue
            contracts[normalized] = OptionContract(
                symbol=normalized,
                underlying=parsed.underlying,
                expiration=parsed.expiration,
                strike=parsed.strike,
                option_type=parsed.option_type,
                status=_status_from_expiration(parsed.expiration),
                root_symbol=parsed.underlying,
                source="massive_flatfiles",
            )
    return tuple(contracts.values())


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
    seen: set[Path] = set()
    current = start_date
    while current <= end_date:
        day_str = current.isoformat()
        for ticker in tickers:
            candidate_dirs = [
                root
                / f"asset_class={asset_class}"
                / f"ticker={ticker}"
                / f"window_date={day_str}"
            ]
            if asset_class == "options":
                underlying = _option_underlying_from_ticker(ticker)
                if underlying:
                    candidate_dirs.append(
                        root
                        / f"asset_class={asset_class}"
                        / f"underlying={underlying}"
                        / f"window_date={day_str}"
                    )
            for part_dir in candidate_dirs:
                if not part_dir.exists():
                    continue
                for path in sorted(part_dir.glob("part-*.parquet")):
                    if path in seen:
                        continue
                    seen.add(path)
                    files.append(path)
        current += timedelta(days=1)
    return files


def _option_underlying_from_ticker(symbol: str) -> str | None:
    raw = _normalize_ticker("options", symbol)
    match = _OPTION_TICKER_RE.match(raw)
    if not match:
        return None
    return match.group(1)


def _allowed_statuses(status: str) -> set[str]:
    normalized = str(status or "active").strip().lower()
    if normalized == "all":
        return {"active", "inactive", "unknown"}
    return {normalized}


def _status_from_expiration(expiration: date) -> str:
    return "active" if expiration >= datetime.now(UTC).date() else "inactive"


def _contract_sort_key(contract: OptionContract) -> tuple[date, str, float, str]:
    return (
        contract.expiration or date.min,
        contract.option_type or "",
        float(contract.strike or 0.0),
        contract.symbol,
    )
