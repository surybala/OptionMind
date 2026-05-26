"""yfinance-backed stock bar provider for historical daily OHLCV data.

yfinance wraps Yahoo Finance — free, no API key required, covers US equities
and ETFs going back decades.  Use it as the ``MarketDataProvider`` when you
need stock bars that predate the Massive/Polygon 2-year retention window.

Data quality notes
------------------
- Yahoo Finance adjusts prices for splits and dividends automatically.
- Bars are returned as of the **local market close** (UTC timestamp varies
  by exchange; US ETFs land at 20:00 UTC / 4 PM ET).
- Volume is share-based (not dollar-based), consistent with other providers.

Caching
-------
Raw downloads are cached to ``artifacts/cache/yfinance/`` (configurable via
``YFINANCE_CACHE_DIR`` env var or the ``cache_dir`` constructor argument).
Each file is keyed by SHA-256 of ``(symbol, start_date, end_date)`` so that
different date ranges get separate cache entries.  Cache files are stable:
historical Yahoo data does not change retroactively.

Usage
-----
    from ml.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    bars = provider.get_stock_bars(["SPY", "QQQ"], start, end, "1Day")
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ml.providers.models import PriceBar

_DEFAULT_CACHE_DIR = "artifacts/cache/yfinance"

# yfinance timeframe aliases accepted by this provider.
_SUPPORTED_TIMEFRAMES = {"1day", "1/day", "day", "1d"}


@dataclass
class YFinanceProvider:
    """MarketDataProvider backed by yfinance (Yahoo Finance).

    Implements the ``MarketDataProvider`` protocol — only ``get_stock_bars``
    is provided.  For option data use MassiveProvider or AlpacaProvider.

    Parameters
    ----------
    cache_dir:
        Directory for disk-cached raw downloads.  Defaults to
        ``YFINANCE_CACHE_DIR`` env var or ``artifacts/cache/yfinance``.
    source:
        Source label embedded in returned ``PriceBar`` objects.
    """

    cache_dir: Path | str | None = None
    source: str = "yfinance"

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = os.getenv("YFINANCE_CACHE_DIR", _DEFAULT_CACHE_DIR)

    # ------------------------------------------------------------------
    # MarketDataProvider protocol
    # ------------------------------------------------------------------

    def get_stock_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> dict[str, list[PriceBar]]:
        """Return normalized daily bars for *symbols* between *start* and *end*.

        Only daily bars are currently supported.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["SPY", "QQQ"]``).
        start:
            Inclusive start datetime (timezone-aware or naive UTC).
        end:
            Inclusive end datetime.
        timeframe:
            Must be a daily-bar alias: ``"1Day"``, ``"1/day"``, ``"day"``,
            or ``"1d"``.  Other values raise ``NotImplementedError``.

        Returns
        -------
        dict[str, list[PriceBar]]
            Keys are uppercased symbols; values are sorted ascending by
            timestamp.  Missing symbols return an empty list.
        """
        if timeframe.lower().replace(" ", "").replace("/", "") not in {
            t.replace("/", "").replace(" ", "") for t in _SUPPORTED_TIMEFRAMES
        }:
            raise NotImplementedError(
                f"YFinanceProvider only supports daily bars; got timeframe={timeframe!r}"
            )

        result: dict[str, list[PriceBar]] = {}
        for symbol in symbols:
            upper = symbol.upper()
            result[upper] = self._fetch_symbol(upper, start, end)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_symbol(self, symbol: str, start: datetime, end: datetime) -> list[PriceBar]:
        """Fetch (from cache or Yahoo) and return sorted PriceBars for one symbol."""
        cache_path = self._cache_path(symbol, start, end)

        if cache_path is not None and cache_path.exists():
            raw: list[dict] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = self._download(symbol, start, end)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")

        return [
            PriceBar(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(bar["t"] / 1000, tz=UTC),
                open=bar["o"],
                high=bar["h"],
                low=bar["l"],
                close=bar["c"],
                volume=bar.get("v"),
                source=self.source,
            )
            for bar in raw
        ]

    def _download(self, symbol: str, start: datetime, end: datetime) -> list[dict]:
        """Download from Yahoo Finance and return a list of raw bar dicts."""
        import yfinance as yf  # lazy import — not a hard dependency at module level

        # yfinance end is exclusive, so add 1 day to include the end date's bar.
        yf_start = start.date().isoformat()
        yf_end = (end.date() + timedelta(days=1)).isoformat()

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=yf_start,
            end=yf_end,
            interval="1d",
            auto_adjust=True,
            actions=False,
            raise_errors=False,
        )
        if df is None or df.empty:
            return []

        # Normalize index to UTC midnight-ish timestamps (Yahoo returns
        # timezone-aware DatetimeIndex for US symbols).
        rows: list[dict] = []
        for idx, row in df.iterrows():
            try:
                ts_ms = int(idx.timestamp() * 1000)
            except Exception:
                continue
            rows.append(
                {
                    "t": ts_ms,
                    "o": float(row["Open"]),
                    "h": float(row["High"]),
                    "l": float(row["Low"]),
                    "c": float(row["Close"]),
                    "v": float(row["Volume"]),
                }
            )
        return rows

    def _cache_path(self, symbol: str, start: datetime, end: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps(
            {
                "symbol": symbol,
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / f"{digest}.json"
