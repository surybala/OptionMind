"""yfinance-backed provider for historical daily stock bars and earnings calendar.

yfinance wraps Yahoo Finance — free, no API key required, covers US equities
and ETFs going back decades.  Use it as the ``MarketDataProvider`` when you
need stock bars that predate the Massive/Polygon 2-year retention window, or
as the ``EventDataProvider`` when you want earnings dates without an FMP key.

Data quality notes
------------------
- Yahoo Finance adjusts prices for splits and dividends automatically.
- Bars are returned as of the **local market close** (UTC timestamp varies
  by exchange; US ETFs land at 20:00 UTC / 4 PM ET).
- Volume is share-based (not dollar-based), consistent with other providers.
- Earnings dates: Yahoo Finance carries roughly 6 years of history per ticker
  (configurable via ``earnings_lookback``).  Future dates are estimates until
  confirmed by the company's IR filing.

Caching
-------
Raw downloads are cached to ``artifacts/cache/yfinance/`` (configurable via
``YFINANCE_CACHE_DIR`` env var or the ``cache_dir`` constructor argument).
Stock bar files are keyed by SHA-256 of ``(symbol, start_date, end_date)``.
Earnings files are keyed by SHA-256 of ``(symbol, as_of_date)`` so the cache
refreshes daily — protecting against stale future-earnings estimates.

Usage
-----
    from ml.providers.yfinance_provider import YFinanceProvider

    provider = YFinanceProvider()
    bars = provider.get_stock_bars(["SPY", "QQQ"], start, end, "1Day")
    events = provider.get_earnings_calendar(["AAPL", "MSFT"], start.date(), end.date())
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from ml.providers.models import EarningsEvent, PriceBar

_DEFAULT_CACHE_DIR = "artifacts/cache/yfinance"

# yfinance timeframe aliases accepted by this provider.
_SUPPORTED_TIMEFRAMES = {"1day", "1/day", "day", "1d"}


@dataclass
class YFinanceProvider:
    """MarketDataProvider and EventDataProvider backed by yfinance (Yahoo Finance).

    Implements ``MarketDataProvider`` (stock bars) and ``EventDataProvider``
    (earnings calendar).  For option data use MassiveProvider or AlpacaProvider.

    Parameters
    ----------
    cache_dir:
        Directory for disk-cached raw downloads.  Defaults to
        ``YFINANCE_CACHE_DIR`` env var or ``artifacts/cache/yfinance``.
    source:
        Source label embedded in returned objects.
    earnings_lookback:
        Maximum number of earnings dates to fetch per ticker.  Each quarter
        counts as one entry, so 24 covers ~6 years for quarterly reporters —
        enough for the full V006 build window (2022-2026).
    """

    cache_dir: Path | str | None = None
    source: str = "yfinance"
    earnings_lookback: int = 24

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
    # EventDataProvider protocol
    # ------------------------------------------------------------------

    def get_earnings_calendar(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> dict[str, list[EarningsEvent]]:
        """Return earnings events keyed by symbol from Yahoo Finance.

        Yahoo Finance carries roughly ``earnings_lookback`` quarters of history
        plus 1-2 upcoming quarters per ticker.  Future dates are estimates and
        may shift slightly; the cache refreshes daily to pick up any changes.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
        start:
            Inclusive start date for filtering returned events.
        end:
            Inclusive end date for filtering returned events.

        Returns
        -------
        dict[str, list[EarningsEvent]]
            Keys are uppercased symbols; values are sorted ascending by
            report_date within the requested window.
        """
        today = date.today().isoformat()
        result: dict[str, list[EarningsEvent]] = {}
        for symbol in symbols:
            upper = symbol.upper()
            result[upper] = self._fetch_earnings(upper, start, end, today)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_earnings(
        self, symbol: str, start: date, end: date, as_of: str
    ) -> list[EarningsEvent]:
        """Return EarningsEvents for *symbol* filtered to [start, end]."""
        cache_path = self._earnings_cache_path(symbol, as_of)

        if cache_path is not None and cache_path.exists():
            raw: list[dict] = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            raw = self._download_earnings(symbol)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")

        events: list[EarningsEvent] = []
        for item in raw:
            try:
                report_date = date.fromisoformat(item["date"])
            except (KeyError, ValueError):
                continue
            if start <= report_date <= end:
                events.append(
                    EarningsEvent(
                        symbol=symbol,
                        report_date=report_date,
                        fiscal_period=item.get("fiscal_period"),
                        source=self.source,
                    )
                )
        events.sort(key=lambda e: e.report_date)
        return events

    def _download_earnings(self, symbol: str) -> list[dict]:
        """Download earnings dates from Yahoo Finance for *symbol*."""
        import yfinance as yf  # lazy import — not a hard dependency at module level

        ticker = yf.Ticker(symbol)
        try:
            df = ticker.get_earnings_dates(limit=self.earnings_lookback)
        except Exception:
            df = getattr(ticker, "earnings_dates", None)

        if df is None or df.empty:
            return []

        rows: list[dict] = []
        for idx in df.index:
            try:
                d = idx.date().isoformat()
            except Exception:
                try:
                    d = str(idx)[:10]
                except Exception:
                    continue
            rows.append({"date": d})
        return rows

    def _earnings_cache_path(self, symbol: str, as_of: str) -> Path | None:
        if self.cache_dir is None:
            return None
        key = json.dumps({"symbol": symbol, "as_of": as_of}, sort_keys=True)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return Path(self.cache_dir) / "earnings" / f"{digest}.json"

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
