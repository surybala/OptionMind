"""CLI for building a prototype historical candidate dataset."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.datasets.candidate_dataset import CandidateDatasetRow
from ml.datasets.etf_universe import broad_etf_underlyings
from ml.providers import AlpacaProvider, FMPProvider, FREDProvider, MassiveProvider, YFinanceProvider
from ml.storage import ParquetDatasetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype option candidate dataset rows.")
    parser.add_argument("--provider", default="alpaca", choices=["alpaca", "massive"], help="Provider adapter to use for option data (contracts, option bars, dividends).")
    parser.add_argument(
        "--stock-provider",
        default="same",
        choices=["same", "yfinance"],
        help=(
            "Provider for underlying stock bars. "
            "'same' (default) uses the same provider as --provider. "
            "'yfinance' uses Yahoo Finance for stock bars — recommended when the main "
            "provider has a limited stock bar retention window (e.g. Massive's 2-year cap)."
        ),
    )
    parser.add_argument("--underlyings", default="SPY", help="Comma-separated underlying symbols, or 'broad-etfs'.")
    parser.add_argument(
        "--underlying-preset",
        default="custom",
        choices=["custom", "broad-etfs"],
        help="Preset universe. Use 'broad-etfs' for liquid ETF premium-selling research.",
    )
    parser.add_argument("--entry-start", required=True, help="Entry window start, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--entry-end", required=True, help="Entry window end, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--contract-status", default="inactive", choices=["active", "inactive"], help="Option contract status.")
    parser.add_argument("--strategy-family", default="credit-spreads", choices=["short-option", "credit-spreads"], help="Label family to generate.")
    parser.add_argument("--strategy-types", default="PCS,CCS", help="Comma-separated strategy types for credit-spreads.")
    parser.add_argument("--spread-widths", default="5,10,15,20", help="Comma-separated spread widths to pair for PCS/CCS rows.")
    parser.add_argument("--spread-stop-loss-max-loss-pct", type=float, default=0.80, help="For spreads, stop when close debit reaches entry credit plus this fraction of max loss. Use a negative value to disable.")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--max-contracts", type=int, default=300, help="Max locally selected contracts per underlying/window after full metadata pagination.")
    parser.add_argument("--option-limit", type=int, default=None, help="Optional provider metadata cap per underlying/window. Omit to fetch all paginated contract metadata before local filtering.")
    parser.add_argument("--max-rows-per-underlying", type=int, default=None)
    parser.add_argument("--max-abs-strike-distance-pct", type=float, default=0.30)
    parser.add_argument("--min-forward-bars", type=int, default=5)
    parser.add_argument("--min-option-entry-price", type=float, default=0.05, help="Skip entry bars whose close is at or below this value (penny/junk option filter).")
    parser.add_argument("--min-option-entry-volume", type=int, default=1, help="Skip entry bars with volume below this threshold (zero-volume stale quote filter).")
    parser.add_argument("--sample-every-n-bars", type=int, default=1)
    parser.add_argument("--stock-lookback-days", type=int, default=60)
    parser.add_argument(
        "--stock-timeframe",
        default="1Day",
        help=(
            "Underlying bar timeframe used for stock/ETF history features. "
            "Keep this at 1Day for the current feature set unless you are deliberately "
            "building an intraday experiment."
        ),
    )
    parser.add_argument(
        "--option-timeframe",
        default="1Day",
        help=(
            "Option bar timeframe used for entry-path sampling and forward labels. "
            "Use 1Min for minute-seeded candidate corpora."
        ),
    )
    parser.add_argument("--market-regime-symbol", default="SPY", help="Benchmark symbol used for market regime features.")
    parser.add_argument(
        "--vix-symbol",
        default="I:VIX",
        help=(
            "Ticker used to fetch VIX index bars. "
            "Polygon/Massive requires 'I:VIX' (default). "
            "Alpaca uses 'VIX'."
        ),
    )
    parser.add_argument("--forward-days", type=int, default=30)
    parser.add_argument("--build-window-days", type=int, default=45, help="Entry-window size for contract fetching. Smaller values fetch more targeted contracts per period for long date ranges.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help=(
            "Number of parallel threads for option contract + bar fetches "
            "(one thread per window × underlying pair). "
            "Increase to 16-32 for large builds if the provider supports it. "
            "Set to 1 to disable parallelism for debugging."
        ),
    )
    parser.add_argument("--dataset-version", default="candidate_rows_v001")
    parser.add_argument("--output-dir", default="artifacts/datasets")
    parser.add_argument("--jsonl-output", default=None, help="Optional JSONL inspection copy.")
    parser.add_argument("--append", action="store_true", help="Append rows to an existing dataset version and refresh its manifest.")
    parser.add_argument(
        "--min-output-rows",
        type=int,
        default=0,
        help="Fail the build if fewer candidate rows are written. Useful for large corpus runs.",
    )

    # Optional event / calendar providers
    parser.add_argument(
        "--economic-calendar",
        default="fred",
        choices=["fred", "fmp", "none"],
        help=(
            "Source for historical macro event dates (CPI, NFP, GDP, PPI). "
            "'fred' (default) fetches full history in one call per series (requires FRED_API_KEY). "
            "'fmp' chunks into 90-day windows (requires FMP_API_KEY). "
            "'none' disables macro events (days_to_macro_event will be null)."
        ),
    )
    parser.add_argument(
        "--event-provider",
        default="fmp",
        choices=["fmp", "yfinance", "none"],
        help=(
            "Source for earnings calendar. "
            "'fmp' requires FMP_API_KEY and covers the full date range in 90-day chunks. "
            "'yfinance' uses Yahoo Finance (free, no API key) with ~6 years of history. "
            "'none' disables earnings features (days_to_earnings will be null)."
        ),
    )
    parser.add_argument(
        "--dividend-provider",
        default="massive",
        choices=["massive", "fmp", "none"],
        help=(
            "Source for ex-dividend calendar. 'massive' reuses Massive/Polygon dividend data, "
            "'fmp' requires an FMP plan with historical dividend calendar access."
        ),
    )
    parser.add_argument(
        "--volatility-provider",
        default="market",
        choices=["fred", "market", "none"],
        help=(
            "Source for VIX regime bars. "
            "'market' fetches vix-symbol through the main market provider; "
            "'fred' maps I:VIX/VIX/VIXCLS to FRED VIXCLS observations."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()

    provider = _provider_from_args(args.provider)
    stock_provider = _stock_provider_from_args(args.stock_provider, provider)
    event_provider = _event_provider_from_args(args.event_provider)
    dividend_provider = _dividend_provider_from_args(args.dividend_provider, provider)
    economic_provider = _economic_provider_from_args(args.economic_calendar)
    volatility_provider = _volatility_provider_from_args(args.volatility_provider, economic_provider)

    underlyings = _underlyings_from_args(args)
    underlying_preset = _underlying_preset_from_args(args)

    config = CandidateDatasetConfig(
        underlyings=underlyings,
        entry_start=_parse_datetime(args.entry_start),
        entry_end=_parse_datetime(args.entry_end, end_of_day=True),
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        contract_status=args.contract_status,
        option_limit=args.option_limit,
        max_contracts_per_underlying=args.max_contracts,
        max_rows_per_underlying=args.max_rows_per_underlying,
        max_abs_strike_distance_pct=args.max_abs_strike_distance_pct,
        min_forward_bars=args.min_forward_bars,
        sample_every_n_bars=args.sample_every_n_bars,
        stock_lookback_days=args.stock_lookback_days,
        market_regime_symbol=args.market_regime_symbol.upper(),
        vix_symbol=args.vix_symbol,
        forward_days=args.forward_days,
        strategy_family=args.strategy_family.replace("-", "_"),
        strategy_types=tuple(item.strip().upper() for item in args.strategy_types.split(",") if item.strip()),
        spread_widths=tuple(float(item.strip()) for item in args.spread_widths.split(",") if item.strip()),
        spread_stop_loss_max_loss_pct=None if args.spread_stop_loss_max_loss_pct < 0 else args.spread_stop_loss_max_loss_pct,
        label_version="credit_spread_labels_v002" if args.strategy_family == "credit-spreads" else "short_option_labels_v002",
        profit_take_pct=0.75,
        min_option_entry_price=args.min_option_entry_price,
        min_option_entry_volume=args.min_option_entry_volume,
        max_workers=args.max_workers,
        build_window_days=args.build_window_days,
        stock_timeframe=args.stock_timeframe,
        option_timeframe=args.option_timeframe,
    )
    rows = HistoricalCandidateDatasetBuilder(
        stock_provider,
        provider,
        provider,
        event_provider=event_provider,
        dividend_provider=dividend_provider,
        economic_provider=economic_provider,
        volatility_provider=volatility_provider,
    ).build(config)
    result = ParquetDatasetWriter(root_dir=args.output_dir).write(
        rows,
        dataset_version=args.dataset_version,
        dataset_type="candidate_rows",
        schema_columns=list(CandidateDatasetRow.__dataclass_fields__.keys()) + ["entry_date"],
        metadata={
            "provider": args.provider,
            "underlyings": config.underlyings,
            "entry_start": config.entry_start.isoformat(),
            "entry_end": config.entry_end.isoformat(),
            "contract_status": config.contract_status,
            "min_dte": config.min_dte,
            "max_dte": config.max_dte,
            "forward_days": config.forward_days,
            "max_rows_per_underlying": config.max_rows_per_underlying,
            "max_contracts_per_underlying": config.max_contracts_per_underlying,
            "option_limit": config.option_limit,
            "max_abs_strike_distance_pct": config.max_abs_strike_distance_pct,
            "min_forward_bars": config.min_forward_bars,
            "min_option_entry_price": config.min_option_entry_price,
            "min_option_entry_volume": config.min_option_entry_volume,
            "profit_take_pct": config.profit_take_pct,
            "sample_every_n_bars": config.sample_every_n_bars,
            "stock_lookback_days": config.stock_lookback_days,
            "stock_timeframe": config.stock_timeframe,
            "option_timeframe": config.option_timeframe,
            "market_regime_symbol": config.market_regime_symbol,
            "build_window_days": config.build_window_days,
            "feature_set_version": "features_v005",
            "label_version": config.label_version,
            "strategy_family": config.strategy_family,
            "strategy_types": list(config.strategy_types),
            "spread_widths": list(config.spread_widths),
            "spread_stop_loss_max_loss_pct": config.spread_stop_loss_max_loss_pct,
            "economic_calendar": args.economic_calendar,
            "event_provider": args.event_provider,
            "dividend_provider": args.dividend_provider,
            "volatility_provider": args.volatility_provider,
            "stock_provider": args.stock_provider,
            "underlying_preset": underlying_preset,
            "min_output_rows": args.min_output_rows,
        },
        append=args.append,
    )

    if args.jsonl_output:
        _write_jsonl(rows, Path(args.jsonl_output))

    print(f"Wrote {result.row_count} row(s) to {result.root_path}")
    print(f"Manifest: {result.manifest_path}")
    if args.min_output_rows and result.row_count < args.min_output_rows:
        print(
            f"ERROR: dataset produced {result.row_count} row(s), below --min-output-rows={args.min_output_rows}",
            file=sys.stderr,
        )
        return 2
    return 0


def _write_jsonl(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), default=str) + "\n")


def _underlyings_from_args(args: argparse.Namespace) -> list[str]:
    if _underlying_preset_from_args(args) == "broad-etfs":
        return broad_etf_underlyings()
    return [item.strip().upper() for item in args.underlyings.split(",") if item.strip()]


def _underlying_preset_from_args(args: argparse.Namespace) -> str:
    if args.underlying_preset == "broad-etfs" or args.underlyings.strip().lower() in {"broad-etfs", "broad_etfs"}:
        return "broad-etfs"
    return "custom"


def _parse_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    if len(value) == 10:
        parsed_date = datetime.fromisoformat(value).replace(tzinfo=UTC)
        if end_of_day:
            return parsed_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        return parsed_date
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _provider_from_args(provider_name: str):
    if provider_name == "alpaca":
        return AlpacaProvider.from_env()
    if provider_name == "massive":
        return MassiveProvider.from_env()
    raise ValueError(f"Unsupported provider: {provider_name}")


def _stock_provider_from_args(name: str, market_provider=None):
    """Return the MarketDataProvider used exclusively for underlying stock bars.

    'same' reuses the main market_provider (e.g. Massive) — suitable when the
    provider has sufficient historical stock bar coverage.  'yfinance' swaps in
    Yahoo Finance, which covers the full date range for free and is the right
    choice when Massive returns only the most-recent 2 years of daily bars.
    """
    if name == "same":
        return market_provider
    if name == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unsupported stock provider: {name}")


def _event_provider_from_args(name: str):
    """Return an EventDataProvider or None."""
    if name == "fmp":
        return FMPProvider.from_env()
    if name == "yfinance":
        return YFinanceProvider()
    if name == "none":
        return None
    raise ValueError(f"Unsupported event provider: {name}")


def _dividend_provider_from_args(name: str, market_provider=None):
    """Return a DividendDataProvider or None."""
    if name == "massive":
        if isinstance(market_provider, MassiveProvider):
            return market_provider
        return MassiveProvider.from_env()
    if name == "fmp":
        return FMPProvider.from_env()
    if name == "none":
        return None
    raise ValueError(f"Unsupported dividend provider: {name}")


def _economic_provider_from_args(name: str):
    """Return an EconomicCalendarProvider or None.

    FRED is the recommended choice for historical builds because it returns
    the full history (back to the 1990s) in a single call per series.
    FMP is available as a fallback but requires 90-day chunked requests.
    """
    if name == "fred":
        return FREDProvider.from_env()
    if name == "fmp":
        return FMPProvider.from_env()
    if name == "none":
        return None
    raise ValueError(f"Unsupported economic calendar provider: {name}")


def _volatility_provider_from_args(name: str, economic_provider=None):
    """Return a VolatilityDataProvider or None."""
    if name == "fred":
        if isinstance(economic_provider, FREDProvider):
            return economic_provider
        return FREDProvider.from_env()
    if name in {"market", "none"}:
        return None
    raise ValueError(f"Unsupported volatility provider: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
