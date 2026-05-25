"""CLI for building a prototype historical candidate dataset."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from ml.datasets import CandidateDatasetConfig, HistoricalCandidateDatasetBuilder
from ml.datasets.candidate_dataset import CandidateDatasetRow
from ml.providers import AlpacaProvider, FMPProvider, FREDProvider, MassiveProvider
from ml.storage import ParquetDatasetWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build prototype option candidate dataset rows.")
    parser.add_argument("--provider", default="alpaca", choices=["alpaca", "massive"], help="Provider adapter to use.")
    parser.add_argument("--underlyings", default="SPY", help="Comma-separated underlying symbols.")
    parser.add_argument("--entry-start", required=True, help="Entry window start, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--entry-end", required=True, help="Entry window end, ISO datetime or YYYY-MM-DD.")
    parser.add_argument("--contract-status", default="inactive", choices=["active", "inactive"], help="Option contract status.")
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=45)
    parser.add_argument("--max-contracts", type=int, default=25)
    parser.add_argument("--option-limit", type=int, default=100, help="Provider contract metadata page limit before local filtering.")
    parser.add_argument("--max-rows-per-underlying", type=int, default=None)
    parser.add_argument("--max-abs-strike-distance-pct", type=float, default=0.30)
    parser.add_argument("--min-forward-bars", type=int, default=2)
    parser.add_argument("--sample-every-n-bars", type=int, default=1)
    parser.add_argument("--stock-lookback-days", type=int, default=60)
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
        choices=["fmp", "none"],
        help="Source for earnings calendar (requires FMP_API_KEY when set to 'fmp').",
    )
    parser.add_argument(
        "--dividend-provider",
        default="fmp",
        choices=["fmp", "none"],
        help="Source for ex-dividend calendar (requires FMP_API_KEY when set to 'fmp').",
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
    event_provider = _event_provider_from_args(args.event_provider)
    dividend_provider = _dividend_provider_from_args(args.dividend_provider)
    economic_provider = _economic_provider_from_args(args.economic_calendar)
    volatility_provider = _volatility_provider_from_args(args.volatility_provider, economic_provider)

    config = CandidateDatasetConfig(
        underlyings=[item.strip().upper() for item in args.underlyings.split(",") if item.strip()],
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
        max_workers=args.max_workers,
        build_window_days=args.build_window_days,
    )
    rows = HistoricalCandidateDatasetBuilder(
        provider,
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
            "max_abs_strike_distance_pct": config.max_abs_strike_distance_pct,
            "min_forward_bars": config.min_forward_bars,
            "sample_every_n_bars": config.sample_every_n_bars,
            "stock_lookback_days": config.stock_lookback_days,
            "market_regime_symbol": config.market_regime_symbol,
            "build_window_days": config.build_window_days,
            "feature_set_version": "features_v002",
            "label_version": config.label_version,
            "economic_calendar": args.economic_calendar,
            "event_provider": args.event_provider,
            "dividend_provider": args.dividend_provider,
            "volatility_provider": args.volatility_provider,
        },
    )

    if args.jsonl_output:
        _write_jsonl(rows, Path(args.jsonl_output))

    print(f"Wrote {result.row_count} row(s) to {result.root_path}")
    print(f"Manifest: {result.manifest_path}")
    return 0


def _write_jsonl(rows, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row), default=str) + "\n")


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


def _event_provider_from_args(name: str):
    """Return an EventDataProvider or None."""
    if name == "fmp":
        return FMPProvider.from_env()
    if name == "none":
        return None
    raise ValueError(f"Unsupported event provider: {name}")


def _dividend_provider_from_args(name: str):
    """Return a DividendDataProvider or None."""
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
