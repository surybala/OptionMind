"""Audit Alpaca data coverage for the OptionMind ML roadmap.

The audit is intentionally separate from the legacy deterministic scanner.
It answers whether Alpaca can supply the raw ingredients for ML training and
which gaps need complementary providers.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
WARN = "WARN"


@dataclass
class AlpacaAuditConfig:
    api_key: str | None
    api_secret: str | None
    paper: bool = True
    underlyings: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "AAPL"])
    option_feed: str = "opra"
    stock_feed: str = "sip"
    lookback_days: int = 30
    opening_window_days_back: int = 10
    limit: int = 50
    coverage_years: list[int] = field(default_factory=lambda: [0, 1, 3, 5, 7])

    @property
    def has_credentials(self) -> bool:
        return bool(self.api_key and self.api_secret)


@dataclass
class AuditCheck:
    name: str
    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AuditReport:
    provider: str
    generated_at: str
    dry_run: bool
    config: dict[str, Any]
    checks: list[AuditCheck]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(config_path: Path = Path("config.json")) -> AlpacaAuditConfig:
    _load_dotenv()
    file_config = _read_json(config_path)
    alpaca = file_config.get("alpaca", {}) if isinstance(file_config, dict) else {}

    api_key = os.getenv("ALPACA_API_KEY") or alpaca.get("api_key") or None
    api_secret = os.getenv("ALPACA_API_SECRET") or alpaca.get("api_secret") or None
    paper_raw = os.getenv("ALPACA_PAPER", alpaca.get("paper", True))

    return AlpacaAuditConfig(
        api_key=api_key,
        api_secret=api_secret,
        paper=_to_bool(paper_raw, default=True),
    )


def run_audit(
    config: AlpacaAuditConfig,
    dry_run: bool = False,
    coverage_matrix: bool = False,
) -> AuditReport:
    checks: list[AuditCheck] = [
        _check_credentials(config),
        _check_local_imports(),
    ]

    if dry_run:
        checks.append(
            AuditCheck(
                name="dry_run",
                status=SKIP,
                summary="Network checks skipped because --dry-run was used.",
            )
        )
        return _make_report(config, checks, dry_run=True)

    if not config.has_credentials:
        checks.append(
            AuditCheck(
                name="network_checks",
                status=SKIP,
                summary="Network checks skipped because Alpaca credentials were not found.",
            )
        )
        return _make_report(config, checks, dry_run=False)

    clients = _create_clients(config)
    if clients["status"] != PASS:
        checks.append(clients["check"])
        return _make_report(config, checks, dry_run=False)

    stock_client = clients["stock_client"]
    option_client = clients["option_client"]
    trading_client = clients["trading_client"]

    checks.append(_check_stock_bars(stock_client, config))
    contracts_check, sample_contract = _check_option_contracts(trading_client, config)
    checks.append(contracts_check)
    inactive_check, inactive_contract = _check_inactive_option_contracts(trading_client, config)
    checks.append(inactive_check)
    checks.append(_check_current_option_chain(option_client, config))

    contract_symbol = sample_contract or inactive_contract
    if contract_symbol:
        checks.append(_check_option_bars(option_client, config, contract_symbol))
        checks.append(_check_option_trades(option_client, config, contract_symbol))
    else:
        checks.append(
            AuditCheck(
                name="historical_option_bars",
                status=SKIP,
                summary="Skipped because no sample option contract symbol was found.",
            )
        )
        checks.append(
            AuditCheck(
                name="historical_option_trades",
                status=SKIP,
                summary="Skipped because no sample option contract symbol was found.",
            )
        )

    checks.append(_check_opening_window(stock_client, config))
    if coverage_matrix:
        checks.append(_check_coverage_matrix(stock_client, option_client, trading_client, config))
    return _make_report(config, checks, dry_run=False)


def write_reports(report: AuditReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"alpaca_audit_{stamp}.json"
    md_path = output_dir / f"alpaca_audit_{stamp}.md"

    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def render_markdown(report: AuditReport) -> str:
    lines = [
        "# Alpaca Data Source Audit",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Dry run: `{report.dry_run}`",
        f"- Underlyings: `{', '.join(report.config['underlyings'])}`",
        f"- Option feed: `{report.config['option_feed']}`",
        f"- Stock feed: `{report.config['stock_feed']}`",
        f"- Coverage years: `{', '.join(str(y) for y in report.config.get('coverage_years', []))}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Summary |",
        "|---|---:|---|",
    ]
    for check in report.checks:
        lines.append(f"| `{check.name}` | `{check.status}` | {check.summary} |")

    lines.extend(["", "## Details", ""])
    for check in report.checks:
        lines.append(f"### {check.name}")
        lines.append("")
        lines.append(f"- Status: `{check.status}`")
        lines.append(f"- Summary: {check.summary}")
        if check.error:
            lines.append(f"- Error: `{check.error}`")
        if check.details:
            if "matrix_rows" in check.details:
                lines.extend(_render_matrix_table(check.details["matrix_rows"]))
                lines.append("")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(check.details, indent=2, sort_keys=True, default=str))
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _make_report(config: AlpacaAuditConfig, checks: list[AuditCheck], dry_run: bool) -> AuditReport:
    return AuditReport(
        provider="alpaca",
        generated_at=datetime.now(UTC).isoformat(),
        dry_run=dry_run,
        config={
            "has_credentials": config.has_credentials,
            "paper": config.paper,
            "underlyings": config.underlyings,
            "option_feed": config.option_feed,
            "stock_feed": config.stock_feed,
            "lookback_days": config.lookback_days,
            "opening_window_days_back": config.opening_window_days_back,
            "limit": config.limit,
            "coverage_years": config.coverage_years,
        },
        checks=checks,
    )


def _check_credentials(config: AlpacaAuditConfig) -> AuditCheck:
    if config.has_credentials:
        return AuditCheck(
            name="credentials",
            status=PASS,
            summary="Alpaca credentials were found without exposing secret values.",
            details={"source_priority": "environment variables, then config.json"},
        )
    return AuditCheck(
        name="credentials",
        status=WARN,
        summary="Alpaca credentials were not found; real API checks will be skipped.",
        details={"required": ["ALPACA_API_KEY", "ALPACA_API_SECRET"]},
    )


def _check_local_imports() -> AuditCheck:
    try:
        from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient  # noqa: F401
        from alpaca.data.requests import OptionBarsRequest, OptionChainRequest, OptionTradesRequest, StockBarsRequest  # noqa: F401
        from alpaca.trading.client import TradingClient  # noqa: F401
        from alpaca.trading.requests import GetOptionContractsRequest  # noqa: F401
        return AuditCheck(
            name="alpaca_py_imports",
            status=PASS,
            summary="Required alpaca-py classes are importable.",
        )
    except Exception as exc:
        return AuditCheck(
            name="alpaca_py_imports",
            status=FAIL,
            summary="Required alpaca-py classes are not importable.",
            error=f"{type(exc).__name__}: {exc}",
        )


def _create_clients(config: AlpacaAuditConfig) -> dict[str, Any]:
    try:
        from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        return {
            "status": PASS,
            "stock_client": StockHistoricalDataClient(config.api_key, config.api_secret),
            "option_client": OptionHistoricalDataClient(config.api_key, config.api_secret),
            "trading_client": TradingClient(config.api_key, config.api_secret, paper=config.paper),
        }
    except Exception as exc:
        return {
            "status": FAIL,
            "check": AuditCheck(
                name="client_creation",
                status=FAIL,
                summary="Could not construct Alpaca clients.",
                error=f"{type(exc).__name__}: {exc}",
            ),
        }


def _check_stock_bars(stock_client: Any, config: AlpacaAuditConfig) -> AuditCheck:
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now(UTC) - timedelta(days=1)
        start = end - timedelta(days=config.lookback_days)
        req = StockBarsRequest(
            symbol_or_symbols=config.underlyings,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=_enum_value(DataFeed, config.stock_feed),
            limit=config.limit,
        )
        bars = stock_client.get_stock_bars(req)
        counts = _bar_counts(bars)
        return AuditCheck(
            name="historical_stock_bars",
            status=PASS if sum(counts.values()) > 0 else WARN,
            summary=f"Retrieved {sum(counts.values())} historical stock bars.",
            details={"counts_by_symbol": counts, "start": start.isoformat(), "end": end.isoformat()},
        )
    except Exception as exc:
        return _exception_check("historical_stock_bars", "Could not retrieve historical stock bars.", exc)


def _check_option_contracts(trading_client: Any, config: AlpacaAuditConfig) -> tuple[AuditCheck, str | None]:
    try:
        from alpaca.trading.requests import GetOptionContractsRequest

        today = date.today()
        req = GetOptionContractsRequest(
            underlying_symbols=config.underlyings,
            expiration_date_gte=today,
            expiration_date_lte=today + timedelta(days=45),
            limit=config.limit,
        )
        response = trading_client.get_option_contracts(req)
        contracts = _contracts_from_response(response)
        sample = _contract_symbol(contracts[0]) if contracts else None
        return (
            AuditCheck(
                name="current_option_contracts",
                status=PASS if contracts else WARN,
                summary=f"Retrieved {len(contracts)} active option contracts.",
                details={"sample_symbol": sample, "requested_underlyings": config.underlyings},
            ),
            sample,
        )
    except Exception as exc:
        return (
            _exception_check("current_option_contracts", "Could not retrieve active option contracts.", exc),
            None,
        )


def _check_inactive_option_contracts(trading_client: Any, config: AlpacaAuditConfig) -> tuple[AuditCheck, str | None]:
    try:
        from alpaca.trading.enums import AssetStatus
        from alpaca.trading.requests import GetOptionContractsRequest

        end = date.today() - timedelta(days=30)
        start = end - timedelta(days=120)
        req = GetOptionContractsRequest(
            underlying_symbols=[config.underlyings[0]],
            status=AssetStatus.INACTIVE,
            expiration_date_gte=start,
            expiration_date_lte=end,
            limit=config.limit,
        )
        response = trading_client.get_option_contracts(req)
        contracts = _contracts_from_response(response)
        sample = _contract_symbol(contracts[0]) if contracts else None
        status = PASS if contracts else WARN
        return (
            AuditCheck(
                name="inactive_option_contract_lookup",
                status=status,
                summary=f"Retrieved {len(contracts)} inactive or expired option contracts.",
                details={
                    "sample_symbol": sample,
                    "underlying": config.underlyings[0],
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            ),
            sample,
        )
    except Exception as exc:
        return (
            _exception_check("inactive_option_contract_lookup", "Could not retrieve inactive option contracts.", exc),
            None,
        )


def _check_current_option_chain(option_client: Any, config: AlpacaAuditConfig) -> AuditCheck:
    try:
        from alpaca.data.enums import OptionsFeed
        from alpaca.data.requests import OptionChainRequest

        req = OptionChainRequest(
            underlying_symbol=config.underlyings[0],
            feed=_enum_value(OptionsFeed, config.option_feed),
            expiration_date_lte=date.today() + timedelta(days=45),
        )
        chain = option_client.get_option_chain(req)
        data = _mapping_data(chain)
        sample_symbol = next(iter(data.keys()), None)
        sample_snapshot = data.get(sample_symbol) if sample_symbol else None
        has_greeks = bool(getattr(sample_snapshot, "greeks", None)) if sample_snapshot else False
        has_iv = getattr(sample_snapshot, "implied_volatility", None) is not None if sample_snapshot else False
        return AuditCheck(
            name="current_option_chain_snapshot",
            status=PASS if data else WARN,
            summary=f"Retrieved {len(data)} current chain snapshots for {config.underlyings[0]}.",
            details={"sample_symbol": sample_symbol, "sample_has_greeks": has_greeks, "sample_has_iv": has_iv},
        )
    except Exception as exc:
        return _exception_check("current_option_chain_snapshot", "Could not retrieve current option chain snapshots.", exc)


def _check_option_bars(option_client: Any, config: AlpacaAuditConfig, option_symbol: str) -> AuditCheck:
    try:
        from alpaca.data.requests import OptionBarsRequest
        from alpaca.data.timeframe import TimeFrame

        end = datetime.now(UTC) - timedelta(days=7)
        start = end - timedelta(days=config.lookback_days)
        req = OptionBarsRequest(
            symbol_or_symbols=option_symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=config.limit,
        )
        bars = option_client.get_option_bars(req)
        counts = _bar_counts(bars)
        total = sum(counts.values())
        return AuditCheck(
            name="historical_option_bars",
            status=PASS if total > 0 else WARN,
            summary=f"Retrieved {total} historical option bars for a sample contract.",
            details={"option_symbol": option_symbol, "counts_by_symbol": counts, "start": start.isoformat(), "end": end.isoformat()},
        )
    except Exception as exc:
        return _exception_check("historical_option_bars", "Could not retrieve historical option bars.", exc)


def _check_option_trades(option_client: Any, config: AlpacaAuditConfig, option_symbol: str) -> AuditCheck:
    try:
        from alpaca.data.requests import OptionTradesRequest

        end = datetime.now(UTC) - timedelta(days=7)
        start = end - timedelta(days=min(config.lookback_days, 10))
        req = OptionTradesRequest(
            symbol_or_symbols=option_symbol,
            start=start,
            end=end,
            limit=config.limit,
        )
        trades = option_client.get_option_trades(req)
        counts = _bar_counts(trades)
        total = sum(counts.values())
        return AuditCheck(
            name="historical_option_trades",
            status=PASS if total > 0 else WARN,
            summary=f"Retrieved {total} historical option trades for a sample contract.",
            details={"option_symbol": option_symbol, "counts_by_symbol": counts, "start": start.isoformat(), "end": end.isoformat()},
        )
    except Exception as exc:
        return _exception_check("historical_option_trades", "Could not retrieve historical option trades.", exc)


def _check_opening_window(stock_client: Any, config: AlpacaAuditConfig) -> AuditCheck:
    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        target = datetime.now(UTC) - timedelta(days=config.opening_window_days_back)
        start = target.replace(hour=13, minute=30, second=0, microsecond=0)
        end = start + timedelta(minutes=45)
        req = StockBarsRequest(
            symbol_or_symbols=config.underlyings,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=_enum_value(DataFeed, config.stock_feed),
            limit=1000,
        )
        bars = stock_client.get_stock_bars(req)
        counts = _bar_counts(bars)
        return AuditCheck(
            name="opening_window_stock_bars",
            status=PASS if sum(counts.values()) > 0 else WARN,
            summary=f"Retrieved {sum(counts.values())} minute bars for a market-open feature window.",
            details={"counts_by_symbol": counts, "start": start.isoformat(), "end": end.isoformat()},
        )
    except Exception as exc:
        return _exception_check("opening_window_stock_bars", "Could not retrieve market-open stock bars.", exc)


def _check_coverage_matrix(stock_client: Any, option_client: Any, trading_client: Any, config: AlpacaAuditConfig) -> AuditCheck:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for years_back in config.coverage_years:
        target_day = _market_weekday(date.today() - timedelta(days=365 * years_back + 10))
        for underlying in config.underlyings:
            row = {
                "underlying": underlying,
                "years_back": years_back,
                "target_date": target_day.isoformat(),
                "stock_daily_bars": 0,
                "stock_opening_minute_bars": 0,
                "option_contracts": 0,
                "sample_option_symbol": None,
                "option_daily_bars": 0,
                "option_trades": 0,
                "current_snapshot_greeks": None,
                "current_snapshot_iv": None,
                "historical_option_bar_fields": [],
                "historical_option_trade_fields": [],
                "historical_greeks_observed": False,
            }

            try:
                row["stock_daily_bars"] = _count_stock_bars_for_window(
                    stock_client,
                    config,
                    underlying,
                    datetime.combine(target_day - timedelta(days=7), datetime.min.time(), UTC),
                    datetime.combine(target_day + timedelta(days=1), datetime.min.time(), UTC),
                    minute=False,
                )
            except Exception as exc:
                failures.append(_matrix_failure(underlying, years_back, "stock_daily_bars", exc))

            try:
                row["stock_opening_minute_bars"] = _count_stock_bars_for_window(
                    stock_client,
                    config,
                    underlying,
                    datetime.combine(target_day, datetime.min.time(), UTC).replace(hour=13, minute=30),
                    datetime.combine(target_day, datetime.min.time(), UTC).replace(hour=14, minute=15),
                    minute=True,
                )
            except Exception as exc:
                failures.append(_matrix_failure(underlying, years_back, "stock_opening_minute_bars", exc))

            try:
                contracts = _fetch_contracts_for_matrix(trading_client, config, underlying, target_day, years_back)
                row["option_contracts"] = len(contracts)
                row["sample_option_symbol"] = _contract_symbol(contracts[0]) if contracts else None
            except Exception as exc:
                failures.append(_matrix_failure(underlying, years_back, "option_contracts", exc))

            if row["sample_option_symbol"]:
                try:
                    bar_count, bar_fields = _count_option_rows_for_window(
                        option_client,
                        row["sample_option_symbol"],
                        target_day,
                        trades=False,
                    )
                    row["option_daily_bars"] = bar_count
                    row["historical_option_bar_fields"] = bar_fields
                except Exception as exc:
                    failures.append(_matrix_failure(underlying, years_back, "option_daily_bars", exc))

                try:
                    trade_count, trade_fields = _count_option_rows_for_window(
                        option_client,
                        row["sample_option_symbol"],
                        target_day,
                        trades=True,
                    )
                    row["option_trades"] = trade_count
                    row["historical_option_trade_fields"] = trade_fields
                except Exception as exc:
                    failures.append(_matrix_failure(underlying, years_back, "option_trades", exc))

            if years_back == 0:
                try:
                    has_greeks, has_iv = _current_snapshot_has_greeks_and_iv(option_client, config, underlying)
                    row["current_snapshot_greeks"] = has_greeks
                    row["current_snapshot_iv"] = has_iv
                except Exception as exc:
                    failures.append(_matrix_failure(underlying, years_back, "current_snapshot_greeks_iv", exc))

            observed_fields = set(row["historical_option_bar_fields"]) | set(row["historical_option_trade_fields"])
            row["historical_greeks_observed"] = bool(
                observed_fields & {"delta", "gamma", "theta", "vega", "rho", "implied_volatility"}
            )
            rows.append(row)

    total_rows = len(rows)
    viable_rows = sum(
        1
        for row in rows
        if row["stock_daily_bars"] > 0
        and row["stock_opening_minute_bars"] > 0
        and row["option_contracts"] > 0
        and (row["option_daily_bars"] > 0 or row["option_trades"] > 0)
    )
    historical_greek_rows = sum(1 for row in rows if row["historical_greeks_observed"])
    status = PASS if viable_rows == total_rows and not failures else WARN
    if viable_rows == 0:
        status = FAIL

    return AuditCheck(
        name="coverage_matrix",
        status=status,
        summary=(
            f"{viable_rows}/{total_rows} symbol-year rows have stock, opening-window, contract, "
            f"and option price/trade coverage; {historical_greek_rows} rows exposed historical Greeks/IV directly."
        ),
        details={
            "matrix_rows": rows,
            "failures": failures,
            "interpretation": {
                "historical_greeks_observed": (
                    "True means Alpaca historical bar/trade responses exposed Greek or IV fields. "
                    "False usually means Greeks/IV must be computed or sourced from another provider."
                ),
                "current_snapshot_greeks": "Current option snapshots can expose Greeks/IV, but this is not the same as historical point-in-time Greeks.",
            },
        },
    )


def _exception_check(name: str, summary: str, exc: Exception) -> AuditCheck:
    return AuditCheck(
        name=name,
        status=FAIL,
        summary=summary,
        error=f"{type(exc).__name__}: {exc}",
    )


def _count_stock_bars_for_window(
    stock_client: Any,
    config: AlpacaAuditConfig,
    underlying: str,
    start: datetime,
    end: datetime,
    minute: bool,
) -> int:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    req = StockBarsRequest(
        symbol_or_symbols=underlying,
        timeframe=TimeFrame.Minute if minute else TimeFrame.Day,
        start=start,
        end=end,
        feed=_enum_value(DataFeed, config.stock_feed),
        limit=1000 if minute else config.limit,
    )
    return sum(_bar_counts(stock_client.get_stock_bars(req)).values())


def _fetch_contracts_for_matrix(
    trading_client: Any,
    config: AlpacaAuditConfig,
    underlying: str,
    target_day: date,
    years_back: int,
) -> list[Any]:
    from alpaca.trading.enums import AssetStatus
    from alpaca.trading.requests import GetOptionContractsRequest

    status = AssetStatus.ACTIVE if years_back == 0 else AssetStatus.INACTIVE
    req = GetOptionContractsRequest(
        underlying_symbols=[underlying],
        status=status,
        expiration_date_gte=target_day,
        expiration_date_lte=target_day + timedelta(days=45),
        limit=min(config.limit, 20),
    )
    return _contracts_from_response(trading_client.get_option_contracts(req))


def _count_option_rows_for_window(
    option_client: Any,
    option_symbol: str,
    target_day: date,
    trades: bool,
) -> tuple[int, list[str]]:
    from alpaca.data.requests import OptionBarsRequest, OptionTradesRequest
    from alpaca.data.timeframe import TimeFrame

    start = datetime.combine(target_day - timedelta(days=7), datetime.min.time(), UTC)
    end = datetime.combine(target_day + timedelta(days=7), datetime.min.time(), UTC)
    if trades:
        req = OptionTradesRequest(symbol_or_symbols=option_symbol, start=start, end=end, limit=100)
        result = option_client.get_option_trades(req)
    else:
        req = OptionBarsRequest(
            symbol_or_symbols=option_symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=100,
        )
        result = option_client.get_option_bars(req)

    data = _mapping_data(result)
    rows = next(iter(data.values()), [])
    return sum(_bar_counts(result).values()), _sample_field_names(rows)


def _current_snapshot_has_greeks_and_iv(option_client: Any, config: AlpacaAuditConfig, underlying: str) -> tuple[bool, bool]:
    from alpaca.data.enums import OptionsFeed
    from alpaca.data.requests import OptionChainRequest

    req = OptionChainRequest(
        underlying_symbol=underlying,
        feed=_enum_value(OptionsFeed, config.option_feed),
        expiration_date_lte=date.today() + timedelta(days=45),
    )
    data = _mapping_data(option_client.get_option_chain(req))
    sample = next(iter(data.values()), None)
    if sample is None:
        return False, False
    return bool(getattr(sample, "greeks", None)), getattr(sample, "implied_volatility", None) is not None


def _sample_field_names(rows: Any) -> list[str]:
    try:
        sample = rows[0]
    except (IndexError, KeyError, TypeError):
        return []
    raw = getattr(sample, "__dict__", {})
    return sorted(k for k in raw.keys() if not k.startswith("_"))


def _market_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _matrix_failure(underlying: str, years_back: int, check: str, exc: Exception) -> dict[str, str]:
    return {
        "underlying": underlying,
        "years_back": str(years_back),
        "check": check,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _render_matrix_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "#### Coverage Matrix",
        "",
        "| Underlying | Years Back | Target Date | Stock Daily | Open Minutes | Contracts | Option Bars | Option Trades | Current Greeks | Historical Greeks/IV |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {underlying} | {years_back} | {target_date} | {stock_daily_bars} | "
            "{stock_opening_minute_bars} | {option_contracts} | {option_daily_bars} | "
            "{option_trades} | {current_snapshot_greeks} | {historical_greeks_observed} |".format(**row)
        )
    return lines


def _enum_value(enum_cls: Any, raw: str) -> Any:
    raw_lower = raw.lower()
    for item in enum_cls:
        if item.value.lower() == raw_lower or item.name.lower() == raw_lower:
            return item
    raise ValueError(f"Unsupported {enum_cls.__name__} value: {raw}")


def _bar_counts(result: Any) -> dict[str, int]:
    data = _mapping_data(result)
    counts: dict[str, int] = {}
    for symbol, rows in data.items():
        try:
            counts[str(symbol)] = len(rows)
        except TypeError:
            counts[str(symbol)] = 1 if rows else 0
    return counts


def _mapping_data(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    data = getattr(result, "data", result)
    if hasattr(data, "items"):
        return dict(data.items())
    return {}


def _contracts_from_response(response: Any) -> list[Any]:
    if response is None:
        return []
    contracts = getattr(response, "option_contracts", None)
    if contracts is not None:
        return list(contracts)
    if isinstance(response, list):
        return response
    return []


def _contract_symbol(contract: Any) -> str | None:
    symbol = getattr(contract, "symbol", None)
    return str(symbol) if symbol else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        return


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "paper"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Alpaca data coverage for OptionMind ML training.")
    parser.add_argument("--dry-run", action="store_true", help="Validate local setup without making API calls.")
    parser.add_argument("--output-dir", default="artifacts/data_audit", help="Directory for JSON and Markdown reports.")
    parser.add_argument("--underlyings", default="SPY,QQQ,AAPL", help="Comma-separated underlyings to probe.")
    parser.add_argument("--option-feed", default="opra", choices=["opra", "indicative"], help="Option feed to request.")
    parser.add_argument("--stock-feed", default="sip", choices=["sip", "iex", "delayed_sip"], help="Stock feed to request.")
    parser.add_argument("--lookback-days", type=int, default=30, help="Historical lookback for sample checks.")
    parser.add_argument("--coverage-matrix", action="store_true", help="Probe multi-symbol, multi-year training-data coverage.")
    parser.add_argument(
        "--coverage-years",
        default="0,1,3,5,7",
        help="Comma-separated year offsets for --coverage-matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    config.underlyings = [s.strip().upper() for s in args.underlyings.split(",") if s.strip()]
    config.option_feed = args.option_feed
    config.stock_feed = args.stock_feed
    config.lookback_days = args.lookback_days
    config.coverage_years = [int(y.strip()) for y in args.coverage_years.split(",") if y.strip()]

    report = run_audit(config, dry_run=args.dry_run, coverage_matrix=args.coverage_matrix)
    json_path, md_path = write_reports(report, Path(args.output_dir))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    return 1 if any(check.status == FAIL for check in report.checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
