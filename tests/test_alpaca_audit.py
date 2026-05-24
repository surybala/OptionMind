from pathlib import Path

from ml.data_audit.alpaca_audit import (
    FAIL,
    SKIP,
    AlpacaAuditConfig,
    AuditCheck,
    AuditReport,
    render_markdown,
    run_audit,
    write_reports,
)


def test_dry_run_skips_network_checks_without_credentials():
    report = run_audit(
        AlpacaAuditConfig(api_key=None, api_secret=None, underlyings=["SPY"]),
        dry_run=True,
    )

    assert report.dry_run is True
    assert report.config["has_credentials"] is False
    assert any(check.name == "dry_run" and check.status == SKIP for check in report.checks)


def test_render_markdown_includes_check_details():
    report = AuditReport(
        provider="alpaca",
        generated_at="2026-05-24T00:00:00+00:00",
        dry_run=True,
        config={
            "underlyings": ["SPY"],
            "option_feed": "opra",
            "stock_feed": "sip",
        },
        checks=[
            AuditCheck(
                name="sample",
                status=FAIL,
                summary="Could not fetch sample data.",
                details={"symbol": "SPY"},
                error="RuntimeError: sample",
            )
        ],
    )

    markdown = render_markdown(report)

    assert "# Alpaca Data Source Audit" in markdown
    assert "`sample`" in markdown
    assert "RuntimeError: sample" in markdown
    assert '"symbol": "SPY"' in markdown


def test_render_markdown_includes_coverage_matrix_table():
    report = AuditReport(
        provider="alpaca",
        generated_at="2026-05-24T00:00:00+00:00",
        dry_run=False,
        config={
            "underlyings": ["SPY"],
            "option_feed": "opra",
            "stock_feed": "sip",
            "coverage_years": [0],
        },
        checks=[
            AuditCheck(
                name="coverage_matrix",
                status="PASS",
                summary="1/1 rows covered.",
                details={
                    "matrix_rows": [
                        {
                            "underlying": "SPY",
                            "years_back": 0,
                            "target_date": "2026-05-14",
                            "stock_daily_bars": 5,
                            "stock_opening_minute_bars": 46,
                            "option_contracts": 20,
                            "option_daily_bars": 2,
                            "option_trades": 8,
                            "current_snapshot_greeks": True,
                            "historical_greeks_observed": False,
                        }
                    ]
                },
            )
        ],
    )

    markdown = render_markdown(report)

    assert "#### Coverage Matrix" in markdown
    assert "| SPY | 0 | 2026-05-14 | 5 | 46 | 20 | 2 | 8 | True | False |" in markdown


def test_write_reports_creates_json_and_markdown(tmp_path: Path):
    report = run_audit(
        AlpacaAuditConfig(api_key=None, api_secret=None, underlyings=["SPY"]),
        dry_run=True,
    )

    json_path, md_path = write_reports(report, tmp_path)

    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"
