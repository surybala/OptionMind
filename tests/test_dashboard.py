import sqlite3
import json
from datetime import date, timedelta

import pytest

pytest.importorskip("flask")
import dashboard


def _make_dashboard_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            expiry TEXT,
            strike REAL,
            type TEXT,
            premium REAL,
            prob_expiry REAL,
            status TEXT,
            order_id TEXT,
            pnl REAL DEFAULT 0,
            legs TEXT,
            contracts INTEGER DEFAULT 1,
            close_order_id TEXT,
            close_reason TEXT,
            status_updated_at TEXT,
            pnl_source TEXT,
            pnl_verified INTEGER DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


@pytest.fixture
def dashboard_client(tmp_path, monkeypatch):
    db_path = tmp_path / "trades.db"
    conn = _make_dashboard_db(db_path)
    conn.close()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"max_capital_per_period": 50000}), encoding="utf-8")
    scan_audit_path = tmp_path / "scanner_picks.json"
    monkeypatch.setattr(dashboard, "DB_PATH", str(db_path))
    monkeypatch.setattr(dashboard, "CONFIG_PATH", str(config_path))
    monkeypatch.setattr(dashboard, "SCAN_AUDIT_PATH", str(scan_audit_path))
    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client(), db_path


def _insert_trade(db_path, **overrides):
    row = {
        "timestamp": "2026-01-10 09:30:00",
        "symbol": "SPY",
        "expiry": "2026-02-20",
        "strike": 450.0,
        "type": "PCS",
        "premium": 0.50,
        "prob_expiry": 0.80,
        "status": "CLOSED",
        "order_id": "OPEN1",
        "pnl": 25.0,
        "legs": "{}",
        "contracts": 1,
        "close_order_id": "CLOSE1",
        "close_reason": "PROFIT_TAKE",
        "status_updated_at": "2026-04-29 13:37:00",
        "pnl_source": "ALPACA_FILLS",
        "pnl_verified": 1,
    }
    row.update(overrides)
    cols = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
            tuple(row.values()),
        )
        conn.commit()


def test_stats_realized_pnl_uses_close_date_not_open_date(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(
        db_path,
        timestamp="2026-01-10 09:30:00",
        status_updated_at="2026-04-29 13:37:00",
        pnl=40.0,
    )

    res = client.get("/api/stats?start=2026-04-29&end=2026-04-29")

    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["realized_trades"] == 1
    assert data["summary"]["total_pnl"] == pytest.approx(40.0)
    assert data["timeline"] == [
        {"date": "2026-04-29", "daily_pnl": 40.0, "trades": 1, "cumulative_pnl": 40.0}
    ]


def test_stats_pnl_excludes_pending_close_estimates(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(db_path, status="CLOSED", pnl=40.0)
    _insert_trade(
        db_path,
        symbol="QQQ",
        status="PENDING_CLOSE",
        pnl=999.0,
        status_updated_at="2026-04-29 14:00:00",
    )

    res = client.get("/api/stats?start=2026-04-29&end=2026-04-29")

    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["total_trades"] == 2
    assert data["summary"]["realized_trades"] == 1
    assert data["summary"]["total_pnl"] == pytest.approx(40.0)
    assert data["by_strategy"][0]["total_pnl"] == pytest.approx(40.0)


def test_stats_premium_metrics_scale_by_contracts(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(db_path, premium=0.50, contracts=3, pnl=90.0)

    res = client.get("/api/stats?start=2026-04-29&end=2026-04-29")

    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["total_premium"] == pytest.approx(150.0)
    assert data["summary"]["avg_premium"] == pytest.approx(150.0)
    assert data["by_strategy"][0]["avg_premium"] == pytest.approx(150.0)


def test_trades_endpoint_can_sort_and_return_contracts(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(db_path, symbol="SPY", contracts=1)
    _insert_trade(db_path, symbol="QQQ", contracts=4)

    res = client.get("/api/trades?sort=contracts&dir=desc")

    assert res.status_code == 200
    trades = res.get_json()["trades"]
    assert [row["symbol"] for row in trades] == ["QQQ", "SPY"]
    assert trades[0]["contracts"] == 4


def test_scanner_picks_endpoint_returns_latest_audit(dashboard_client):
    client, _db_path = dashboard_client
    with open(dashboard.SCAN_AUDIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": "2026-05-05T09:30:00",
                "selected": [{
                    "symbol": "SPY",
                    "mispricing_score": 0.1234,
                    "ranking_reason": "score=0.123400, dte=12, otm=4.20%",
                    "large_loss_prob": 0.08,
                    "stop_loss_prob": 0.11,
                }],
                "rejected": [{
                    "symbol": "QQQ",
                    "reject_reason": "Portfolio gamma risk",
                    "ranking_reason": "score=0.100000, dte=18, otm=2.10%",
                    "stop_loss_prob": 0.19,
                }],
            },
            fh,
        )

    res = client.get("/api/scanner-picks")

    assert res.status_code == 200
    data = res.get_json()
    assert data["selected"][0]["symbol"] == "SPY"
    assert data["selected"][0]["mispricing_score"] == pytest.approx(0.1234)
    assert "otm=4.20%" in data["selected"][0]["ranking_reason"]
    assert data["selected"][0]["stop_loss_prob"] == pytest.approx(0.11)
    assert data["rejected"][0]["reject_reason"] == "Portfolio gamma risk"


def test_open_positions_unrealized_pnl_scales_by_contracts(dashboard_client, monkeypatch):
    client, db_path = dashboard_client
    expiry = (date.today() + timedelta(days=7)).isoformat()
    _insert_trade(
        db_path,
        status="EXECUTED",
        expiry=expiry,
        premium=1.00,
        contracts=3,
    )
    monkeypatch.setattr(dashboard, "_get_mark_and_spot", lambda pos: (0.25, 500.0))

    res = client.get("/api/open-positions")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["positions"][0]["contracts"] == 3
    assert payload["positions"][0]["unrealized_pnl"] == pytest.approx(225.0)
    assert payload["total_unrealized_pnl"] == pytest.approx(225.0)


def test_stats_period_filter_limits_realized_close_window(dashboard_client):
    client, db_path = dashboard_client
    today = date.today()
    _insert_trade(
        db_path,
        symbol="SPY",
        pnl=10.0,
        status_updated_at=today.isoformat() + " 10:00:00",
    )
    _insert_trade(
        db_path,
        symbol="IWM",
        pnl=20.0,
        status_updated_at=(today - timedelta(days=10)).isoformat() + " 10:00:00",
    )

    res = client.get("/api/stats?period=weekly")

    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["realized_trades"] == 1
    assert data["summary"]["total_pnl"] == pytest.approx(10.0)


def test_stats_reports_pnl_source_quality(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(db_path, symbol="SPY", pnl=40.0, pnl_source="ALPACA_FILLS", pnl_verified=1)
    _insert_trade(db_path, symbol="QQQ", pnl=10.0, pnl_source="EXTERNAL_PLACEHOLDER", pnl_verified=0)

    res = client.get("/api/stats?start=2026-04-29&end=2026-04-29")

    assert res.status_code == 200
    data = res.get_json()
    assert data["summary"]["verified_pnl"] == pytest.approx(40.0)
    assert data["summary"]["unverified_pnl"] == pytest.approx(10.0)
    sources = {row["pnl_source"]: row for row in data["by_pnl_source"]}
    assert sources["ALPACA_FILLS"]["pnl_verified"] == 1
    assert sources["EXTERNAL_PLACEHOLDER"]["pnl_verified"] == 0


def test_stats_reports_capital_deployed_remaining_and_by_strategy(dashboard_client):
    client, db_path = dashboard_client
    _insert_trade(
        db_path,
        status="EXECUTED",
        type="PCS",
        strike=150.0,
        legs='{"short_strike": 150.0, "long_strike": 145.0}',
        contracts=2,
    )
    _insert_trade(
        db_path,
        status="PENDING_CLOSE",
        type="CSP",
        strike=50.0,
        legs='{"short_strike": 50.0}',
        contracts=1,
    )
    _insert_trade(
        db_path,
        status="DRY_RUN",
        type="PCS",
        strike=200.0,
        legs='{"short_strike": 200.0, "long_strike": 190.0}',
        contracts=10,
    )

    res = client.get("/api/stats")

    assert res.status_code == 200
    capital = res.get_json()["capital"]
    assert capital["budget"] == pytest.approx(50000.0)
    assert capital["deployed"] == pytest.approx(6000.0)
    assert capital["remaining"] == pytest.approx(44000.0)
    by_strategy = {row["strategy"]: row for row in capital["by_strategy"]}
    assert by_strategy["CSP"]["capital_deployed"] == pytest.approx(5000.0)
    assert by_strategy["PCS"]["capital_deployed"] == pytest.approx(1000.0)


def test_risk_monitor_reports_portfolio_gamma_summary(dashboard_client, monkeypatch):
    client, db_path = dashboard_client
    expiry = (date.today() + timedelta(days=7)).isoformat()
    _insert_trade(
        db_path,
        status="EXECUTED",
        expiry=expiry,
        type="PCS",
        strike=480.0,
        premium=0.80,
        legs='{"short_strike": 480.0, "long_strike": 475.0}',
        contracts=2,
    )

    class FakeRiskService:
        _data = None

        @staticmethod
        def _get_position_leg_specs(pos):
            return []

        @staticmethod
        def _build_osi_symbol(*args):
            return ""

        def enrich_position(self, pos, prefetch=None):
            return {
                **pos,
                "spot": 500.0,
                "current_mark": 0.60,
                "pnl_per_share": 0.20,
                "profit_captured_pct": 25.0,
                "gamma_theta_ratio": 0.6,
                "net_short_delta": 0.08,
                "risk_score": 0.6,
                "net_delta": 0.04,
                "net_gamma": -0.001,
                "net_theta": 0.04,
                "net_vega": -3.0,
                "dte": 7,
                "has_broker_greeks": True,
                "risk_level": "SAFE",
            }

    monkeypatch.setattr(dashboard, "_get_risk_service", lambda: FakeRiskService())

    res = client.get("/api/risk-monitor")

    assert res.status_code == 200
    payload = res.get_json()
    assert payload["count"] == 1
    assert "portfolio_risk" in payload
    portfolio = payload["portfolio_risk"]
    assert "scenario_losses" in portfolio
    assert "expiry_bucket_losses" in portfolio
    assert "symbol_bucket_losses" in portfolio
    assert payload["thresholds"]["portfolio_stress_cap_pct"] == pytest.approx(0.10)
    assert payload["thresholds"]["symbol_stress_cap_pct"] == pytest.approx(0.05)
    assert payload["thresholds"]["min_stress_loss_dollars"] == pytest.approx(500.0)
    assert payload["thresholds"]["min_symbol_stress_dollars"] == pytest.approx(250.0)
    assert payload["regime"]["label"] == "GREEN"
    assert payload["regime"]["metrics"]["enabled"] is False
    assert portfolio["daily_theta"] == pytest.approx(8.0)
    assert payload["positions"][0]["net_delta"] == pytest.approx(0.04)
    assert payload["positions"][0]["net_vega"] == pytest.approx(-3.0)


def test_risk_monitor_surfaces_regime_filter_state(dashboard_client, monkeypatch):
    client, _db_path = dashboard_client
    with open(dashboard.CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "max_capital_per_period": 50000,
                "risk_parameters": {
                    "regime_filter": {
                        "enabled": True,
                        "vix": {
                            "green_below": 18,
                            "yellow_below": 25,
                            "orange_below": 32,
                        },
                    }
                },
            },
            fh,
        )

    monkeypatch.setattr(dashboard, "_get_risk_service", lambda: object())
    monkeypatch.setattr(
        dashboard,
        "_fetch_dashboard_regime_history",
        lambda symbol, period="90d": [28.0] if symbol == "^VIX" else [],
    )

    res = client.get("/api/risk-monitor")

    assert res.status_code == 200
    regime = res.get_json()["regime"]
    assert regime["label"] == "ORANGE"
    assert regime["quantity_multiplier"] == pytest.approx(0.30)
    assert regime["top_n_multiplier"] == pytest.approx(0.35)
    assert regime["pause_new_trades"] is False
    assert regime["reasons"][0].startswith("VIX 28.0")
