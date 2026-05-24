"""
OptionMind Dashboard
=====================
A local web dashboard for browsing, filtering, and analysing past trades
stored in data/trades.db.

Usage
-----
    python dashboard.py                   # opens on http://localhost:5000
    python dashboard.py --port 8080       # custom port
    python dashboard.py --db data/trades.db

Requires: flask  (pip install flask)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Optional

_log = logging.getLogger('optionwheel')

# ── Load .env (same logic as agent.py) ────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
                 override=False)
except ImportError:
    pass

import yfinance as yf

from flask import Flask, jsonify, render_template, request
from src.capital import capital_by_strategy, capital_for_position
from src.portfolio_risk import PortfolioRiskService
from src.regime import RegimeResult, RegimeService

app = Flask(__name__, template_folder="templates")
DB_PATH: str = "data/trades.db"
CONFIG_PATH: str = "config.json"
SCAN_AUDIT_PATH: str = "data/model_candidates.json"


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    cur = conn.execute(sql, params)
    return [dict(r) for r in cur.fetchall()]


def _fetch_dashboard_regime_history(symbol: str, period: str = "90d") -> list[float]:
    """Fetch adjusted closes for dashboard regime display; empty on failure."""
    try:
        hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if hist is None or getattr(hist, "empty", True):
            return []
        close = hist.get("Close")
        if close is None:
            return []
        return [float(x) for x in close.dropna().tolist()]
    except Exception:
        return []


def _dashboard_regime(config: dict) -> dict:
    svc = RegimeService(config)
    cfg = config.get("risk_parameters", {}).get("regime_filter", {})
    vix_history: list[float] = []
    trend_history: list[float] = []
    if cfg.get("enabled", False):
        trend_cfg = cfg.get("trend", {})
        trend_symbol = str(trend_cfg.get("symbol", "SPY") or "SPY")
        vix_history = _fetch_dashboard_regime_history("^VIX")
        trend_history = _fetch_dashboard_regime_history(trend_symbol)

    result: RegimeResult = svc.evaluate(
        vix_history=vix_history,
        spy_history=trend_history,
    )
    return {
        "label": result.label,
        "quantity_multiplier": result.quantity_multiplier,
        "top_n_multiplier": result.top_n_multiplier,
        "pause_new_trades": result.pause_new_trades,
        "reasons": result.reasons,
        "metrics": result.metrics,
    }


def _period_start(period: str, today: Optional[datetime] = None) -> Optional[str]:
    """Return an inclusive ISO start date for a dashboard analytics period."""
    today_d = (today or datetime.today()).date()
    key = (period or "").strip().lower()
    if key in ("", "all"):
        return None
    days = {
        "daily": 1,
        "day": 1,
        "weekly": 7,
        "week": 7,
        "quarterly": 90,
        "quarter": 90,
        "6m": 182,
        "six_months": 182,
        "yearly": 365,
        "year": 365,
    }.get(key)
    if days is None:
        return None
    return (today_d - timedelta(days=days - 1)).isoformat()


def _analytics_date_expr() -> str:
    """
    Date used for dashboard analytics.

    Closed and pending-close positions belong to the day their lifecycle state
    changed. Open rows still use entry timestamp for non-P&L status counts.
    """
    return """
        DATE(
            CASE
                WHEN UPPER(status) IN ('CLOSED', 'PENDING_CLOSE')
                THEN COALESCE(status_updated_at, timestamp)
                ELSE timestamp
            END
        )
    """


def _load_dashboard_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _capital_positions(conn: sqlite3.Connection) -> list[dict]:
    rows = _rows(
        conn,
        """
        SELECT * FROM trades
         WHERE status IN ('EXECUTED', 'PENDING_CLOSE')
         ORDER BY timestamp ASC
        """,
    )
    for row in rows:
        raw = row.get('legs')
        if raw and isinstance(raw, str):
            try:
                row['legs'] = json.loads(raw)
            except Exception:
                row['legs'] = {}
    return rows


# ── REST API ───────────────────────────────────────────────────────────────────

@app.route("/api/trades")
def api_trades():
    """
    Return filtered trade rows.

    Query params
    ------------
    symbol      str   partial match (case-insensitive)
    strategy    str   exact match on `type` column
    status      str   exact match on `status` column
    start       str   ISO date  YYYY-MM-DD  (inclusive)
    end         str   ISO date  YYYY-MM-DD  (inclusive)
    sort        str   column name to sort by (default: timestamp)
    dir         str   asc | desc (default: desc)
    limit       int   max rows to return (default: 500)
    offset      int   pagination offset (default: 0)
    """
    symbol   = request.args.get("symbol", "").strip().upper()
    strategy = request.args.get("strategy", "").strip().upper()
    status   = request.args.get("status", "").strip().upper()
    start    = request.args.get("start", "")
    end      = request.args.get("end", "")
    sort_col = request.args.get("sort", "timestamp")
    sort_dir = request.args.get("dir", "desc").upper()
    limit    = int(request.args.get("limit", 500))
    offset   = int(request.args.get("offset", 0))

    # whitelist sort column to prevent injection
    allowed_cols = {"id", "timestamp", "symbol", "expiry", "strike", "type",
                    "premium", "prob_expiry", "status", "pnl", "contracts"}
    if sort_col not in allowed_cols:
        sort_col = "timestamp"
    if sort_dir not in ("ASC", "DESC"):
        sort_dir = "DESC"

    conditions: list[str] = []
    params: list[Any] = []

    if symbol:
        conditions.append("UPPER(symbol) LIKE ?")
        params.append(f"%{symbol}%")
    if strategy:
        conditions.append("UPPER(type) = ?")
        params.append(strategy)
    if status:
        conditions.append("UPPER(status) = ?")
        params.append(status)
    if start:
        conditions.append("DATE(timestamp) >= ?")
        params.append(start)
    if end:
        conditions.append("DATE(timestamp) <= ?")
        params.append(end)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = (f"SELECT * FROM trades {where} "
           f"ORDER BY {sort_col} {sort_dir} LIMIT ? OFFSET ?")
    params += [limit, offset]

    # total count (for pagination)
    count_sql = f"SELECT COUNT(*) AS n FROM trades {where}"
    with _get_conn() as conn:
        rows  = _rows(conn, sql, tuple(params))
        total = _rows(conn, count_sql, tuple(params[:-2]))[0]["n"]

    return jsonify({"trades": rows, "total": total, "offset": offset, "limit": limit})


@app.route("/api/stats")
def api_stats():
    """
    Aggregate statistics, optionally filtered by date range / strategy / status.
    """
    strategy = request.args.get("strategy", "").strip().upper()
    status   = request.args.get("status", "").strip().upper()
    start    = request.args.get("start", "")
    end      = request.args.get("end", "")
    period   = request.args.get("period", "").strip().lower()

    if not start:
        start = _period_start(period) or ""

    conditions: list[str] = []
    params: list[Any] = []
    date_expr = _analytics_date_expr()
    if strategy:
        conditions.append("UPPER(type) = ?")
        params.append(strategy)
    if status:
        conditions.append("UPPER(status) = ?")
        params.append(status)
    if start:
        conditions.append(f"{date_expr} >= ?")
        params.append(start)
    if end:
        conditions.append(f"{date_expr} <= ?")
        params.append(end)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    realized_conditions = [*conditions, "UPPER(status) = 'CLOSED'"]
    realized_where = "WHERE " + " AND ".join(realized_conditions)

    sql_agg = f"""
        SELECT
            (SELECT COUNT(*) FROM trades {where})       AS total_trades,
            COUNT(*)                                    AS realized_trades,
            COALESCE(SUM(pnl), 0)                       AS total_pnl,
            COALESCE(AVG(premium * 100 * COALESCE(contracts, 1)), 0) AS avg_premium,
            COALESCE(SUM(premium * 100 * COALESCE(contracts, 1)), 0) AS total_premium,
            COALESCE(AVG(prob_expiry), 0)               AS avg_prob,
            COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS winning_trades,
            COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0) AS losing_trades,
            COALESCE(MAX(pnl), 0)                       AS best_trade,
            COALESCE(MIN(pnl), 0)                       AS worst_trade,
            COALESCE(AVG(CASE WHEN pnl != 0 THEN pnl END), 0) AS avg_pnl
        FROM trades {realized_where}
    """

    sql_by_strategy = f"""
        SELECT type AS strategy,
               COUNT(*) AS trades,
               COALESCE(SUM(pnl), 0) AS total_pnl,
               COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) AS wins,
               COALESCE(AVG(premium * 100 * COALESCE(contracts, 1)), 0) AS avg_premium
        FROM trades {realized_where}
        GROUP BY type
        ORDER BY trades DESC
    """

    sql_by_status = f"""
        SELECT status,
               COUNT(*) AS count
        FROM trades {where}
        GROUP BY status
    """

    sql_by_pnl_source = f"""
        SELECT COALESCE(pnl_source, 'UNKNOWN') AS pnl_source,
               COALESCE(pnl_verified, 0) AS pnl_verified,
               COUNT(*) AS trades,
               COALESCE(SUM(pnl), 0) AS total_pnl
        FROM trades {realized_where}
        GROUP BY COALESCE(pnl_source, 'UNKNOWN'), COALESCE(pnl_verified, 0)
        ORDER BY trades DESC
    """

    # Cumulative P&L by day
    sql_pnl_timeline = f"""
        SELECT DATE(COALESCE(status_updated_at, timestamp)) AS date,
               SUM(pnl)        AS daily_pnl,
               COUNT(*)        AS trades
        FROM trades {realized_where}
        GROUP BY DATE(COALESCE(status_updated_at, timestamp))
        ORDER BY date ASC
    """

    with _get_conn() as conn:
        realized_params = tuple(params)
        agg         = _rows(conn, sql_agg, tuple(params) + realized_params)[0]
        by_strategy = _rows(conn, sql_by_strategy, realized_params)
        by_status   = _rows(conn, sql_by_status, tuple(params))
        by_pnl_source = _rows(conn, sql_by_pnl_source, realized_params)
        timeline    = _rows(conn, sql_pnl_timeline, realized_params)
        cap_positions = _capital_positions(conn)

    # Build cumulative P&L series
    cumulative = 0.0
    for row in timeline:
        cumulative += row["daily_pnl"] or 0
        row["cumulative_pnl"] = round(cumulative, 2)

    win_rate = 0.0
    if agg["total_trades"] > 0:
        closed = agg["winning_trades"] + agg["losing_trades"]
        win_rate = (agg["winning_trades"] / closed * 100) if closed > 0 else 0.0

    config = _load_dashboard_config()
    capital_budget = config.get('max_capital_per_period')
    try:
        capital_budget = float(capital_budget) if capital_budget is not None else None
    except (TypeError, ValueError):
        capital_budget = None
    deployed_capital = round(sum(capital_for_position(p) for p in cap_positions), 2)
    remaining_capital = (
        round(capital_budget - deployed_capital, 2)
        if capital_budget is not None else None
    )
    verified_pnl = sum(float(r.get('total_pnl') or 0) for r in by_pnl_source if r.get('pnl_verified'))
    unverified_pnl = sum(float(r.get('total_pnl') or 0) for r in by_pnl_source if not r.get('pnl_verified'))

    return jsonify({
        "summary": {
            **agg,
            "win_rate": round(win_rate, 1),
            "verified_pnl": round(verified_pnl, 2),
            "unverified_pnl": round(unverified_pnl, 2),
        },
        "by_strategy": by_strategy,
        "by_status":   by_status,
        "by_pnl_source": by_pnl_source,
        "capital": {
            "budget": capital_budget,
            "deployed": deployed_capital,
            "remaining": remaining_capital,
            "open_positions": len(cap_positions),
            "by_strategy": capital_by_strategy(cap_positions),
        },
        "timeline":    timeline,
    })


@app.route("/api/filters")
def api_filters():
    """Return distinct values for filter dropdowns."""
    with _get_conn() as conn:
        strategies = [r["type"]   for r in _rows(conn, "SELECT DISTINCT type   FROM trades ORDER BY type")]
        statuses   = [r["status"] for r in _rows(conn, "SELECT DISTINCT status FROM trades ORDER BY status")]
        symbols    = [r["symbol"] for r in _rows(conn, "SELECT DISTINCT symbol FROM trades ORDER BY symbol")]
    return jsonify({"strategies": strategies, "statuses": statuses, "symbols": symbols})


@app.route("/api/scanner-picks")
def api_scanner_picks():
    """Return the latest ML candidate/rejection audit snapshot."""
    try:
        with open(SCAN_AUDIT_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        payload = {
            "generated_at": None,
            "score_basis": "Run the ML scanner hook after configuring a model provider to populate candidate rankings.",
            "selected": [],
            "rejected": [],
        }
    except Exception as exc:
        return jsonify({"error": str(exc), "selected": [], "rejected": []}), 500
    payload.setdefault("selected", [])
    payload.setdefault("rejected", [])
    return jsonify(payload)


@app.route("/api/symbol/<symbol>")
def api_symbol_detail(symbol: str):
    """Per-symbol breakdown: all trades, P&L history, stats."""
    with _get_conn() as conn:
        trades = _rows(
            conn,
            "SELECT * FROM trades WHERE UPPER(symbol)=? ORDER BY timestamp DESC",
            (symbol.upper(),),
        )
        stats = _rows(conn, """
            SELECT COUNT(*) AS trades,
                   SUM(pnl) AS total_pnl,
                   AVG(premium) * 100 AS avg_premium,
                   SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins
            FROM trades WHERE UPPER(symbol)=?
        """, (symbol.upper(),))[0]
    return jsonify({"symbol": symbol.upper(), "trades": trades, "stats": stats})


# ── Open-position live pricing ─────────────────────────────────────────────────

def _parse_legs_json(pos: dict) -> dict:
    """Return the legs dict from a DB row (already decoded or JSON string)."""
    raw = pos.get('legs')
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ── Shared risk service (lazy, module-level singleton) ─────────────────────────
# Instantiated on first call to _get_risk_service() so startup never blocks on
# network I/O.  Both /api/risk-monitor and any future endpoints share one instance.

_risk_service_instance = None


def _get_risk_service():
    """Return the module-level :class:`~src.risk_service.PositionRiskService` instance."""
    global _risk_service_instance
    if _risk_service_instance is None:
        from src.risk_service import PositionRiskService
        from src.utils import load_config
        _risk_service_instance = PositionRiskService.from_config(load_config(CONFIG_PATH))
    return _risk_service_instance


# _fetch_chain_maps removed — chain fetching is now handled by PositionRiskService
# (via DataAdapter), which is HFT-aware and shared with PositionMonitor.


# _compute_net_mark removed — mark computation is now handled by
# PositionRiskService.compute_mark(), shared with PositionMonitor.
# _build_greeks_legs_for_pos removed — Greek leg building is now handled by
# PositionRiskService, which is HFT-aware and shared with PositionMonitor.


def _get_mark_and_spot(
    pos: dict,
    conservative: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    """
    Fetch the cost-to-close and the underlying spot price for one open position.
    Returns (current_mark, spot_price).  Either value may be None.

    conservative=False (default) — uses mid prices for both legs; good for
        unrealized P&L display where fair-value is preferred.
    conservative=True — uses the realistic transaction prices:
        • SHORT option legs (buy-to-close) → ask price
        • LONG  option legs (sell-to-close) → bid price
        Use this when recording realized P&L on an actual close.

    Delegates to PositionRiskService so the data source (Alpaca snapshots in
    HFT mode, yfinance/Alpaca chain otherwise) is identical to the daemon.
    """
    svc = _get_risk_service()
    try:
        chain = svc._data.get_position_chain(
            pos,
            svc._get_position_leg_specs,
            svc._build_osi_symbol,
        )
    except RuntimeError as exc:
        _log.warning("[dashboard] _get_mark_and_spot: Alpaca error for %s — %s",
                     pos.get('symbol'), exc)
        return None, None
    mark = svc.compute_mark(pos, chain, conservative=conservative)
    return mark, chain.spot


@app.route("/api/open-positions")
def api_open_positions():
    """
    Return all currently open positions enriched with live mark prices and
    unrealized P&L.  Delegates to PositionRiskService (HFT-aware: Alpaca
    snapshots when hft_mode=true, yfinance/Alpaca chain otherwise).
    """
    today = datetime.today().date().isoformat()
    with _get_conn() as conn:
        rows = _rows(
            conn,
            """
            SELECT * FROM trades
             WHERE status IN ('EXECUTED', 'DRY_RUN', 'PENDING_CLOSE')
               AND expiry >= ?
             ORDER BY timestamp ASC
            """,
            (today,),
        )

    enriched: list[dict] = []
    for pos in rows:
        # Decode legs JSON
        raw = pos.get('legs')
        if raw and isinstance(raw, str):
            try:
                pos['legs'] = json.loads(raw)
            except Exception:
                pos['legs'] = {}

        premium = float(pos.get('premium', 0) or 0)

        current_mark, spot_price = _get_mark_and_spot(pos)

        unrealized_pnl: Optional[float] = None
        pnl_pct: Optional[float]        = None
        contracts = int(pos.get('contracts') or 1)
        if current_mark is not None:
            unrealized_pnl = round((premium - current_mark) * 100 * contracts, 2)
            pnl_pct = round(
                (premium - current_mark) / premium * 100, 1
            ) if premium > 0 else 0.0

        # Days to expiry
        dte: Optional[int] = None
        try:
            exp_d = datetime.fromisoformat(pos['expiry']).date()
            dte   = (exp_d - datetime.today().date()).days
        except Exception:
            pass

        legs_dict = pos.get('legs') or {}
        enriched.append({
            **pos,
            'legs':           legs_dict,
            'spot_price':     spot_price,
            'current_mark':   round(current_mark, 4) if current_mark is not None else None,
            'unrealized_pnl': unrealized_pnl,
            'pnl_pct':        pnl_pct,
            'dte':            dte,
            'market_cap':     legs_dict.get('market_cap'),
            'short_oi':       legs_dict.get('short_oi'),
            'short_volume':   legs_dict.get('short_volume'),
        })

    total_unrealized = sum(
        p['unrealized_pnl'] for p in enriched if p['unrealized_pnl'] is not None
    )

    return jsonify({
        'positions':           enriched,
        'count':               len(enriched),
        'total_unrealized_pnl': round(total_unrealized, 2),
    })


# ── Manual close endpoints ─────────────────────────────────────────────────────

def _get_open_position(trade_id: int) -> Optional[dict]:
    """Fetch a single open position by ID. Returns None if not found or not open."""
    with _get_conn() as conn:
        rows = _rows(
            conn,
            "SELECT * FROM trades WHERE id = ? AND status IN ('EXECUTED', 'DRY_RUN')",
            (trade_id,),
        )
    if not rows:
        return None
    pos = rows[0]
    # Decode legs JSON
    raw = pos.get('legs')
    if raw and isinstance(raw, str):
        try:
            pos['legs'] = json.loads(raw)
        except Exception:
            pos['legs'] = {}
    return pos


def _do_close(pos: dict, dry_run: bool) -> dict:
    """
    Price and close one open position.  Returns a result dict with keys:
      success, order_id, current_mark, pnl, error, forced_dry_run

    ``forced_dry_run`` is True when the caller requested a live close but the
    position was originally paper-traded (status='DRY_RUN') and therefore has
    no real exchange position to close against.
    """
    from src.executor import AlpacaExecutor
    from src.database import TradeDatabase
    from src.position_lifecycle import PositionLifecycleService

    # A position opened in dry-run mode was never submitted to the exchange, so
    # attempting a live close would fail with "position not found".  Force the
    # close to dry-run regardless of what the caller requested.
    forced_dry_run = False
    if pos.get('status') == 'DRY_RUN' and not dry_run:
        _log.info(
            "[dashboard] position %s (%s %s) is a dry-run entry — forcing dry-run close",
            pos.get('id'), pos.get('type'), pos.get('symbol'),
        )
        dry_run = True
        forced_dry_run = True

    # Use conservative (ask/bid) pricing for the actual close so that
    # limit_price and realized P&L reflect true transaction costs, not inflated mids.
    current_mark, _spot = _get_mark_and_spot(pos, conservative=True)

    contracts = int(pos.get('contracts') or 1)
    premium = float(pos.get('premium', 0) or 0)
    pnl = round((premium - current_mark) * 100 * contracts, 2) if current_mark is not None else 0.0

    db = TradeDatabase(DB_PATH)
    executor = AlpacaExecutor(config_path=CONFIG_PATH)
    result = PositionLifecycleService(db, executor).close_position(
        pos,
        limit_price=round(current_mark, 2) if current_mark is not None else None,
        pnl=pnl,
        dry_run=dry_run,
        reason='DASHBOARD_CLOSE',
    )
    if not result.success:
        return {
            "success": False,
            "error": result.error,
            "forced_dry_run": forced_dry_run,
            "status": result.status,
        }

    return {
        "success": True,
        "order_id": result.order_id,
        "current_mark": round(current_mark, 4) if current_mark is not None else None,
        "pnl": pnl,
        "forced_dry_run": forced_dry_run,
        "submitted": result.submitted,
        "already_pending": result.already_pending,
        "status": result.status,
    }


@app.route("/api/position-detail/<int:trade_id>")
def api_position_detail(trade_id: int):
    """
    Return per-leg quote detail (bid/ask/mid/last/IV/OI/volume) for a single
    open position by fetching the live option chain.
    """
    pos = _get_open_position(trade_id)
    if pos is None:
        return jsonify({"error": f"No open position with id={trade_id}"}), 404

    symbol = pos.get('symbol')
    expiry = pos.get('expiry')
    strat  = pos.get('type', '')
    legs   = _parse_legs_json(pos)

    # ── Fetch option chain via PositionRiskService (HFT-aware) ───────────────
    put_map: dict  = {}
    call_map: dict = {}
    svc = _get_risk_service()
    try:
        _chain = svc._data.get_position_chain(
            pos,
            svc._get_position_leg_specs,
            svc._build_osi_symbol,
        )
        if _chain.put_map:
            put_map  = {k: (v if isinstance(v, dict) else v)
                        for k, v in _chain.put_map.items()}
        if _chain.call_map:
            call_map = {k: (v if isinstance(v, dict) else v)
                        for k, v in _chain.call_map.items()}
        if not put_map and not call_map:
            return jsonify({"error": "Could not fetch option chain"}), 502
    except RuntimeError as exc:
        _log.warning("[dashboard] position-detail: Alpaca error for %s — %s", symbol, exc)
        return jsonify({"error": f"Could not fetch option chain: {exc}"}), 502

    def _leg_row(strike, side: str) -> Optional[dict]:
        m   = put_map if side == 'put' else call_map
        row = m.get(float(strike))
        if row is None:
            return None
        bid  = float(row.get('bid',              0) or 0)
        ask  = float(row.get('ask',              0) or 0)
        last = float(row.get('lastPrice',        0) or 0)
        iv   = float(row.get('impliedVolatility', 0) or 0)
        oi   = int(row.get('openInterest',       0) or 0)
        vol  = int(row.get('volume',             0) or 0)
        mid  = round((bid + ask) / 2.0, 4) if bid > 0 and ask > 0 else last
        return {
            'bid':    round(bid,  4),
            'ask':    round(ask,  4),
            'mid':    round(mid,  4),
            'last':   round(last, 4),
            'iv':     round(iv * 100, 2),   # as percentage
            'oi':     oi if oi != -1 else None,
            'volume': vol if vol > 0 else None,
        }

    # ── Build per-leg list ───────────────────────────────────────────────────
    leg_rows: list[dict] = []

    def _add(label: str, strike, side: str) -> None:
        if strike is None:
            return
        detail = _leg_row(float(strike), side)
        entry  = {'label': label, 'side': side, 'strike': float(strike)}
        if detail:
            entry.update(detail)
        leg_rows.append(entry)

    if strat == 'CSP':
        _add('Short Put',  legs.get('short_strike') or pos.get('strike'), 'put')
    elif strat == 'PCS':
        _add('Short Put',  legs.get('short_strike') or pos.get('strike'), 'put')
        _add('Long Put',   legs.get('long_strike'),                        'put')
    elif strat == 'CCS':
        _add('Short Call', legs.get('short_strike') or pos.get('strike'), 'call')
        _add('Long Call',  legs.get('long_strike'),                        'call')
    elif strat == 'IC':
        _add('Long Put',   legs.get('long_put'),   'put')
        _add('Short Put',  legs.get('short_put'),  'put')
        _add('Short Call', legs.get('short_call'), 'call')
        _add('Long Call',  legs.get('long_call'),  'call')
    elif strat == 'IFLY':
        _add('Long Put',   legs.get('long_put'),   'put')
        _add('Short Put',  legs.get('short_put'),  'put')
        _add('Long Call',  legs.get('long_call'),  'call')
    elif strat == 'CC':
        _add('Short Call', legs.get('short_strike') or pos.get('strike'), 'call')
    elif strat in ('STRANGLE',):
        _add('Short Put',  legs.get('short_put')  or pos.get('strike'), 'put')
        _add('Short Call', legs.get('short_call'),                       'call')

    return jsonify({
        'id':     trade_id,
        'symbol': symbol,
        'expiry': expiry,
        'type':   strat,
        'legs':   leg_rows,
    })


@app.route("/api/close-position/<int:trade_id>", methods=["POST"])
def api_close_position(trade_id: int):
    """
    Manually close a single open position.

    Body (JSON, optional):
      dry_run : bool   default true — set false for a live broker order
    """
    body    = request.get_json(silent=True) or {}
    dry_run = bool(body.get("dry_run", True))

    pos = _get_open_position(trade_id)
    if pos is None:
        return jsonify({"success": False, "error": f"No open position with id={trade_id}"}), 404

    result = _do_close(pos, dry_run)
    status_code = 200 if result["success"] else 500
    return jsonify(result), status_code


@app.route("/api/close-all-positions", methods=["POST"])
def api_close_all_positions():
    """
    Manually close ALL currently open positions.

    Body (JSON, optional):
      dry_run : bool   default true — set false for live broker orders
    """
    body    = request.get_json(silent=True) or {}
    dry_run = bool(body.get("dry_run", True))

    today = datetime.today().date().isoformat()
    with _get_conn() as conn:
        rows = _rows(
            conn,
            """
            SELECT * FROM trades
             WHERE status IN ('EXECUTED', 'DRY_RUN')
               AND expiry >= ?
             ORDER BY id ASC
            """,
            (today,),
        )

    # Decode legs for each row
    for pos in rows:
        raw = pos.get('legs')
        if raw and isinstance(raw, str):
            try:
                pos['legs'] = json.loads(raw)
            except Exception:
                pos['legs'] = {}

    results = []
    for pos in rows:
        result = _do_close(pos, dry_run)
        result["trade_id"] = pos["id"]
        result["symbol"]   = pos.get("symbol")
        result["type"]     = pos.get("type")
        results.append(result)

    success_count = sum(1 for r in results if r["success"])
    total_pnl     = sum(r.get("pnl", 0) for r in results if r["success"])

    return jsonify({
        "closed":        success_count,
        "total":         len(results),
        "total_pnl":     round(total_pnl, 2),
        "dry_run":       dry_run,
        "results":       results,
    })


# ── Risk monitor endpoint ──────────────────────────────────────────────────────

@app.route("/api/risk-monitor")
def api_risk_monitor():
    """
    Return per-position risk data with live Greeks.

    Delegates enrichment to :class:`~src.risk_service.PositionRiskService`,
    which is HFT-aware: uses Alpaca broker-supplied Greeks when
    ``hft_mode=true`` in config, otherwise computes them locally via
    Black-Scholes.  This guarantees the dashboard shows the same Greek values
    as the position-monitor daemon.

    Columns returned per position
    ------------------------------
    short_delta         — |net delta of short legs|; rises as position moves ITM
    gamma_theta_ratio   — |net_gamma| / |net_theta_per_day|; rising = dangerous
    risk_score          — composite score = ratio × (1 + delta_penalty)
    net_theta_per_day   — daily theta income per contract (×100); positive = earning
    profit_captured_pct — % of entry premium already secured
    sl_distance_pct     — how far (% of premium) the mark is from stop-loss trigger
    has_broker_greeks   — true when Alpaca broker Greeks were used (hft_mode=true)
    risk_level          — SAFE / WATCH / CAUTION / CRITICAL  (same as daemon)
    trigger_status      — SAFE / WATCH / WARNING / TRIGGER / STOP_LOSS / UNKNOWN
    """
    # ── Load risk thresholds from config ─────────────────────────────────────
    try:
        from src.utils import load_config
        cfg = load_config(CONFIG_PATH)
        rp  = cfg.get('risk_parameters', {})
        gr  = rp.get('gamma_risk', {})
        pgr = rp.get('portfolio_gamma_risk', {})
    except Exception:
        cfg = {}
        rp = {}; gr = {}; pgr = {}

    stop_loss_mult  = float(rp.get('stop_loss_multiplier',             1.0))
    gr_ratio_thresh = float(gr.get('gamma_theta_ratio_threshold',      1.5))
    gr_min_delta    = float(gr.get('min_delta_to_trigger',             0.15))
    gr_min_profit   = float(gr.get('min_profit_captured_pct',          0.25))
    gr_urgent_delta = float(gr.get('urgent_delta_threshold',           0.35))
    portfolio_stress_cap_pct = float(pgr.get('max_stress_loss_pct',    0.10))
    symbol_stress_cap_pct    = float(pgr.get('max_symbol_stress_pct',  0.05))
    min_stress_loss_dollars  = float(pgr.get('min_stress_loss_dollars', 500.0) or 0.0)
    min_symbol_stress_dollars = float(pgr.get('min_symbol_stress_dollars', 250.0) or 0.0)

    # ── Fetch open positions ──────────────────────────────────────────────────
    today = datetime.today().date().isoformat()
    with _get_conn() as conn:
        rows = _rows(
            conn,
            """
            SELECT * FROM trades
             WHERE status IN ('EXECUTED', 'DRY_RUN')
               AND expiry >= ?
             ORDER BY timestamp ASC
            """,
            (today,),
        )

    for pos in rows:
        raw = pos.get('legs')
        if raw and isinstance(raw, str):
            try:
                pos['legs'] = json.loads(raw)
            except Exception:
                pos['legs'] = {}

    # ── Per-position enrichment via shared PositionRiskService ────────────────
    svc    = _get_risk_service()
    result: list[dict] = []

    for pos in rows:
        symbol  = pos.get('symbol')
        expiry  = pos.get('expiry')
        premium = float(pos.get('premium', 0) or 0)

        # enrich_position handles chain fetch, mark, P&L, Greeks, risk_level
        try:
            enriched = svc.enrich_position(dict(pos))
        except RuntimeError as exc:
            # HFT mode: Alpaca unreachable
            _log.warning("[dashboard] risk-monitor: pos %s (%s) Alpaca error — %s",
                         pos.get('id'), symbol, exc)
            enriched = dict(pos)

        current_mark         = enriched.get('current_mark')
        pnl_per_share        = enriched.get('pnl_per_share')
        profit_captured_frac = (enriched.get('profit_captured_pct') or 0.0) / 100.0
        spot                 = enriched.get('spot')
        dte                  = enriched.get('dte')
        has_greeks           = bool(enriched.get('gamma_theta_ratio') is not None)

        if has_greeks:
            short_delta = enriched.get('net_short_delta', 0.0)
            ratio       = enriched.get('gamma_theta_ratio', 0.0)
        else:
            short_delta = 0.0
            ratio       = 0.0

        # ── Stop-loss distance ────────────────────────────────────────────────
        sl_trigger_mark = (1.0 + stop_loss_mult) * premium
        if current_mark is not None and premium > 0:
            sl_distance_pct = round((sl_trigger_mark - current_mark) / premium * 100, 1)
        else:
            sl_distance_pct = None

        # ── Trigger status (dashboard-specific proximity bucketing) ───────────
        loss       = -(pnl_per_share or 0.0)
        urgent     = short_delta >= gr_urgent_delta
        enough_pnl = profit_captured_frac >= gr_min_profit

        if spot is None and current_mark is None:
            trigger_status = 'UNKNOWN'
        elif premium > 0 and loss > stop_loss_mult * premium:
            trigger_status = 'STOP_LOSS'
        elif (ratio >= gr_ratio_thresh
              and short_delta >= gr_min_delta
              and (enough_pnl or urgent)):
            trigger_status = 'TRIGGER'
        elif ratio >= gr_ratio_thresh * 0.70 or short_delta >= gr_min_delta * 0.70:
            trigger_status = 'WARNING'
        elif ratio >= gr_ratio_thresh * 0.40 or short_delta >= gr_min_delta * 0.40:
            trigger_status = 'WATCH'
        else:
            trigger_status = 'SAFE'

        contracts = int(pos.get('contracts') or 1)
        unrealized_pnl = round((pnl_per_share or 0.0) * 100 * contracts, 2) if pnl_per_share is not None else None
        pnl_pct        = round(profit_captured_frac * 100, 1) if pnl_per_share is not None else None

        result.append({
            'id':                  pos['id'],
            'symbol':              symbol,
            'type':                pos.get('type'),
            'expiry':              expiry,
            'dte':                 dte,
            'premium':             round(premium, 4),
            'current_mark':        round(current_mark, 4) if current_mark is not None else None,
            'spot_price':          spot,
            'unrealized_pnl':      unrealized_pnl,
            'pnl_pct':             pnl_pct,
            'profit_captured_pct': round(profit_captured_frac * 100, 1),
            'sl_distance_pct':     sl_distance_pct,
            'short_delta':         round(short_delta, 4),
            'gamma_theta_ratio':   round(ratio, 3),
            'risk_score':          round(enriched.get('risk_score', 0.0), 3),
            'net_delta':           round(enriched.get('net_delta', 0.0), 6),
            # net_theta * 100 converts per-share/day → per-contract/day in dollars
            'net_theta_per_day':   round((enriched.get('net_theta', 0.0)) * 100, 4),
            'net_gamma':           round(enriched.get('net_gamma', 0.0), 8),
            'net_vega':            round(enriched.get('net_vega', 0.0), 6),
            'has_broker_greeks':   enriched.get('has_broker_greeks', False),
            'risk_level':          enriched.get('risk_level', 'WATCH'),
            'trigger_status':      trigger_status,
            'greeks_available':    has_greeks,
            'timestamp':           pos.get('timestamp'),
            'status':              pos.get('status'),
        })

    # ── Aggregate summary ─────────────────────────────────────────────────────
    status_counts: dict[str, int] = {
        'SAFE': 0, 'WATCH': 0, 'WARNING': 0,
        'TRIGGER': 0, 'STOP_LOSS': 0, 'UNKNOWN': 0,
    }
    for r in result:
        key = r.get('trigger_status', 'UNKNOWN')
        status_counts[key] = status_counts.get(key, 0) + 1

    ratios    = [r['gamma_theta_ratio'] for r in result if r.get('greeks_available')]
    avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 0.0
    at_risk   = status_counts['TRIGGER'] + status_counts['STOP_LOSS']
    account_capital = cfg.get('account_capital') or cfg.get('max_capital_per_period')
    try:
        account_capital = float(account_capital or 0)
    except (TypeError, ValueError):
        account_capital = 0.0
    portfolio_risk = PortfolioRiskService(cfg, position_risk_service=svc).summarize_positions(
        rows, account_capital,
    )
    regime = _dashboard_regime(cfg)

    return jsonify({
        'positions':     result,
        'count':         len(result),
        'status_counts': status_counts,
        'at_risk_count': at_risk,
        'avg_ratio':     avg_ratio,
        'thresholds': {
            'stop_loss_multiplier':        stop_loss_mult,
            'gamma_theta_ratio_threshold': gr_ratio_thresh,
            'min_delta_to_trigger':        gr_min_delta,
            'urgent_delta_threshold':      gr_urgent_delta,
            'min_profit_captured_pct':     gr_min_profit,
            'portfolio_stress_cap_pct':    portfolio_stress_cap_pct,
            'symbol_stress_cap_pct':       symbol_stress_cap_pct,
            'min_stress_loss_dollars':     min_stress_loss_dollars,
            'min_symbol_stress_dollars':   min_symbol_stress_dollars,
        },
        'portfolio_risk': portfolio_risk,
        'regime': regime,
    })


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


# ── Ensure DB exists ───────────────────────────────────────────────────────────

def _ensure_db():
    """Create the trades table if the DB doesn't exist yet."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
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
                pnl REAL DEFAULT 0
            )
        """)
        conn.commit()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OptionMind Dashboard — view and analyse your trade history"
    )
    p.add_argument("--port",   type=int, default=5000,           help="Port to listen on (default: 5000)")
    p.add_argument("--host",   default="127.0.0.1",              help="Host to bind to (default: 127.0.0.1)")
    p.add_argument("--db",     default="data/trades.db",         help="Path to the SQLite trades database")
    p.add_argument("--config", default="config.json",            help="Path to config.json (for close orders)")
    p.add_argument("--debug",  action="store_true",              help="Enable Flask debug / auto-reload")
    p.add_argument(
        "--daemon", action="store_true",
        help=(
            "Run headlessly: suppress startup print, route Flask/Werkzeug logs "
            "through the optionwheel logger, and use threaded production mode."
        ),
    )
    p.add_argument(
        "--log-file", default=None, metavar="PATH", dest="log_file",
        help="Write logs to this file with rotation (10 MB × 5 backups). Default: stdout only.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    DB_PATH     = args.db
    CONFIG_PATH = args.config
    _ensure_db()

    if args.log_file:
        from src.utils import get_logger as _get_logger
        _get_logger('optionwheel', log_file=args.log_file)

    if args.daemon:
        # Headless mode: log through the shared logger instead of printing,
        # and silence Flask/Werkzeug's default stderr handler so all output
        # goes through the configured log handler (file, etc.).
        import logging as _logging
        _logging.getLogger('werkzeug').handlers = []
        _logging.getLogger('werkzeug').propagate = True
        _log.info(
            "Dashboard starting in daemon mode on http://%s:%d",
            args.host, args.port,
        )
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
                threaded=True)
    else:
        url = f"http://{args.host}:{args.port}"
        print(f"\n  OptionMind Dashboard  →  {url}\n")
        app.run(host=args.host, port=args.port, debug=args.debug)
