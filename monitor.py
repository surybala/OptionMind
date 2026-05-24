"""
OptionWheel Position Monitor — Top-Level Daemon
================================================

Polls open positions for stop-loss triggers during market hours and sends
an EOD risk-report email each time the market closes.

Behaviour
---------
  • Only runs stop-loss checks during regular market hours (configurable,
    default Mon–Fri 09:30–16:00 US/Eastern).
  • On the first cycle after market close each trading day, sends an EOD
    risk-report email (requires email configured in config.json).
  • Polling interval is configurable via monitor_schedule.run_interval_minutes
    in config.json (default: 15 minutes).

Usage
-----
  # Daemon mode — loop at configured interval (dry-run by default):
  python monitor.py --daemon

  # Daemon mode with live close orders:
  python monitor.py --daemon --live

  # One-shot check then exit:
  python monitor.py

  # One-shot with live close orders:
  python monitor.py --live

  # Override config / database paths:
  python monitor.py --daemon --config my_config.json --db data/trades.db
"""
from __future__ import annotations

import argparse
import datetime
import os
import time
from typing import Optional

# Load .env from the project directory so ALPACA_API_KEY / ALPACA_API_SECRET
# and OPTIONWHEEL_EMAIL_PASSWORD are available without exporting them in every
# shell session.  python-dotenv is optional; falls back to existing env vars.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
                 override=False)
except ImportError:
    pass

from src.database import TradeDatabase
from src.notifier import EmailNotifier
from src.capital import capital_for_position
from src.position_monitor import PositionMonitor, _within_market_hours
from src.position_reconciler import PositionReconciler
from src.utils import get_logger, load_config

log = get_logger('optionwheel')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_eod_risk_report(
    config: dict,
    db_path: str,
    config_path: str,
    closed_today: Optional[list] = None,
) -> None:
    """
    Collect a risk snapshot for all open positions and email the daily report.

    ``closed_today`` is a list of position dicts that were stopped-out during
    the trading day (accumulated by the daemon loop).  When provided, a
    "Today's Stop-Loss Closures" section is appended to the report email.
    """
    from src.executor import AlpacaExecutor
    db       = TradeDatabase(db_path)
    db.fix_negative_premiums()
    executor = AlpacaExecutor(config_path)
    notifier = EmailNotifier(config)
    monitor  = PositionMonitor(db, executor, config)

    log.info("[monitor] Building EOD risk snapshot...")
    snapshot = monitor.get_risk_snapshot()
    n_closed = len(closed_today) if closed_today else 0
    log.info("[monitor] EOD report: %d open position(s), %d closed today.",
             len(snapshot), n_closed)

    if notifier.enabled:
        notifier.send_daily_risk_report(snapshot, closed_today=closed_today or [])
    else:
        log.warning("[monitor] EOD risk report: email not configured — skipping send.")


def _run_once(
    config: dict,
    db_path: str,
    config_path: str,
    dry_run: bool,
    accumulate_closed: Optional[list] = None,
) -> None:
    """
    Run one risk-management cycle (stop-loss / profit-take / gamma-risk checks).

    If ``accumulate_closed`` is provided (a list owned by the daemon loop),
    any positions closed this cycle are appended to it so the EOD report
    can include a closure summary for the whole day.
    """
    from src.executor import AlpacaExecutor
    db       = TradeDatabase(db_path)
    db.fix_negative_premiums()
    executor = AlpacaExecutor(config_path)
    notifier = EmailNotifier(config)
    monitor  = PositionMonitor(db, executor, config)

    hft_mode = bool(config.get('hft_mode', False))
    log.info("[monitor] Running risk check (dry_run=%s, hft_mode=%s)...", dry_run, hft_mode)

    # Settle any positions whose expiry date has passed (pure bookkeeping, no orders).
    settled = monitor.settle_expired()
    if settled:
        log.info("[monitor] Settled %d expired position(s).", len(settled))

    closed = monitor.run_hft(dry_run=dry_run) if hft_mode else monitor.run(dry_run=dry_run)
    for cp in closed:
        notifier.send_position_closed(cp, cp.get('reason_tag', 'STOP_LOSS'))
    if accumulate_closed is not None:
        accumulate_closed.extend(closed)
    log.info("[monitor] Cycle complete — %d position(s) closed.", len(closed))


def _send_weekly_digest(
    config: dict,
    db_path: str,
    config_path: str,
    week_start: datetime.date,
    week_end: datetime.date,
) -> None:
    """
    Pull the week's closed trades + current risk snapshot from the DB and
    email the weekly digest.
    """
    import sqlite3
    from src.executor import AlpacaExecutor
    db       = TradeDatabase(db_path)
    executor = AlpacaExecutor(config_path)
    notifier = EmailNotifier(config)
    monitor  = PositionMonitor(db, executor, config)

    if not notifier.enabled:
        log.warning("[monitor] Weekly digest: email not configured — skipping send.")
        return

    ws = week_start.isoformat()
    we = week_end.isoformat()

    # Closed trades this week.  Use the close/status timestamp, not the open
    # timestamp, so profit-takes on older positions appear in the digest.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT *,
               COALESCE(status_updated_at, timestamp) AS closed_at
          FROM trades
         WHERE status='CLOSED'
           AND DATE(COALESCE(status_updated_at, timestamp)) >= ?
           AND DATE(COALESCE(status_updated_at, timestamp)) <= ?
         ORDER BY COALESCE(status_updated_at, timestamp) DESC
        """,
        (ws, we),
    )
    weekly_trades = [dict(r) for r in cur.fetchall()]

    # Cumulative P&L (all time)
    cur2 = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='CLOSED'")
    cumulative_pnl = float(cur2.fetchone()[0] or 0)
    conn.close()

    # Risk snapshot for open positions
    snapshot = monitor.get_risk_snapshot()

    capital_deployed = sum(capital_for_position(p) for p in snapshot)

    log.info(
        "[monitor] Sending weekly digest: %d closed trades, cum P&L $%.2f, "
        "%d open positions, $%.0f deployed.",
        len(weekly_trades), cumulative_pnl, len(snapshot), capital_deployed,
    )
    notifier.send_weekly_digest(
        ws, we, weekly_trades, cumulative_pnl, snapshot, capital_deployed,
    )


def _run_reconcile(config: dict, db_path: str, config_path: str) -> None:
    """Run one Alpaca ↔ DB reconciliation cycle (no dry_run flag — read/write only)."""
    from src.executor import AlpacaExecutor
    db         = TradeDatabase(db_path)
    executor   = AlpacaExecutor(config_path)
    sched_cfg  = config.get('monitor_schedule', {})
    grace_min  = int(sched_cfg.get('ghost_grace_minutes', 10))
    reconciler = PositionReconciler(db, executor, ghost_grace_minutes=grace_min)
    try:
        reconciler.run()
    except Exception as exc:
        log.error("[reconciler] Cycle failed: %s", exc, exc_info=True)


def _run_daemon(config: dict, db_path: str, config_path: str, dry_run: bool) -> None:
    """
    Loop running two independent cycles:

    • Risk cycle  — stop-loss / profit-take / gamma-risk checks at the
                    configured poll interval (15 s HFT or N minutes normal).
    • Sync cycle  — Alpaca ↔ DB reconciliation at a slower interval
                    (default 5 minutes, configurable via
                    monitor_schedule.reconcile_interval_seconds).
                    Runs regardless of market hours so PENDING rows created
                    just before/after the bell are also resolved.
    """
    hft_mode   = bool(config.get('hft_mode', False))
    sched_cfg  = config.get('monitor_schedule', {})
    mkt_open   = sched_cfg.get('market_open',  '09:30')
    mkt_close  = sched_cfg.get('market_close', '16:00')
    tz_name    = sched_cfg.get('timezone',     'US/Eastern')
    weekdays   = bool(sched_cfg.get('weekdays_only', True))
    eod_report = bool(sched_cfg.get('eod_risk_report', True))
    recon_secs = int(sched_cfg.get('reconcile_interval_seconds', 300))   # default 5 min

    if hft_mode:
        hft_cfg      = config.get('hft', {})
        poll_seconds = int(hft_cfg.get('poll_interval_seconds', 15))
        interval_desc = f"{poll_seconds}s (HFT)"
    else:
        interval     = int(sched_cfg.get('run_interval_minutes', 15))
        poll_seconds = interval * 60
        interval_desc = f"{interval} min"

    log.info(
        "[monitor daemon] Starting — risk interval %s, reconcile interval %ds, "
        "market hours %s–%s %s%s, eod_report=%s, dry_run=%s",
        interval_desc, recon_secs, mkt_open, mkt_close, tz_name,
        " (weekdays)" if weekdays else "",
        eod_report, dry_run,
    )

    weekly_digest  = bool(sched_cfg.get('weekly_digest', True))

    was_in_market:       bool                    = False
    eod_sent_date:       Optional[datetime.date] = None
    weekly_digest_sent:  Optional[datetime.date] = None   # ISO week (Monday) of last send
    today_closed:        list                    = []
    last_recon_time:     float                   = 0.0   # epoch; 0 forces immediate first run

    while True:
        now       = time.time()
        in_market = _within_market_hours(mkt_open, mkt_close, tz_name, weekdays)

        # ── Risk management cycle (market hours only) ──────────────────────────
        if in_market:
            try:
                _run_once(config, db_path, config_path, dry_run,
                          accumulate_closed=today_closed)
            except Exception as exc:
                log.error("[monitor daemon] Risk cycle failed: %s", exc, exc_info=True)

        elif was_in_market and eod_report:
            today = datetime.date.today()
            if today != eod_sent_date:
                log.info(
                    "[monitor daemon] Market closed — sending EOD risk report "
                    "(%d closure(s) today)...", len(today_closed)
                )
                try:
                    _send_eod_risk_report(config, db_path, config_path,
                                          closed_today=today_closed)
                    eod_sent_date = today
                    today_closed  = []
                except Exception as exc:
                    log.error("[monitor daemon] EOD report failed: %s", exc, exc_info=True)

            # ── Weekly digest: send on Friday EOD (weekday 4) ────────────────
            if weekly_digest and today.weekday() == 4:  # 0=Mon … 4=Fri
                week_monday = today - datetime.timedelta(days=today.weekday())
                if weekly_digest_sent != week_monday:
                    log.info("[monitor daemon] Friday close — sending weekly digest...")
                    try:
                        _send_weekly_digest(config, db_path, config_path,
                                            week_start=week_monday,
                                            week_end=today)
                        weekly_digest_sent = week_monday
                    except Exception as exc:
                        log.error("[monitor daemon] Weekly digest failed: %s",
                                  exc, exc_info=True)

        else:
            log.debug("[monitor daemon] Outside market hours — skipping risk cycle.")

        # ── Reconciliation cycle (always runs, own timer) ──────────────────────
        if now - last_recon_time >= recon_secs:
            try:
                _run_reconcile(config, db_path, config_path)
            except Exception as exc:
                log.error("[monitor daemon] Reconcile cycle failed: %s", exc, exc_info=True)
            last_recon_time = time.time()

        was_in_market = in_market
        time.sleep(poll_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog='monitor.py',
        description='OptionWheel position monitor — stop-loss daemon.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--daemon', action='store_true',
                        help='Loop continuously at monitor_schedule.run_interval_minutes.')
    parser.add_argument('--live',   action='store_true',
                        help='Submit real close orders to Alpaca (default: dry-run).')
    parser.add_argument('--config', default='config.json', metavar='PATH',
                        help='Path to config.json (default: config.json).')
    parser.add_argument('--db',     default='data/trades.db', metavar='PATH',
                        help='Path to the SQLite trades database.')
    parser.add_argument('--log-file', default=None, metavar='PATH', dest='log_file',
                        help='Write logs to this file with rotation (10 MB × 5 backups). '
                             'Default: stdout only.')
    args = parser.parse_args(argv)

    if args.log_file:
        get_logger('optionwheel', log_file=args.log_file)

    config  = load_config(args.config)
    dry_run = not args.live

    if args.daemon:
        _run_daemon(config, args.db, args.config, dry_run)
    else:
        _run_once(config, args.db, args.config, dry_run)


if __name__ == '__main__':
    main()
