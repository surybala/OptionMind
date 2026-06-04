"""
OptionMind Agent — Main Entry Point
===================================

Runs the ML scanner hook, applies deterministic risk controls, presents a
ranked plan of model-suggested trades, and optionally executes approved picks
via Alpaca.

Execution modes
---------------
  approve   (default)  Show the plan table, ask for user approval, then
                        execute only the approved subset.
  auto                 Execute all picks that survive the ML pipeline
                        (ranker scoring + classifier veto + selection
                        controls).  No additional prob_win gate — the
                        model is the gate.
                        (useful for scheduled / headless runs).
  scan-only            Score and print the plan but never execute anything.
                        Picks are also saved to data/pending_picks.json.

Trade safety
------------
  --dry-run  (default) Simulate every order — nothing reaches Alpaca.
  --live               Submit real orders to Alpaca.  Requires valid
                       api_key / api_secret in config.json.

Usage examples
--------------
  # Interactive approve mode (dry-run by default):
  python agent.py

  # Interactive approve mode, submit real orders when approved:
  python agent.py --live

  # Auto-execute high-confidence picks (live):
  python agent.py --mode auto --live

  # Score only — no execution, no prompts:
  python agent.py --mode scan-only

  # Use full index universe instead of the default 10-ticker sample:
  python agent.py --universe index

  # Use the full exchange-wide ETF universe instead of the curated live ETF list:
  python agent.py --universe etf-all

  # Override max capital per period:
  python agent.py --max-capital 30000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor
from src.notifier import EmailNotifier
from src.capital import capital_for_position as _capital_for_position
from src.position_monitor import PositionMonitor
from src.utils import get_logger, load_config

# ── Extracted modules ────────────────────────────────────────────────────────
from src.agent_risk import (
    capital_for_pick,
    max_loss_multiple as max_loss_multiple_for_pick,
    filter_max_loss_multiple,
    apply_directional_exposure_caps,
    apply_portfolio_gamma_risk,
    apply_regime_quantity_multiplier,
)
from src.agent_audit import (
    mispricing_score as mispricing_score_for_pick,
    annotate_mispricing_scores,
    capture_rejections,
    record_model_predictions,
    write_scan_audit,
)
from src.agent_display import (
    print_open_positions,
    print_plan,
)
from src.agent_execution import (
    approval_gate,
    confirm_execution,
    execute_picks,
    run_close_command,
)
from src.agent_market import (
    fetch_vix,
    evaluate_regime_filter,
    reconcile_positions_before_budget,
)

log = get_logger()


# ── Backward-compatible aliases ──────────────────────────────────────────────
# Tests and other call-sites may still import these underscore-prefixed names
# from ``agent``.  Provide thin aliases so nothing breaks.
from src.agent_risk import (                        # noqa: F811
    capital_for_pick as _capital_for_pick,
    max_loss_per_contract as _max_loss_per_contract_for_pick,
    max_loss_multiple as _max_loss_multiple_for_pick,
    directional_exposure as _directional_exposure,
    filter_max_loss_multiple as _filter_max_loss_multiple,
    apply_directional_exposure_caps as _apply_directional_exposure_caps,
)
from src.agent_market import (                      # noqa: F811
    reconcile_positions_before_budget as _reconcile_positions_before_budget,
)


# ── Universe helpers ─────────────────────────────────────────────────────────

def _count_enabled_strategies(config: dict) -> int:
    """Return the number of enabled strategies in config (minimum 1)."""
    strategies = config.get('strategies', {})
    count = sum(1 for v in strategies.values() if v.get('enabled', False))
    return max(count, 1)


def _load_tickers(args, config: dict) -> list[str]:
    """Return the ticker list based on --universe / --tickers CLI args."""
    if args.tickers:
        return args.tickers

    # CLI flag > config["universe"] > "etf" (hard default)
    universe = args.universe or config.get('universe', 'etf')

    if universe == 'etf':
        from src.universe import get_stable_etf_universe
        print("Universe: loading curated stable ETF preset for options selling ...")
        tickers = get_stable_etf_universe(log=log)
        print(f"Universe: {len(tickers)} stable ETF tickers ready")
        return tickers

    if universe == 'etf-all':
        from src.universe import get_etf_universe
        print("Universe: loading all NASDAQ / NYSE / NYSE Arca ETFs ...")
        tickers = get_etf_universe(force_refresh=args.refresh_universe, log=log)
        print(f"Universe: {len(tickers)} all-ETF tickers ready")
        return tickers

    if universe == 'index':
        from src.universe_indices import get_index_tickers
        print("Universe: loading S&P 500 + NASDAQ-100 + Dow 30 constituents...")
        tickers = get_index_tickers(force_refresh=args.refresh_universe)
        print(f"Universe: {len(tickers)} index tickers ready")
        return tickers

    # default / full: fall through to get_ticker_universe (NASDAQ/NYSE by market cap)
    from src.universe import get_ticker_universe
    min_cap = config.get('market_cap_min', 1_000_000_000)
    print(f"Universe: loading NASDAQ/NYSE tickers with market cap >= ${min_cap:,.0f} ...")
    tickers = get_ticker_universe(min_cap, log=log)
    print(f"Universe: {len(tickers)} tickers loaded")
    return tickers


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='agent.py',
        description='OptionMind Agent — score model candidates, approve, and execute option trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--mode',
        choices=['approve', 'auto', 'scan-only'],
        default=None,        # None = read from config
        metavar='MODE',
        help=(
            'Execution mode: '
            '"approve" (default) — show plan, ask for approval, execute approved picks; '
            '"auto" — execute all picks that survive the ML pipeline without prompting; '
            '"scan-only" — print model-ranked plan and save to data/pending_picks.json, never execute.'
        ),
    )
    parser.add_argument(
        '--dry-run', action='store_true', dest='dry_run',
        help='Simulate orders without submitting to Alpaca (default when --live is absent).',
    )
    parser.add_argument(
        '--live', action='store_true', dest='live',
        help='Submit REAL orders to Alpaca.  Requires valid credentials in config.json.',
    )
    parser.add_argument(
        '--universe',
        choices=['etf', 'etf-all', 'index', 'default', 'full'],
        default=None,           # None = read from config["universe"], fallback "etf"
        metavar='SRC',
        help=(
            '"etf" (default) — curated stable ETF preset for live options selling; '
            '"etf-all" — all ETFs on NASDAQ / NYSE / NYSE Arca; '
            '"index" — S&P 500 + NASDAQ-100 + Dow 30 (~540 tickers); '
            '"default" / "full" — NASDAQ/NYSE stocks by market cap.'
        ),
    )
    parser.add_argument(
        '--refresh-universe', action='store_true', dest='refresh_universe',
        help='Force-refresh the cached index constituent list (only with --universe index).',
    )
    parser.add_argument(
        '--tickers', nargs='+', metavar='SYM', default=None,
        help='Explicit ticker list (overrides --universe).',
    )
    parser.add_argument(
        '--top-n', type=int, default=None, metavar='N', dest='top_n',
        help=(
            'Total number of model candidates to surface '
            '(overrides config ml_scanner.top_n).'
        ),
    )
    parser.add_argument(
        '--max-capital', type=float, default=None, metavar='DOLLARS', dest='max_capital',
        help='Cap total capital per period in dollars (overrides config max_capital_per_period).',
    )
    parser.add_argument(
        '--config', default='config.json', metavar='PATH',
        help='Path to config.json (default: config.json).',
    )
    parser.add_argument(
        '--db', default='data/trades.db', metavar='PATH', dest='db_path',
        help='Path to the SQLite trades database (default: data/trades.db).',
    )
    # ── Manual close options (short-circuit the normal candidate flow) ────────
    close_group = parser.add_argument_group(
        'manual close',
        'Close open positions immediately without requesting model candidates.',
    )
    close_group.add_argument(
        '--close', nargs='+', type=int, metavar='ID', dest='close_ids',
        help='Close one or more open positions by trade ID (e.g. --close 7 12 15).',
    )
    close_group.add_argument(
        '--close-all', action='store_true', dest='close_all',
        help='Close ALL currently open positions.',
    )
    close_group.add_argument(
        '--list-open', action='store_true', dest='list_open',
        help='Print all open positions and exit (no scan, no close).',
    )
    parser.add_argument(
        '--daemon', action='store_true',
        help=(
            'Run continuously, waking at the configured schedule.run_time each day '
            '(weekdays only by default). Fully headless — approval via email only; '
            'if email is not configured or times out, the cycle is skipped.'
        ),
    )
    parser.add_argument(
        '--log-file', default=None, metavar='PATH', dest='log_file',
        help='Write logs to this file with rotation (10 MB × 5 backups). '
             'Default: stdout only.',
    )
    return parser.parse_args()


# ── Daemon scheduler ─────────────────────────────────────────────────────────

def _timezone(tz_name: str):
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(tz_name)
    except Exception:
        try:
            import pytz
            return pytz.timezone(tz_name)
        except Exception:
            return None


def _now_in_tz(tz, now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(tz)
    if tz is None:
        return now.replace(tzinfo=None) if now.tzinfo else now
    if now.tzinfo is None:
        return now.replace(tzinfo=tz)
    return now.astimezone(tz)


def _parse_hhmm(value: str) -> tuple[int, int]:
    hh, mm = (int(p) for p in value.split(':'))
    return hh, mm


def _within_trading_window(
    open_t: str,
    close_t: str,
    tz_name: str,
    weekdays_only: bool,
    *,
    now: datetime | None = None,
) -> bool:
    """Return True when *now* is inside the configured trading window."""
    tz = _timezone(tz_name)
    current = _now_in_tz(tz, now)
    if weekdays_only and current.weekday() >= 5:
        return False

    hh_o, mm_o = _parse_hhmm(open_t)
    hh_c, mm_c = _parse_hhmm(close_t)
    window_open = current.replace(hour=hh_o, minute=mm_o, second=0, microsecond=0)
    window_close = current.replace(hour=hh_c, minute=mm_c, second=0, microsecond=0)

    if window_close <= window_open:
        if current < window_close:
            window_open -= timedelta(days=1)
        else:
            window_close += timedelta(days=1)

    return window_open <= current < window_close


def _next_run_dt(
    run_time: str,
    tz_name: str,
    weekdays_only: bool,
    *,
    now: datetime | None = None,
) -> datetime:
    """
    Return the next wall-clock datetime at which the agent should fire.

    ``run_time`` is a 24-h "HH:MM" string.  The result is always in the
    future (at least 1 second from now).  Tries zoneinfo (stdlib, Python 3.9+)
    then pytz, then falls back to local system time.
    """
    tz = _timezone(tz_name)
    hh, mm = _parse_hhmm(run_time)
    now = _now_in_tz(tz, now)

    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)

    if weekdays_only:
        # 0=Monday … 4=Friday  5=Saturday  6=Sunday
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)

    return candidate


def _next_daemon_run_dt(
    run_time: str,
    tz_name: str,
    weekdays_only: bool,
    market_open: str,
    market_close: str,
    market_tz_name: str,
    market_weekdays_only: bool,
    *,
    immediate_if_in_trading_window: bool,
    now: datetime | None = None,
) -> datetime:
    """Return the next daemon wake time, with optional startup catch-up."""
    schedule_tz = _timezone(tz_name)
    schedule_now = _now_in_tz(schedule_tz, now)
    if immediate_if_in_trading_window and _within_trading_window(
        market_open,
        market_close,
        market_tz_name,
        market_weekdays_only,
        now=schedule_now,
    ):
        return schedule_now

    return _next_run_dt(run_time, tz_name, weekdays_only, now=schedule_now)


def run_daemon(args) -> None:
    """
    Loop forever, waking at the configured schedule time each day to call
    _run_once().  Fully headless — if email is not configured or the
    approval reply times out, the cycle is skipped rather than blocking.
    """
    config     = load_config(args.config)
    sched_cfg  = config.get('schedule', {})
    run_time   = sched_cfg.get('run_time', '09:35')
    tz_name    = sched_cfg.get('timezone', 'US/Eastern')
    weekdays   = bool(sched_cfg.get('weekdays_only', True))
    market_cfg = config.get('monitor_schedule', {})
    market_open = market_cfg.get('market_open', '09:30')
    market_close = market_cfg.get('market_close', '16:00')
    market_tz_name = market_cfg.get('timezone', tz_name)
    market_weekdays = bool(market_cfg.get('weekdays_only', weekdays))

    log = get_logger('optionwheel')
    log.info(
        "[daemon] Starting — will fire at %s %s%s",
        run_time, tz_name,
        " (weekdays only)" if weekdays else "",
    )

    startup = True
    while True:
        next_dt = _next_daemon_run_dt(
            run_time,
            tz_name,
            weekdays,
            market_open,
            market_close,
            market_tz_name,
            market_weekdays,
            immediate_if_in_trading_window=startup,
        )
        startup = False
        sleep_s = (next_dt - datetime.now(next_dt.tzinfo)).total_seconds()
        log.info("[daemon] Next run scheduled for %s (in %.0f s / %.1f h)",
                 next_dt.strftime('%Y-%m-%d %H:%M %Z'), sleep_s, sleep_s / 3600)
        time.sleep(max(sleep_s, 0))

        log.info("[daemon] Waking up — starting agent cycle")
        try:
            _run_once(args, headless=True)
        except SystemExit:
            pass   # _run_close_command calls sys.exit; ignore in daemon
        except Exception as exc:
            log.error("[daemon] Cycle failed: %s", exc, exc_info=True)

        # Brief pause so we don't re-trigger within the same minute
        time.sleep(90)


# ── Scanner factory ──────────────────────────────────────────────────────────

def _build_scanner(config: dict):
    """Return the ML scanner instance (LivePaperInferenceProvider or ModelScanner fallback).

    Loads the champion model from the registry and scores short-option candidates.
    Honors a configured ml_scanner.provider only for legacy/backward-compatible
    call sites; the live agent runtime validator rejects that override.
    Automatically injects pick_selection.mode=model_ranked unless already set.
    """
    # Default pick_selection to model_ranked unless overridden
    ml_config = dict(config)
    ps = ml_config.get('pick_selection')
    if not isinstance(ps, dict) or 'mode' not in ps:
        ml_config['pick_selection'] = dict(ps or {})
        ml_config['pick_selection'].setdefault('mode', 'model_ranked')

    ml_cfg = ml_config.get('ml_scanner', {})
    explicit_provider = ml_cfg.get('provider')

    if explicit_provider:
        from src.model_scanner import ModelScanner
        log.info("[agent] Scanner: ML (ModelScanner, provider=%s)", explicit_provider)
        return ModelScanner(ml_config)

    from src.model_scanner import LivePaperInferenceProvider
    try:
        scanner = LivePaperInferenceProvider(ml_config)
        log.info(
            "[agent] Scanner: ML (LivePaperInferenceProvider, registry=%s, mode=model_ranked)",
            ml_cfg.get('registry_path', 'artifacts/model_registry.json'),
        )
        return scanner
    except Exception as exc:
        log.warning(
            "[agent] LivePaperInferenceProvider init failed (%s: %s) — "
            "no ML picks will be generated this run.",
            type(exc).__name__, exc,
        )
        from src.model_scanner import ModelScanner
        return ModelScanner({'ml_scanner': {'enabled': False}})


def _model_artifact_errors(label: str, artifact_path: Path, artifact: dict | None = None) -> list[str]:
    errors: list[str] = []
    if artifact is None:
        try:
            artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
        except Exception as exc:
            return [f"{label} artifact could not be read: `{artifact_path}` ({exc})."]

    model_path = str(artifact.get('model_path') or '').strip()
    model_type = str(artifact.get('model_type') or '').strip().lower()
    if model_type.startswith('xgboost') and not model_path:
        errors.append(f"{label} XGBoost artifact is missing `model_path`: `{artifact_path}`.")
    if model_path and not Path(model_path).exists():
        errors.append(f"{label} model file not found: `{model_path}`.")
    return errors


def _validate_ml_hft_runtime(config: dict) -> None:
    """
    Refuse to run the live candidate pipeline if it drifts away from the
    intended ML-only + Alpaca-backed HFT configuration.
    """
    ml_cfg = config.get('ml_scanner', {}) if isinstance(config.get('ml_scanner'), dict) else {}
    pick_cfg = config.get('pick_selection', {}) if isinstance(config.get('pick_selection'), dict) else {}

    errors: list[str] = []

    if not ml_cfg.get('enabled', False):
        errors.append("`ml_scanner.enabled` must be true.")

    if str(pick_cfg.get('mode', 'model_ranked')) != 'model_ranked':
        errors.append("`pick_selection.mode` must be `model_ranked` so trade selection stays model-ranked.")

    explicit_provider = str(ml_cfg.get('provider') or '').strip()
    if explicit_provider:
        errors.append(
            "`ml_scanner.provider` must be empty at runtime; live picks must come from "
            "`LivePaperInferenceProvider`, not a custom provider override."
        )

    explicit_data_provider = str(ml_cfg.get('data_provider') or '').strip()
    allowed_hft_data_providers = {
        '',
        'ml.providers:AlpacaProvider',
        'ml.providers.alpaca:AlpacaProvider',
    }
    if explicit_data_provider not in allowed_hft_data_providers:
        errors.append(
            "`ml_scanner.data_provider` must be empty or point to `AlpacaProvider` in HFT mode."
        )

    if not bool(config.get('hft_mode', False)):
        errors.append("`hft_mode` must be true.")

    registry_path = Path(str(ml_cfg.get('registry_path') or 'artifacts/model_registry.json'))
    explicit_artifact_path = str(ml_cfg.get('artifact_path') or '').strip()
    if ml_cfg.get('strategy_rankers'):
        errors.append(
            "`ml_scanner.strategy_rankers` must not be configured; PCS and CCS "
            "must both use the generic champion ranker."
        )
    if not registry_path.exists() and not explicit_artifact_path:
        errors.append(
            f"Champion model registry not found: `{registry_path}` and no explicit champion artifact was configured."
        )
    elif registry_path.exists() and not explicit_artifact_path:
        try:
            from ml.models.registry import load_champion_artifact

            entry, artifact = load_champion_artifact(registry_path)
            errors.extend(
                _model_artifact_errors(
                    "Champion model",
                    Path(entry.artifact_manifest.artifact_path),
                    artifact,
                )
            )
            model_path = entry.artifact_manifest.model_path
            if model_path and not Path(model_path).exists():
                errors.append(f"Champion model file not found: `{model_path}`.")
        except Exception as exc:
            errors.append(f"Champion model registry could not load a champion: `{registry_path}` ({exc}).")
    if explicit_artifact_path and not Path(explicit_artifact_path).exists():
        errors.append(f"Champion model artifact not found: `{explicit_artifact_path}`.")
    elif explicit_artifact_path:
        errors.extend(_model_artifact_errors("Champion model", Path(explicit_artifact_path)))

    large_loss_path = str(ml_cfg.get('large_loss_classifier_path') or '').strip()
    if not large_loss_path:
        errors.append("`ml_scanner.large_loss_classifier_path` must be set.")
    elif not Path(large_loss_path).exists():
        errors.append(f"Large-loss classifier artifact not found: `{large_loss_path}`.")
    else:
        errors.extend(_model_artifact_errors("Large-loss classifier", Path(large_loss_path)))

    from src.alpaca_data import make_alpaca_data_client
    if make_alpaca_data_client(config) is None:
        errors.append(
            "Alpaca credentials are required for HFT mode; set `ALPACA_API_KEY` / "
            "`ALPACA_API_SECRET` or populate `config.json`."
        )

    if errors:
        detail = "\n - ".join(errors)
        raise ValueError(
            "Runtime configuration is not ML-only / HFT-strict:\n - " + detail
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def _run_once(args, headless: bool = False) -> None:
    """Single agent cycle: monitor → score candidates → approve → execute."""
    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(args.config)

    # ── Resolve execution mode ────────────────────────────────────────────────
    # CLI --mode > config approve_mode > default 'approve'
    if args.mode is not None:
        mode = args.mode
    else:
        raw = config.get('alpaca', {}).get('approve_mode', True)
        if raw is True:
            mode = 'approve'
        elif raw is False or raw == 'auto':
            mode = 'auto'
        else:
            mode = str(raw)   # 'scan-only' etc. passed as string in config

    # ── Resolve dry-run ───────────────────────────────────────────────────────
    # --live wins over --dry-run; default is dry_run=True (safe)
    if args.live:
        dry_run = False
    else:
        dry_run = True   # --dry-run or neither flag → always safe default

    # ── Resolve top-n and capital cap ─────────────────────────────────────────
    if args.top_n:
        top_n = args.top_n  # CLI value is treated as the raw total
    else:
        top_n = config.get('ml_scanner', {}).get(
            'top_n',
            config.get('top_n_picks', config.get('top_n_per_strategy', 10)),
        )
    capital_budget = args.max_capital or config.get('max_capital_per_period')

    # ── Bootstrap components ──────────────────────────────────────────────────
    db = TradeDatabase(args.db_path)

    # Self-heal any open positions whose premium was stored as a negative value
    _fixed = db.fix_negative_premiums()
    if _fixed:
        log.warning("[agent] Corrected %d open position(s) with negative entry "
                    "premium (Alpaca MLEG sign convention).", _fixed)

    executor = AlpacaExecutor(args.config)

    # ── Settle expired positions first (pure bookkeeping, no orders) ─────────
    monitor  = PositionMonitor(db, executor, config)
    notifier = EmailNotifier(config)
    log.info("Email notifier %s.", "enabled" if notifier.enabled else "disabled")

    # ── Short-circuit for manual close / list-open commands ──────────────────
    if args.close_ids or args.close_all or args.list_open:
        run_close_command(args, db, executor, monitor, dry_run)
        # run_close_command always calls sys.exit(); this line is not reached.
        return

    _validate_ml_hft_runtime(config)
    scanner = _build_scanner(config)

    # ── Fetch current VIX for account-level regime controls ───────────────────
    current_vix: Optional[float] = None
    vix_cfg = config.get('risk_parameters', {}).get('vix_filter', {})
    if vix_cfg.get('enabled', False):
        _vix = fetch_vix(config)
        if _vix is not None:
            current_vix = _vix
            log.info("VIX=%0.1f captured for regime/risk controls.", current_vix)
        else:
            log.warning("Could not fetch VIX level from any source — VIX-based regime controls will not apply.")

    monitor.settle_expired()

    # A prior close can have been submitted/filled by the monitor daemon, the
    # dashboard, or directly in Alpaca.  Reconcile stale state before the risk
    # pass and before using DB rows for dedup and budget.
    reconcile_positions_before_budget(db, executor, config)

    # ── Stop-loss monitor: check open positions before considering new ones ───
    closed_positions = monitor.run(dry_run=dry_run)
    for _cp in closed_positions:
        notifier.send_position_closed(_cp, _cp.get('reason_tag', 'STOP_LOSS'))

    # ── Snapshot open positions for dedup + capital accounting ───────────────
    open_positions = db.get_open_positions()
    pending_close_positions = db.get_pending_close_positions()
    capital_positions = open_positions + pending_close_positions

    # Keys used to deduplicate new picks: same symbol + strategy already held
    open_keys: set[tuple[str, str]] = {
        (p['symbol'], p['type']) for p in capital_positions
    }

    # Capital already committed to open positions — subtracted from budget
    deployed_capital = sum(_capital_for_position(p) for p in capital_positions)

    if capital_positions:
        log.info(
            f"{len(open_positions)} open position(s), {len(pending_close_positions)} pending close(s) found — "
            f"${deployed_capital:,.0f} capital already deployed."
        )

    regime = evaluate_regime_filter(config, current_vix=current_vix)
    if hasattr(scanner, 'set_runtime_regime'):
        scanner.set_runtime_regime(regime)
    if regime.pause_new_trades:
        log.warning(
            "Regime filter is RED — skipping new trades this run. Existing "
            "positions were still monitored and reconciled."
        )
        return

    scan_top_n = top_n
    if regime.top_n_multiplier < 1.0:
        scan_top_n = max(1, int(top_n * regime.top_n_multiplier))
        log.info(
            "Regime filter reduced candidate top-N from %d to %d.",
            top_n, scan_top_n,
        )

    # ── Load ticker universe ──────────────────────────────────────────────────
    tickers = _load_tickers(args, config)
    if not tickers:
        log.error("No tickers loaded — exiting.")
        return

    # ── Candidate scoring ─────────────────────────────────────────────────────
    log.info(
        "Requesting up to %d ML-ranked candidates from %d tickers ...",
        scan_top_n, len(tickers),
    )
    picks = scanner.get_top_picks(tickers, n=scan_top_n)
    for _p in picks:
        _p.setdefault('source', 'ml_model')
    risk_rejected: list[dict] = []
    annotate_mispricing_scores(picks)
    record_model_predictions(db, picks)

    if not picks:
        log.info(
            "No ML candidates returned — check that the champion model registry "
            "exists at ml_scanner.registry_path and that Alpaca credentials are set."
        )
        write_scan_audit([], [], db=db)
        return

    # ── Deduplicate against open positions ───────────────────────────────────
    if open_keys:
        kept_picks: list[dict] = []
        skipped = 0
        for p in picks:
            if (p['symbol'], p['strategy']) in open_keys:
                item = dict(p)
                item['filtered_stage'] = 'Open-position dedup'
                item['reject_reason'] = 'Same symbol and strategy already held as an open or pending-close position'
                item['mispricing_score'] = mispricing_score_for_pick(item)
                risk_rejected.append(item)
                skipped += 1
            else:
                kept_picks.append(p)
        picks = kept_picks
        if skipped:
            log.info(
                f"Dedup: skipped {skipped} pick(s) already held as open positions."
            )

    if not picks:
        log.info("All picks are already held as open positions — nothing new to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return

    # ── Pre-flight: filter picks whose contracts are inactive on Alpaca ───────
    picks, _rejected = executor.preflight_check_picks(picks)
    for _pick in _rejected:
        _pick['filtered_stage'] = 'Pre-flight contract validation'
        bad = ', '.join(_pick.get('_inactive_contracts') or [])
        _pick['reject_reason'] = f"Inactive/non-tradeable contract leg(s): {bad or 'unknown'}"
        _pick['mispricing_score'] = mispricing_score_for_pick(_pick)
        risk_rejected.append(_pick)
    if not picks:
        log.info("No picks survived pre-flight contract validation — nothing to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return

    before_gate = list(picks)
    picks = filter_max_loss_multiple(picks, config)
    capture_rejections(
        before_gate,
        picks,
        'Max-loss multiple',
        lambda p: (
            f"Max-loss multiple {p.get('max_loss_multiple', max_loss_multiple_for_pick(p)):.2f}x "
            "exceeded configured limit"
        ),
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived max-loss multiple filtering — nothing to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return

    # ── Apply capital budget (net of already-deployed capital) ───────────────
    if capital_budget is not None:
        remaining_budget = capital_budget - deployed_capital
        if deployed_capital > 0:
            log.info(
                f"Capital budget ${capital_budget:,.0f} — "
                f"${deployed_capital:,.0f} already deployed → "
                f"${remaining_budget:,.0f} available for new picks."
            )
        if remaining_budget <= 0:
            log.info("No remaining capital budget for new picks.")
            for _pick in picks:
                _item = dict(_pick)
                _item['filtered_stage'] = 'Capital budget'
                _item['reject_reason'] = 'No remaining capital budget after open positions'
                _item['mispricing_score'] = mispricing_score_for_pick(_item)
                risk_rejected.append(_item)
            write_scan_audit([], risk_rejected, db=db)
            return
        max_contracts = int(config.get('max_contracts_per_pick', 50))
        picks_sorted  = sorted(picks, key=lambda x: x.get('score', 0.0), reverse=True)
        # Keep only picks we can afford at least 1 contract of
        affordable = [p for p in picks_sorted if capital_for_pick(p) <= remaining_budget]
        capture_rejections(
            picks_sorted,
            affordable,
            'Capital budget',
            lambda p: f"Capital requirement ${capital_for_pick(p):,.0f} exceeded remaining budget ${remaining_budget:,.0f}",
            risk_rejected,
        )
        if not affordable:
            log.info("No picks fit within the remaining capital budget.")
            write_scan_audit([], risk_rejected, db=db)
            return
        # Equal-split remaining budget across affordable picks, then size each
        per_pick_alloc = remaining_budget / len(affordable)
        new_deployed   = 0.0
        for p in affordable:
            cap = capital_for_pick(p)
            qty = max(1, int(per_pick_alloc // cap))
            qty = min(qty, max_contracts)
            p['quantity'] = qty
            new_deployed += cap * qty
        log.info(
            f"Capital budget: {len(affordable)} picks, {sum(p['quantity'] for p in affordable)} "
            f"total contracts — ${new_deployed:,.0f} of ${remaining_budget:,.0f} available deployed "
            f"(max {max_contracts} contracts/pick)"
        )
        picks = affordable

    picks = apply_regime_quantity_multiplier(picks, regime)
    if not picks:
        log.info("No picks survived regime quantity throttle — nothing to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return
    if regime.quantity_multiplier < 1.0:
        log.info(
            "Regime filter applied %.0f%% quantity throttle to %d pick(s).",
            regime.quantity_multiplier * 100,
            len(picks),
        )

    account_capital = config.get('account_capital') or capital_budget
    before_gate = list(picks)
    picks = apply_directional_exposure_caps(
        picks, capital_positions, config, account_capital,
    )
    capture_rejections(
        before_gate,
        picks,
        'Directional exposure cap',
        'Position would exceed configured put/call side exposure limits',
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived directional exposure caps — nothing to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return

    before_gate = list(picks)
    picks = apply_portfolio_gamma_risk(
        picks, capital_positions, config, account_capital, monitor,
    )
    capture_rejections(
        before_gate,
        picks,
        'Portfolio gamma risk',
        'Position would exceed portfolio stress/gamma concentration limits',
        risk_rejected,
    )
    if not picks:
        log.info("No picks survived portfolio gamma-risk controls — nothing to trade.")
        write_scan_audit([], risk_rejected, db=db)
        return

    annotate_mispricing_scores(picks)
    write_scan_audit(picks, risk_rejected, db=db)

    # ── Print open positions with current P&L ────────────────────────────────
    if open_positions:
        print_open_positions(open_positions, monitor)

    # ── Print plan ────────────────────────────────────────────────────────────
    print_plan(picks, capital_budget=capital_budget)

    # ── scan-only mode ────────────────────────────────────────────────────────
    if mode == 'scan-only':
        os.makedirs('data', exist_ok=True)
        out_path = 'data/pending_picks.json'
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(picks, fh, indent=2, default=str)
        log.info(f"Scan-only mode: {len(picks)} picks saved to {out_path}.")
        return

    # ── auto mode ─────────────────────────────────────────────────────────────
    if mode == 'auto':
        # All picks already passed the ML pipeline: XGBoost ranker scoring,
        # large-loss classifier veto, and pick_selection controls.
        # No redundant delta-based prob_win gate — the model IS the gate.
        approved = picks
        log.info(f"Auto mode: executing {len(approved)} ML-approved picks.")
        if not approved:
            log.info("No picks from ML pipeline — nothing submitted.")
            return
        exec_results = execute_picks(approved, executor, db, dry_run)
        for _pick, _oid in exec_results:
            if _oid:
                notifier.send_trade_executed(_pick, _oid, 'AUTO')
        return

    # ── approve mode — replan loop ────────────────────────────────────────────
    max_replans = int(config.get('email', {}).get('max_replan_attempts', 3))
    replan_count = 0

    while True:
        # Stamp pre-computed capital onto each pick so the notifier uses the
        # same value as the terminal display (strike-based, × 100).
        for _p in picks:
            _p['capital'] = capital_for_pick(_p) * _p.get('quantity', 1)

        # Collect the approval decision (list of approved picks, 'REPLAN', or None)
        approved_or_signal = None
        if notifier.enabled:
            _msg_id = notifier.send_trade_plan(picks, capital_budget,
                                               deployed_capital=deployed_capital)
            if _msg_id:
                _result = notifier.wait_for_approval(_msg_id, picks)
                if _result == 'REPLAN':
                    approved_or_signal = 'REPLAN'
                elif _result is not None:
                    approved_or_signal = _result
                elif headless:
                    log.warning(
                        "Email approval timed out in headless mode — skipping this cycle. "
                        "Increase email.approval_timeout_seconds or switch to --mode auto."
                    )
                    return
                else:
                    log.info("Email approval timed out — falling back to TTY.")
                    approved_or_signal = approval_gate(picks)
            elif headless:
                log.warning("Trade plan email failed to send in headless mode — skipping cycle.")
                return
            else:
                log.warning("Trade plan email failed to send — falling back to TTY.")
                approved_or_signal = approval_gate(picks)
        elif headless:
            log.warning(
                "Running headless (--daemon) without email configured — skipping cycle. "
                "Add email credentials to config.json or use --mode auto."
            )
            return
        else:
            approved_or_signal = approval_gate(picks)

        # ── Handle REPLAN ─────────────────────────────────────────────────────
        if approved_or_signal == 'REPLAN':
            replan_count += 1
            if replan_count > max_replans:
                log.warning(
                    "[agent] Max replan attempts (%d) reached — aborting cycle.",
                    max_replans,
                )
                return
            log.info(
                "[agent] REPLAN requested (attempt %d/%d) — requesting fresh model candidates...",
                replan_count, max_replans,
            )
            print(f"\n  Requesting fresh model candidates (attempt {replan_count}/{max_replans}) ...\n")

            new_picks = scanner.get_top_picks(tickers, n=top_n)
            if not new_picks:
                log.info("[agent] Fresh model request returned no picks — aborting cycle.")
                return
            for _p in new_picks:
                _p.setdefault('source', 'ml_model')
            annotate_mispricing_scores(new_picks)
            record_model_predictions(db, new_picks)
            if open_keys:
                new_picks = [p for p in new_picks
                             if (p['symbol'], p['strategy']) not in open_keys]
            if not new_picks:
                log.info("[agent] All fresh model candidates already held as open positions — aborting.")
                return
            new_picks, _ = executor.preflight_check_picks(new_picks)
            if not new_picks:
                log.info("[agent] No picks survived pre-flight after fresh model request — aborting.")
                return
            new_picks = filter_max_loss_multiple(new_picks, config)
            if not new_picks:
                log.info("[agent] No picks survived max-loss multiple filtering after fresh model request — aborting.")
                return
            if capital_budget is not None:
                remaining_budget = capital_budget - deployed_capital
                if remaining_budget <= 0:
                    log.info("[agent] No remaining capital budget after fresh model request — aborting.")
                    return
                _sorted = sorted(new_picks, key=lambda x: x.get('score', 0.0), reverse=True)
                budgeted: list[dict] = []
                _new_dep = 0.0
                for _p in _sorted:
                    _cap = capital_for_pick(_p)
                    if _new_dep + _cap <= remaining_budget:
                        budgeted.append(_p)
                        _new_dep += _cap
                new_picks = budgeted
            if not new_picks:
                log.info("[agent] No picks fit within capital budget after fresh model request — aborting.")
                return
            new_picks = apply_directional_exposure_caps(
                new_picks, capital_positions, config, account_capital,
            )
            if not new_picks:
                log.info("[agent] No picks survived directional exposure caps after fresh model request — aborting.")
                return
            new_picks = apply_portfolio_gamma_risk(
                new_picks, capital_positions, config, account_capital, monitor,
            )
            if not new_picks:
                log.info("[agent] No picks survived portfolio gamma-risk controls after fresh model request — aborting.")
                return

            picks = new_picks
            print_plan(picks, capital_budget=capital_budget)
            continue   # ← loop back to approval with fresh picks

        # ── Normal execution ──────────────────────────────────────────────────
        approved = approved_or_signal or []
        if not approved:
            return

        # In headless mode the email reply IS the confirmation; skip the TTY prompt.
        if not headless and not confirm_execution(approved, dry_run):
            print("  Execution cancelled.")
            return

        exec_results = execute_picks(approved, executor, db, dry_run)
        for _pick, _oid in exec_results:
            if _oid:
                notifier.send_trade_executed(_pick, _oid, 'TRADE_PLAN')

        # Save the full plan (including rejected picks) for auditing
        os.makedirs('data', exist_ok=True)
        with open('data/pending_picks.json', 'w', encoding='utf-8') as fh:
            json.dump(picks, fh, indent=2, default=str)
        log.info("Full plan saved to data/pending_picks.json for audit trail.")
        break  # normal exit from replan loop


def run_agent(argv=None) -> None:
    try:
        from dotenv import load_dotenv as _load_dotenv
        _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        _load_dotenv(_env_path, override=False)
    except ImportError:
        pass
    args = _parse_args()
    if args.log_file:
        get_logger('optionwheel', log_file=args.log_file)
    if args.daemon:
        run_daemon(args)
    else:
        _run_once(args, headless=False)


if __name__ == "__main__":
    run_agent()
