"""
agent_execution.py
==================

Approval gate, trade execution, and manual close mechanics.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

import sys
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor
from src.position_monitor import PositionMonitor
from src.agent_risk import capital_for_pick
from src.agent_display import legs_from_pick, print_positions_table
from src.utils import get_logger

log = get_logger()


# ── Approval gate ────────────────────────────────────────────────────────────

def approval_gate(picks: list[dict]):
    """
    Interactively ask the user which picks to approve.

    Input options
    -------------
    a          Approve ALL picks
    n          Reject ALL picks (exit without executing)
    q          Quit the agent entirely
    replan     Discard this plan and request fresh model candidates
    1,3,5      Approve picks by comma-separated number
    1-5        Approve a range of picks
    1,3-5,8   Mix of individual numbers and ranges

    Returns the approved subset of picks, or the string 'REPLAN'.
    """
    if not sys.stdin.isatty():
        # Non-interactive (piped / CI) — safe default: reject all
        print("stdin is not a TTY — running in non-interactive mode.")
        print("Use --mode auto or pipe approval input to run headlessly.")
        return []

    print("  Enter the numbers of the trades to approve, 'a' for all, 'n' for none,")
    print("  or 'replan' to discard this plan and request fresh model candidates.")
    print("  Examples:  a   |   n   |   1,3,5   |   1-5   |   2,4-7,10   |   replan")
    print()

    while True:
        try:
            raw = input("  Your selection: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return []

        if raw in ('q', 'quit'):
            print("Quitting.")
            sys.exit(0)

        if raw in ('n', 'none', ''):
            print("  No picks approved — nothing will be executed.")
            return []

        if raw in ('a', 'all'):
            print(f"  All {len(picks)} picks approved.")
            return picks

        if raw in ('replan', 'rescan', 'retry'):
            print("  REPLAN requested — will request fresh model candidates...")
            return 'REPLAN'

        # Parse individual numbers and ranges
        approved_indices: set[int] = set()
        valid = True
        for part in raw.split(','):
            part = part.strip()
            if '-' in part:
                lo_s, _, hi_s = part.partition('-')
                try:
                    lo, hi = int(lo_s.strip()), int(hi_s.strip())
                    approved_indices.update(range(lo, hi + 1))
                except ValueError:
                    print(f"  Invalid range '{part}'. Try again.")
                    valid = False
                    break
            else:
                try:
                    approved_indices.add(int(part))
                except ValueError:
                    print(f"  Invalid number '{part}'. Try again.")
                    valid = False
                    break

        if not valid:
            continue

        # Map 1-based display numbers back to 0-based list indices
        out_of_range = [n for n in approved_indices if n < 1 or n > len(picks)]
        if out_of_range:
            print(f"  Number(s) out of range (1-{len(picks)}): {out_of_range}. Try again.")
            continue

        approved = [picks[n - 1] for n in sorted(approved_indices)]
        print(f"  {len(approved)} pick(s) approved: {sorted(approved_indices)}")
        return approved


def confirm_execution(approved: list[dict], dry_run: bool) -> bool:
    """Ask for a final 'yes' before submitting orders."""
    if not sys.stdin.isatty():
        return False

    mode_label = "[DRY RUN]" if dry_run else "[LIVE - REAL MONEY]"
    total_cap  = sum(capital_for_pick(p) for p in approved)
    total_prem = sum(p.get('premium', 0) * 100 for p in approved)

    print()
    print(f"  {mode_label}  About to submit {len(approved)} order(s).")
    print(f"  Capital required: ${total_cap:,.0f}   |   Premium collected: ${total_prem:,.2f}")
    if not dry_run:
        print("  WARNING: This will submit REAL orders to Alpaca!")
    print()

    try:
        answer = input("  Confirm? [yes / no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False

    return answer in ('yes', 'y')


# ── Trade execution ──────────────────────────────────────────────────────────

def execute_picks(
    approved: list[dict],
    executor: AlpacaExecutor,
    db: TradeDatabase,
    dry_run: bool,
) -> list[tuple[dict, Optional[str]]]:
    """Submit each approved pick and log it to the database.

    Returns a list of (pick, order_id) tuples — order_id is None on failure.
    """
    results: list[tuple[dict, Optional[str]]] = []
    print()
    for i, pick in enumerate(approved, start=1):
        strat  = pick.get('strategy', '?')
        symbol = pick.get('symbol', '?')
        expiry = pick.get('expiry', '?')
        short  = (pick.get('short_strike')
                  or pick.get('short_put')
                  or pick.get('short_call')
                  or 0)
        prem   = pick.get('premium', 0)
        prob   = pick.get('prob_win', 0)

        print(f"  [{i}/{len(approved)}] {strat} {symbol} {expiry}  "
              f"credit=${prem:.2f}  prob={prob:.1%} ...", end=" ", flush=True)

        # ── Phase 1: pre-insert PENDING row before touching the broker ─────────
        # If the process dies between broker submission and DB write the row
        # is already present (PENDING) and will be visible for reconciliation
        # rather than silently lost.
        if dry_run:
            # Dry-run: log immediately as DRY_RUN (no broker call, no two-phase)
            trade_id = db.log_trade(
                symbol, expiry, short, strat, prem, prob,
                'DRY_RUN', 'DRY_RUN',
                legs=legs_from_pick(pick),
                contracts=pick.get('quantity', 1),
                model_prediction_id=pick.get('model_prediction_id'),
                model_decision_id=pick.get('model_decision_id'),
            )
            print(f"OK  (DRY_RUN)")
            results.append((pick, 'DRY_RUN'))
            continue

        trade_id = db.log_trade(
            symbol, expiry, short, strat, prem, prob,
            'PENDING', None,
            legs=legs_from_pick(pick),
            contracts=pick.get('quantity', 1),
            model_prediction_id=pick.get('model_prediction_id'),
            model_decision_id=pick.get('model_decision_id'),
        )

        # ── Phase 2: submit to broker ──────────────────────────────────────────
        order_id = None
        try:
            order_id = executor.execute_pick(pick, dry_run=False, amount=pick.get('quantity', 1))
        except Exception as exc:
            log.error("FAILED  (%s: %s)", type(exc).__name__, exc, exc_info=True)

        if not order_id:
            db.void_trade(trade_id)
            log.warning("[agent] Broker submission failed for %s %s — PENDING row voided.",
                        strat, symbol)
            print("FAILED")
            results.append((pick, None))
            continue

        # ── Phase 3: record accepted order, then confirm only after fill ──────
        db.record_open_order(trade_id, order_id)

        # Poll Alpaca for the actual fill price before marking EXECUTED so
        # that stop-loss calculations use the real entry premium rather than
        # the (potentially hours-old) scanned value.  If Alpaca reports the
        # order as canceled/expired, void the record so the monitor does not
        # treat it as an open position.
        fill, canceled = executor.get_fill_price(order_id)
        if canceled:
            db.void_trade(trade_id)
            log.warning("[agent] Order %s for %s %s was not filled — "
                        "DB entry voided.", order_id, strat, symbol)
            print(f"NOT FILLED  ({order_id})")
            results.append((pick, None))
            continue
        if fill is None:
            log.warning(
                "[agent] Order %s for %s %s accepted but fill is unconfirmed — "
                "leaving DB row PENDING for reconciliation.",
                order_id, strat, symbol,
            )
            print(f"PENDING  ({order_id})")
            results.append((pick, order_id))
            continue
        if fill is not None:
            # Alpaca returns filled_avg_price as a negative value for
            # net-credit MLEG orders (PCS/CCS/IC/IFLY/STRANGLE).
            # Normalise to a positive credit before storing.
            fill = abs(fill)
        db.confirm_open(trade_id, order_id)
        if abs(fill - prem) > 0.001:
            db.update_premium(trade_id, fill)
            pick['premium'] = fill
            log.info("[agent] Fill price for %s %s: $%.2f (scanned $%.2f)",
                     strat, symbol, fill, prem)

        print(f"OK  ({order_id})")

        results.append((pick, order_id))

    print()
    return results


# ── Manual close helpers ─────────────────────────────────────────────────────

def close_one(pos: dict, executor: AlpacaExecutor, db: TradeDatabase,
              monitor: PositionMonitor, dry_run: bool) -> bool:
    """
    Price and close a single position.  Returns True on success.
    Prints a one-line result summary.
    """
    trade_id = pos['id']
    symbol   = pos.get('symbol', '?')
    strat    = pos.get('type',   '?')

    # Try to get a live mark for a sensible limit price
    current_mark = None
    try:
        # conservative=True: use ask for short legs (buy-to-close) and bid for
        # long legs (sell-to-close) so realized P&L reflects actual fill costs.
        current_mark = monitor._get_current_mark(pos, conservative=True)
    except Exception:
        pass

    limit_px = round(current_mark, 2) if current_mark is not None else None

    contracts = int(pos.get('contracts') or 1)
    premium  = float(pos.get('premium', 0) or 0)
    pnl      = round((premium - current_mark) * 100 * contracts, 2) if current_mark is not None else 0.0
    from src.position_lifecycle import PositionLifecycleService
    result = PositionLifecycleService(db, executor).close_position(
        pos,
        limit_price=limit_px,
        pnl=pnl,
        dry_run=dry_run,
        reason='MANUAL_CLOSE',
    )
    if not result.success:
        log.error("[%s] %s %s — close failed: %s", trade_id, strat, symbol, result.error)
        return False

    tag      = ' [DRY RUN]' if dry_run else ''
    pnl_sign = '+' if pnl >= 0 else ''
    mark_str = f"${current_mark:.4f}" if current_mark is not None else 'N/A'
    action = 'closed' if result.status == 'CLOSED' else 'close submitted'
    pending = ' (already pending)' if result.already_pending else ''
    print(f"  ✓  [{trade_id}] {strat} {symbol} {action}{tag}{pending}  "
          f"mark={mark_str}  est P&L={pnl_sign}${pnl:,.2f}  order={result.order_id or '—'}")
    return True


def run_close_command(args, db: TradeDatabase, executor: AlpacaExecutor,
                      monitor: PositionMonitor, dry_run: bool) -> None:
    """
    Handle --list-open, --close <ids>, and --close-all.
    Exits after completing the requested action.
    """
    open_positions = db.get_open_positions()

    # ── --list-open ───────────────────────────────────────────────────────────
    if args.list_open:
        print()
        print("=" * 72)
        print(f"  OPEN POSITIONS  ({len(open_positions)} found)")
        print("=" * 72)
        print_positions_table(open_positions)
        print("=" * 72)
        sys.exit(0)

    # ── --close <ids> ─────────────────────────────────────────────────────────
    if args.close_ids:
        by_id = {p['id']: p for p in open_positions}
        missing = [i for i in args.close_ids if i not in by_id]
        if missing:
            print(f"  [WARN] Trade ID(s) not found among open positions: {missing}")

        targets = [by_id[i] for i in args.close_ids if i in by_id]
        if not targets:
            print("  Nothing to close.")
            sys.exit(1)

        tag = ' [DRY RUN]' if dry_run else ' [LIVE]'
        print()
        print("=" * 72)
        print(f"  MANUAL CLOSE{tag}  —  {len(targets)} position(s)")
        print("=" * 72)
        print_positions_table(targets)

        # Confirmation prompt (skip in dry-run)
        if not dry_run:
            ans = input(f"  Close {len(targets)} position(s) with REAL orders? [y/N] ").strip().lower()
            if ans != 'y':
                print("  Aborted.")
                sys.exit(0)

        ok = sum(close_one(p, executor, db, monitor, dry_run) for p in targets)
        print()
        print(f"  Closed {ok}/{len(targets)} position(s).")
        print("=" * 72)
        sys.exit(0 if ok == len(targets) else 1)

    # ── --close-all ───────────────────────────────────────────────────────────
    if args.close_all:
        if not open_positions:
            print("  No open positions to close.")
            sys.exit(0)

        tag = ' [DRY RUN]' if dry_run else ' [LIVE]'
        print()
        print("=" * 72)
        print(f"  CLOSE ALL POSITIONS{tag}  —  {len(open_positions)} position(s)")
        print("=" * 72)
        print_positions_table(open_positions)

        if not dry_run:
            ans = input(f"  Close ALL {len(open_positions)} position(s) with REAL orders? [y/N] ").strip().lower()
            if ans != 'y':
                print("  Aborted.")
                sys.exit(0)

        ok = sum(close_one(p, executor, db, monitor, dry_run) for p in open_positions)
        print()
        print(f"  Closed {ok}/{len(open_positions)} position(s).")
        print("=" * 72)
        sys.exit(0 if ok == len(open_positions) else 1)
