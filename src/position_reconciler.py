"""
PositionReconciler — Alpaca ↔ Database Sync Daemon
====================================================

Runs independently from the risk manager on a slower interval (default 5 min).
Detects and resolves four types of drift between OptionWheel's local database
and Alpaca's live option positions:

  1. PENDING opens      — open order submitted; confirm fill or void if unfilled
  2. PENDING_CLOSE      — close in-flight; confirm if legs gone from Alpaca, else reopen
  3. Ghost EXECUTED     — DB says EXECUTED but all legs gone from Alpaca
                          (expired naturally, assignment, or manually closed externally)
  4. Orphan Alpaca      — option position in Alpaca with no matching DB record
                          (manually opened, or DB record was lost); log warning only

All four checks are defensive — every resolution path is logged clearly and,
where the outcome is ambiguous (orphans, externally-closed P&L), the operator
is prompted to verify manually via the Alpaca dashboard.
"""
from __future__ import annotations

import datetime
import logging
import numbers
from typing import TYPE_CHECKING

from src.order_status import normalize_order_status

if TYPE_CHECKING:
    from src.database import TradeDatabase
    from src.executor import AlpacaExecutor

_log = logging.getLogger('optionwheel')


# ── OSI symbol builder (mirrors executor._osi_symbol) ─────────────────────────

def _osi(symbol: str, expiry: str, strike: float, option_type: str) -> str:
    ymd         = expiry.replace('-', '')[2:]
    flag        = 'C' if option_type.upper() == 'CALL' else 'P'
    strike_int  = int(round(float(strike) * 1000))
    return f"{symbol}{ymd}{flag}{strike_int:08d}"


def _pos_to_osi_set(pos: dict) -> set[str]:
    """Convert a DB position row to the set of OSI symbols for its legs."""
    symbol = pos.get('symbol', '')
    expiry = pos.get('expiry', '')
    strat  = pos.get('type',   '')
    legs   = pos.get('legs') or {}

    osi: set[str] = set()

    if strat == 'CSP':
        s = legs.get('short_strike') or pos.get('strike')
        if s: osi.add(_osi(symbol, expiry, s, 'PUT'))

    elif strat == 'CC':
        s = legs.get('short_strike') or legs.get('short_call') or pos.get('strike')
        if s: osi.add(_osi(symbol, expiry, s, 'CALL'))

    elif strat == 'PCS':
        ss = legs.get('short_strike') or legs.get('short_put')
        ls = legs.get('long_strike')  or legs.get('long_put')
        if ss: osi.add(_osi(symbol, expiry, ss, 'PUT'))
        if ls: osi.add(_osi(symbol, expiry, ls, 'PUT'))

    elif strat == 'CCS':
        ss = legs.get('short_strike') or legs.get('short_call')
        ls = legs.get('long_strike')  or legs.get('long_call')
        if ss: osi.add(_osi(symbol, expiry, ss, 'CALL'))
        if ls: osi.add(_osi(symbol, expiry, ls, 'CALL'))

    elif strat in ('IC', 'IFLY'):
        for key, opt_type in [('short_put', 'PUT'),  ('long_put',  'PUT'),
                               ('short_call', 'CALL'), ('long_call', 'CALL')]:
            s = legs.get(key)
            if s: osi.add(_osi(symbol, expiry, s, opt_type))

    elif strat == 'STRANGLE':
        sp = legs.get('short_put');  sc = legs.get('short_call')
        if sp: osi.add(_osi(symbol, expiry, sp, 'PUT'))
        if sc: osi.add(_osi(symbol, expiry, sc, 'CALL'))

    return osi


# ── Reconciler ─────────────────────────────────────────────────────────────────

class PositionReconciler:
    """
    Keeps OptionWheel's database in sync with Alpaca's live option positions.

    Instantiate once per monitor cycle and call ``run()``::

        reconciler = PositionReconciler(db, executor)
        summary    = reconciler.run()

    Each of the four checks is independent; a failure in one does not block
    the others.  The returned ``summary`` dict contains per-check counts and
    can be used for monitoring/alerting.
    """

    def __init__(
        self,
        db: TradeDatabase,
        executor: AlpacaExecutor,
        ghost_grace_minutes: int = 10,
    ) -> None:
        self.db                  = db
        self.executor            = executor
        self.ghost_grace_minutes = ghost_grace_minutes

    # ── Alpaca query ───────────────────────────────────────────────────────────

    def get_alpaca_option_symbols(self) -> set[str]:
        """
        Return the set of OSI symbol strings for every open option position
        currently held in the Alpaca account.

        Raises ``RuntimeError`` if the Alpaca client cannot be reached.
        """
        if not self.executor.is_logged_in:
            if not self.executor.login():
                raise RuntimeError("Cannot connect to Alpaca for reconciliation")

        positions = self.executor.client.get_all_positions()
        result: set[str] = set()
        for p in positions:
            ac = str(getattr(p, 'asset_class', '')).lower()
            if 'option' in ac:
                result.add(str(p.symbol))
        _log.debug("[reconciler] Alpaca holds %d open option position leg(s).", len(result))
        return result

    # ── Case 1: PENDING opens ──────────────────────────────────────────────────

    def reconcile_pending_opens(self) -> dict:
        """
        For each PENDING row, query Alpaca for the order's current status.

        - Filled   → ``confirm_open`` + ``update_premium`` with actual fill price
        - Terminal (canceled/rejected/expired/done_for_day) → ``void_trade``
        - Still pending / unknown  → leave as-is (will be retried next cycle)

        Returns ``{'confirmed': N, 'voided': N, 'still_pending': N}``.
        """
        pending = self.db.get_pending_open_positions()
        if not pending:
            return {'confirmed': 0, 'voided': 0, 'still_pending': 0}

        _FILLED   = {'filled', 'partially_filled'}
        _TERMINAL = {'canceled', 'expired', 'rejected', 'done_for_day'}

        confirmed = voided = still_pending = 0

        for pos in pending:
            order_id = pos.get('order_id')
            pos_id   = pos['id']

            if not order_id:
                # No order_id — cannot check; void as unrecoverable
                _log.warning(
                    "[reconciler] PENDING id=%s has no order_id — voiding.", pos_id
                )
                self.db.void_trade(pos_id)
                voided += 1
                continue

            try:
                order  = self.executor.client.get_order_by_id(order_id)
                status = normalize_order_status(getattr(order, 'status', ''))
                self.db.upsert_trade_order(pos_id, order_id, role='OPEN', order=order)
            except Exception as exc:
                _log.warning(
                    "[reconciler] Cannot fetch order %s for PENDING id=%s: %s — skipping.",
                    order_id, pos_id, exc,
                )
                still_pending += 1
                continue

            if status in _FILLED:
                self.db.confirm_open(pos_id, order_id)
                fill = getattr(order, 'filled_avg_price', None)
                if fill is not None:
                    fill = abs(float(fill))
                    self.db.update_premium(pos_id, fill)
                    self.db.upsert_trade_order(
                        pos_id,
                        order_id,
                        role='OPEN',
                        status=status,
                        filled_avg_price=fill,
                        order=order,
                    )
                _log.info(
                    "[reconciler] PENDING id=%s confirmed EXECUTED (order=%s, fill=%.4f)",
                    pos_id, order_id, fill or 0,
                )
                confirmed += 1

            elif status in _TERMINAL:
                self.db.void_trade(pos_id)
                _log.info(
                    "[reconciler] PENDING id=%s voided — order %s status='%s'.",
                    pos_id, order_id, status,
                )
                voided += 1

            else:
                _log.debug(
                    "[reconciler] PENDING id=%s order %s still '%s' — leaving.",
                    pos_id, order_id, status,
                )
                still_pending += 1

        return {'confirmed': confirmed, 'voided': voided, 'still_pending': still_pending}

    # ── Case 2: PENDING_CLOSE ──────────────────────────────────────────────────

    def reconcile_pending_closes(self, alpaca_symbols: set[str]) -> dict:
        """
        For each PENDING_CLOSE row, check whether its legs are still in Alpaca.

        - All legs absent  → close order went through; ``confirm_close``
          (P&L recorded as 0 — no close order_id was stored pre-crash; operator
          should correct via ``db.update_status`` if the exact figure matters)
        - Any leg present and close order still open → leave PENDING_CLOSE
        - Any leg present and close order terminal/missing → ``reopen_position`` for retry

        Returns ``{'confirmed': N, 'reopened': N}``.
        """
        stuck = self.db.get_pending_close_positions()
        if not stuck:
            return {'confirmed': 0, 'reopened': 0}

        _OPEN     = {'new', 'accepted', 'pending_new', 'partially_filled', 'held'}
        _TERMINAL = {'canceled', 'expired', 'rejected', 'done_for_day'}
        confirmed = reopened = 0

        for pos in stuck:
            pos_id = pos['id']
            osis   = _pos_to_osi_set(pos)

            if not osis:
                _log.warning(
                    "[reconciler] PENDING_CLOSE id=%s — cannot build OSI symbols "
                    "(missing legs data); manual reconciliation required.",
                    pos_id,
                )
                continue

            legs_still_in_alpaca = osis & alpaca_symbols
            order_id = pos.get('close_order_id') or pos.get('order_id')
            order_status = None

            order = None
            if order_id:
                try:
                    order = self.executor.client.get_order_by_id(order_id)
                    order_status = normalize_order_status(getattr(order, 'status', ''))
                    self.db.upsert_trade_order(pos_id, order_id, role='CLOSE', order=order)
                except Exception as exc:
                    if legs_still_in_alpaca:
                        _log.warning(
                            "[reconciler] PENDING_CLOSE id=%s %s %s — legs still "
                            "in Alpaca, but close order %s could not be fetched "
                            "(%s); leaving PENDING_CLOSE.",
                            pos_id, pos.get('type'), pos.get('symbol'), order_id, exc,
                        )
                        continue
                    _log.warning(
                        "[reconciler] PENDING_CLOSE id=%s %s %s — legs absent but "
                        "close order %s could not be fetched (%s); leaving "
                        "PENDING_CLOSE for another reconciliation cycle.",
                        pos_id, pos.get('type'), pos.get('symbol'), order_id, exc,
                    )
                    continue

                if order_status in _OPEN:
                    _log.info(
                        "[reconciler] PENDING_CLOSE id=%s %s %s — close order "
                        "%s still '%s'; leaving PENDING_CLOSE.",
                        pos_id, pos.get('type'), pos.get('symbol'), order_id, order_status,
                    )
                    continue

                if order_status not in _TERMINAL and order_status not in {'filled'}:
                    _log.warning(
                        "[reconciler] PENDING_CLOSE id=%s %s %s — close order "
                        "%s has unknown status '%s'; leaving PENDING_CLOSE.",
                        pos_id, pos.get('type'), pos.get('symbol'), order_id, order_status,
                    )
                    continue

            if order_status == 'filled' or not legs_still_in_alpaca:
                # All legs gone — the close order filled.
                # Use the estimated P&L and close order ID stored by the monitor
                # when it submitted the order.  Fall back to safe defaults for
                # crash-recovery cases where those were never written.
                stored_pnl = pos.get('pnl') or 0.0
                pnl_source = 'PENDING_CLOSE_ESTIMATE'
                pnl_verified = False
                if order is not None:
                    fill = getattr(order, 'filled_avg_price', None)
                    if fill is not None:
                        if isinstance(fill, numbers.Real) or isinstance(fill, str):
                            try:
                                fill = abs(float(fill))
                            except (TypeError, ValueError):
                                fill = None
                        else:
                            fill = None
                        if fill is not None:
                            premium = float(pos.get('premium') or 0)
                            contracts = int(pos.get('contracts') or 1)
                            stored_pnl = round((premium - fill) * 100 * contracts, 2)
                            pnl_source = 'ALPACA_FILLS'
                            pnl_verified = True
                            self.db.upsert_trade_order(
                                pos_id,
                                order_id,
                                role='CLOSE',
                                status='filled',
                                filled_avg_price=fill,
                                order=order,
                            )
                stored_oid = order_id or 'RECONCILED'
                if stored_oid == 'RECONCILED':
                    pnl_source = 'RECONCILED_PLACEHOLDER'
                    pnl_verified = False
                self.db.confirm_close(
                    pos_id,
                    pnl=stored_pnl,
                    close_order_id=stored_oid,
                    pnl_source=pnl_source,
                    pnl_verified=pnl_verified,
                )
                _log.info(
                    "[reconciler] PENDING_CLOSE id=%s %s %s — legs absent from Alpaca; "
                    "confirmed CLOSED (pnl=$%.2f, order=%s).",
                    pos_id, pos.get('type'), pos.get('symbol'), stored_pnl, stored_oid,
                )
                confirmed += 1
            else:
                # Legs still present and no active close order — retry later.
                self.db.reopen_position(pos_id)
                _log.warning(
                    "[reconciler] PENDING_CLOSE id=%s %s %s — legs still in Alpaca; "
                    "reopened to EXECUTED for retry next risk cycle.",
                    pos_id, pos.get('type'), pos.get('symbol'),
                )
                reopened += 1

        return {'confirmed': confirmed, 'reopened': reopened}

    # ── Case 3: Ghost EXECUTED positions ──────────────────────────────────────

    def reconcile_executed(self, alpaca_symbols: set[str]) -> dict:
        """
        Find EXECUTED positions whose legs have all disappeared from Alpaca.

        Two sub-cases:
        - Expiry has passed → option expired naturally; close with full premium as P&L
        - Expiry still future → externally closed (manual close, assignment, roll);
          close with P&L=$0 and log a warning so the operator can correct it

        Positions created within ``ghost_grace_minutes`` are skipped — in auto
        mode the agent and reconciler run concurrently, and Alpaca MLEG positions
        can take several seconds to appear in the positions API after a fill.
        Closing a freshly-written EXECUTED row before Alpaca reflects the fill
        would produce a false ghost.

        DRY_RUN positions are ignored — they never had real Alpaca legs.

        Returns ``{'ghost_closed': N, 'grace_skipped': N}``.
        """
        today    = datetime.date.today().isoformat()
        grace_cutoff = (
            datetime.datetime.now()
            - datetime.timedelta(minutes=self.ghost_grace_minutes)
        )
        # Use get_all_executed_positions so that past-expiry ghost rows (where
        # the monitor was offline at expiry) are also caught and closed.
        open_pos = self.db.get_all_executed_positions()
        ghosts = 0
        skipped = 0

        for pos in open_pos:
            # Grace period: skip positions created very recently — the agent may
            # have just written this row and Alpaca hasn't reflected the fill yet.
            ts_raw = pos.get('timestamp', '')
            if ts_raw:
                try:
                    ts = datetime.datetime.fromisoformat(ts_raw)
                    if ts > grace_cutoff:
                        skipped += 1
                        continue
                except ValueError:
                    pass

            osis = _pos_to_osi_set(pos)
            if not osis:
                continue  # can't verify without OSI symbols; skip silently

            if osis & alpaca_symbols:
                continue  # at least one leg still in Alpaca — position is live

            # No legs found in Alpaca
            pos_id = pos['id']
            expiry = pos.get('expiry', '')
            strat  = pos.get('type',  '?')
            sym    = pos.get('symbol', '?')
            prem   = float(pos.get('premium') or 0)

            if expiry < today:
                # Expired naturally — all short legs expired worthless, full credit kept
                contracts = int(pos.get('contracts') or 1)
                pnl = round(prem * 100 * contracts, 2)
                self.db.close_position(
                    pos_id,
                    pnl,
                    'EXPIRED_RECONCILED',
                    pnl_source='EXPIRED',
                    pnl_verified=True,
                )
                _log.info(
                    "[reconciler] Ghost id=%s %s %s expired %s — closed in DB (pnl=+$%.2f).",
                    pos_id, strat, sym, expiry, pnl,
                )
            else:
                # Pre-expiry disappearance — manual close, assignment, or early exercise
                self.db.close_position(
                    pos_id,
                    0.0,
                    'EXTERNALLY_CLOSED',
                    pnl_source='EXTERNAL_PLACEHOLDER',
                    pnl_verified=False,
                )
                _log.warning(
                    "[reconciler] Ghost id=%s %s %s — legs absent from Alpaca but expiry %s "
                    "is in the future. Marked CLOSED (P&L=$0 placeholder). "
                    "Likely manually closed or assigned — update P&L manually.",
                    pos_id, strat, sym, expiry,
                )
            ghosts += 1

        return {'ghost_closed': ghosts, 'grace_skipped': skipped}

    # ── Case 3b: Recover false ghosts ─────────────────────────────────────────

    def recover_false_ghosts(self, alpaca_symbols: set[str]) -> dict:
        """
        Reopen positions that were incorrectly closed as EXTERNALLY_CLOSED by
        the ghost-detection race condition.

        Looks for CLOSED rows where order_id='EXTERNALLY_CLOSED', expiry is
        still in the future, AND at least one leg OSI is now present in Alpaca.
        These are positions the reconciler closed prematurely while Alpaca was
        still processing the MLEG fill.  Reopening them restores normal
        stop-loss and P&L tracking.

        Returns ``{'recovered': N}``.
        """
        candidates = self.db.get_false_ghost_positions()
        recovered = 0

        for pos in candidates:
            osis = _pos_to_osi_set(pos)
            if not osis:
                continue

            if not (osis & alpaca_symbols):
                continue  # legs still absent — not a false ghost

            pos_id = pos['id']
            strat  = pos.get('type',   '?')
            sym    = pos.get('symbol', '?')
            expiry = pos.get('expiry', '?')
            self.db.reopen_false_ghost(pos_id)
            _log.info(
                "[reconciler] Recovered false ghost id=%s %s %s (expiry %s) — "
                "legs found in Alpaca, status restored to EXECUTED.",
                pos_id, strat, sym, expiry,
            )
            recovered += 1

        return {'recovered': recovered}

    # ── Case 4: Orphan Alpaca positions ───────────────────────────────────────

    def find_orphans(self, alpaca_symbols: set[str]) -> list[str]:
        """
        Return Alpaca option OSI symbols that have no matching DB record.

        These are positions that were opened outside OptionWheel (manually, via
        another system) or whose DB record was lost.  We log warnings only —
        auto-importing is unsafe because we don't know the entry premium.

        PENDING_CLOSE rows still count as known DB records: their close order is
        in flight, so the legs may legitimately remain at Alpaca until the order
        reaches a terminal status.

        Returns the sorted list of orphan OSI strings.
        """
        known_positions = (
            self.db.get_open_positions()
            + self.db.get_pending_close_positions()
        )
        db_osis: set[str] = set()
        for pos in known_positions:
            db_osis |= _pos_to_osi_set(pos)

        orphans = sorted(alpaca_symbols - db_osis)
        if orphans:
            _log.warning(
                "[reconciler] %d Alpaca option position leg(s) have no DB record "
                "(manually opened or DB record lost — review on Alpaca dashboard):",
                len(orphans),
            )
            for sym in orphans:
                _log.warning("  orphan: %s", sym)
        return orphans

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Run all reconciliation checks in order.

        Failures in individual checks are caught and logged; the remaining
        checks still run.  Returns a summary dict with keys:
          alpaca_positions, pending_opens, pending_closes, recovered,
          executed, orphans.

        Check order matters:
          1. recover_false_ghosts — reopen EXTERNALLY_CLOSED rows whose legs
             are back in Alpaca (must run before reconcile_executed so recovered
             rows are not immediately re-ghosted).
          2. reconcile_pending_opens / reconcile_pending_closes — normal flow.
          3. reconcile_executed — ghost detection with grace period.
          4. find_orphans — log unmatched Alpaca positions.
        """
        _log.info("[reconciler] Starting reconciliation cycle ...")
        summary: dict = {}

        # Fetch Alpaca positions once and share across checks
        try:
            alpaca_symbols = self.get_alpaca_option_symbols()
        except Exception as exc:
            _log.error(
                "[reconciler] Cannot fetch Alpaca positions (%s) — skipping cycle.", exc
            )
            return {'error': str(exc)}

        summary['alpaca_positions'] = len(alpaca_symbols)

        for name, fn, needs_symbols in [
            ('recovered',      self.recover_false_ghosts,    True),
            ('pending_opens',  self.reconcile_pending_opens,  False),
            ('pending_closes', self.reconcile_pending_closes, True),
            ('executed',       self.reconcile_executed,       True),
            ('orphans',        self.find_orphans,             True),
        ]:
            try:
                summary[name] = fn(alpaca_symbols) if needs_symbols else fn()
            except Exception as exc:
                _log.error("[reconciler] %s check failed: %s", name, exc, exc_info=True)
                summary[name] = {'error': str(exc)}

        _log.info("[reconciler] Cycle complete: %s", summary)
        return summary
