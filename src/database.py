import sqlite3
import datetime
import json
import os


class TradeDatabase:
    def __init__(self, db_path="data/trades.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._migrate()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT,
                    symbol      TEXT,
                    expiry      TEXT,
                    strike      REAL,
                    type        TEXT,
                    premium     REAL,
                    prob_expiry REAL,
                    status      TEXT,
                    order_id    TEXT,
                    pnl         REAL DEFAULT 0,
                    legs        TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_orders (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id         INTEGER NOT NULL,
                    order_id         TEXT NOT NULL UNIQUE,
                    role             TEXT NOT NULL,
                    status           TEXT,
                    submitted_at     TEXT,
                    filled_at        TEXT,
                    filled_qty       REAL,
                    filled_avg_price REAL,
                    raw_json         TEXT,
                    updated_at       TEXT,
                    FOREIGN KEY(trade_id) REFERENCES trades(id)
                )
            """)
            conn.commit()

    def _migrate(self):
        """Add any columns introduced after the initial schema (non-destructive)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(trades)")
            existing = {row[1] for row in cursor.fetchall()}
            if 'legs' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN legs TEXT")
                conn.commit()
            if 'contracts' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN contracts INTEGER DEFAULT 1")
                conn.commit()
            if 'open_order_id' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN open_order_id TEXT")
                conn.commit()
            if 'close_order_id' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN close_order_id TEXT")
                conn.commit()
            if 'close_reason' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN close_reason TEXT")
                conn.commit()
            if 'status_updated_at' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN status_updated_at TEXT")
                conn.commit()
            if 'pnl_source' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN pnl_source TEXT")
                conn.commit()
            if 'pnl_verified' not in existing:
                cursor.execute("ALTER TABLE trades ADD COLUMN pnl_verified INTEGER DEFAULT 0")
                conn.commit()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_orders (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id         INTEGER NOT NULL,
                    order_id         TEXT NOT NULL UNIQUE,
                    role             TEXT NOT NULL,
                    status           TEXT,
                    submitted_at     TEXT,
                    filled_at        TEXT,
                    filled_qty       REAL,
                    filled_avg_price REAL,
                    raw_json         TEXT,
                    updated_at       TEXT,
                    FOREIGN KEY(trade_id) REFERENCES trades(id)
                )
            """)
            conn.commit()
            cursor.execute(
                """
                UPDATE trades
                   SET pnl_source = CASE
                         WHEN order_id IN ('EXPIRED', 'EXPIRED_RECONCILED')
                              OR close_order_id IN ('EXPIRED', 'EXPIRED_RECONCILED')
                              THEN 'EXPIRED'
                         WHEN order_id = 'DRY_RUN_CLOSE'
                              OR close_order_id = 'DRY_RUN_CLOSE'
                              THEN 'DRY_RUN_ESTIMATE'
                         WHEN order_id = 'EXTERNALLY_CLOSED'
                              OR close_order_id = 'EXTERNALLY_CLOSED'
                              THEN 'EXTERNAL_PLACEHOLDER'
                         ELSE 'LEGACY_RECORDED'
                       END,
                       pnl_verified = CASE
                         WHEN order_id IN ('EXPIRED', 'EXPIRED_RECONCILED')
                              OR close_order_id IN ('EXPIRED', 'EXPIRED_RECONCILED')
                              THEN 1
                         ELSE COALESCE(pnl_verified, 0)
                       END
                 WHERE status = 'CLOSED'
                   AND pnl_source IS NULL
                """
            )
            conn.commit()

    @staticmethod
    def _order_attr(order, name, default=None):
        if isinstance(order, dict):
            return order.get(name, default)
        return getattr(order, name, default)

    @classmethod
    def _order_raw_json(cls, order) -> str | None:
        if order is None:
            return None
        try:
            if isinstance(order, dict):
                return json.dumps(order, default=str)
            if hasattr(order, 'model_dump'):
                return json.dumps(order.model_dump(), default=str)
            if hasattr(order, 'dict'):
                return json.dumps(order.dict(), default=str)
            data = {
                name: cls._order_attr(order, name)
                for name in (
                    'id', 'status', 'submitted_at', 'filled_at',
                    'filled_qty', 'filled_avg_price'
                )
                if cls._order_attr(order, name) is not None
            }
            return json.dumps(data, default=str) if data else None
        except Exception:
            return None

    def upsert_trade_order(
        self,
        trade_id: int,
        order_id: str,
        *,
        role: str,
        status: str = None,
        filled_qty: float = None,
        filled_avg_price: float = None,
        submitted_at: str = None,
        filled_at: str = None,
        raw_json: str = None,
        order=None,
    ) -> None:
        """Insert or update the Alpaca order fact ledger for one strategy trade."""
        if not order_id:
            return
        if order is not None:
            status = status or self._order_attr(order, 'status')
            filled_qty = filled_qty if filled_qty is not None else self._order_attr(order, 'filled_qty')
            filled_avg_price = (
                filled_avg_price
                if filled_avg_price is not None
                else self._order_attr(order, 'filled_avg_price')
            )
            submitted_at = submitted_at or self._order_attr(order, 'submitted_at')
            filled_at = filled_at or self._order_attr(order, 'filled_at')
            raw_json = raw_json or self._order_raw_json(order)

        def _float_or_none(value):
            if value in (None, ''):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        now = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trade_orders (
                    trade_id, order_id, role, status, submitted_at, filled_at,
                    filled_qty, filled_avg_price, raw_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    trade_id         = excluded.trade_id,
                    role             = excluded.role,
                    status           = COALESCE(excluded.status, trade_orders.status),
                    submitted_at     = COALESCE(excluded.submitted_at, trade_orders.submitted_at),
                    filled_at        = COALESCE(excluded.filled_at, trade_orders.filled_at),
                    filled_qty       = COALESCE(excluded.filled_qty, trade_orders.filled_qty),
                    filled_avg_price = COALESCE(excluded.filled_avg_price, trade_orders.filled_avg_price),
                    raw_json         = COALESCE(excluded.raw_json, trade_orders.raw_json),
                    updated_at       = excluded.updated_at
                """,
                (
                    trade_id, order_id, role, str(status).lower() if status else None,
                    str(submitted_at) if submitted_at is not None else None,
                    str(filled_at) if filled_at is not None else None,
                    _float_or_none(filled_qty),
                    _float_or_none(filled_avg_price),
                    raw_json,
                    now,
                ),
            )
            conn.commit()
    def get_trade_orders(self, trade_id: int) -> list[dict]:
        """Return broker order facts linked to a trade, oldest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_orders WHERE trade_id = ? ORDER BY id ASC",
                (trade_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Write ─────────────────────────────────────────────────────────────────

    def log_trade(
        self,
        symbol,
        expiry,
        strike,
        type,
        premium,
        prob_expiry,
        status="PENDING",
        order_id=None,
        legs=None,
        contracts=1,
    ):
        """
        Insert a new trade row.

        *legs*      — dict of all leg strikes, e.g.
                      {'short_put': 150, 'long_put': 145, 'short_call': 160, 'long_call': 165}
                      Stored as JSON; used by PositionMonitor for stop-loss pricing.
        *contracts* — number of contracts placed (default 1).
        """
        legs_json = json.dumps(legs) if legs else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades
                    (timestamp, symbol, expiry, strike, type, premium, prob_expiry,
                     status, order_id, legs, contracts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.datetime.now().isoformat(),
                    symbol, expiry, strike, type, premium, prob_expiry,
                    status, order_id, legs_json, int(contracts),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_status(self, trade_id, status, pnl=0):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status = ?,
                       pnl = ?,
                       pnl_source = CASE
                           WHEN UPPER(?) = 'CLOSED' THEN COALESCE(pnl_source, 'MANUAL_OVERRIDE')
                           ELSE pnl_source
                       END,
                       pnl_verified = CASE
                           WHEN UPPER(?) = 'CLOSED' THEN COALESCE(pnl_verified, 0)
                           ELSE pnl_verified
                       END,
                       status_updated_at = ?
                 WHERE id = ?
                """,
                (
                    status, pnl, status, status,
                    datetime.datetime.now().isoformat(), trade_id,
                ),
            )
            conn.commit()

    def confirm_open(self, trade_id: int, order_id: str) -> None:
        """
        Two-phase open — step 2 (success path).

        Transition PENDING → EXECUTED once the broker has accepted the order.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status            = 'EXECUTED',
                       order_id          = ?,
                       open_order_id     = COALESCE(open_order_id, ?),
                       status_updated_at = ?
                 WHERE id = ?
                """,
                (order_id, order_id, datetime.datetime.now().isoformat(), trade_id),
            )
            conn.commit()
        if order_id:
            self.upsert_trade_order(trade_id, order_id, role='OPEN', status='filled')

    def record_open_order(self, trade_id: int, order_id: str) -> None:
        """
        Store an accepted opening order ID without marking the trade EXECUTED.

        The row remains PENDING until a fill is confirmed.  This prevents the
        monitor/reconciler from treating an accepted-but-unfilled order as a
        real broker position.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET order_id          = ?,
                       open_order_id     = ?,
                       status_updated_at = ?
                 WHERE id = ?
                   AND status = 'PENDING'
                """,
                (order_id, order_id, datetime.datetime.now().isoformat(), trade_id),
            )
            conn.commit()
        self.upsert_trade_order(trade_id, order_id, role='OPEN', status='accepted')

    def void_trade(self, trade_id: int) -> None:
        """Mark a trade VOID — order was submitted but never filled by the broker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE trades SET status = 'VOID', status_updated_at = ? WHERE id = ?",
                (datetime.datetime.now().isoformat(), trade_id),
            )
            conn.commit()

    def get_trade(self, trade_id: int) -> dict | None:
        """Return one trade row by id with legs decoded."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        raw = out.get('legs')
        if raw and isinstance(raw, str):
            try:
                out['legs'] = json.loads(raw)
            except Exception:
                out['legs'] = {}
        return out

    def update_premium(self, trade_id: int, premium: float) -> None:
        """Update the recorded entry premium after confirming the actual Alpaca fill price."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE trades SET premium = ? WHERE id = ?", (premium, trade_id))
            conn.commit()

    def fix_negative_premiums(self) -> int:
        """
        Correct open-position records where ``premium`` is negative.

        Alpaca returns ``filled_avg_price`` as a negative value for net-credit
        MLEG orders; if the agent recorded the raw fill before the abs() guard
        was added, the DB ends up with negative premiums.  This method
        normalises them to their absolute value.

        Returns the number of rows updated.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "UPDATE trades SET premium = ABS(premium) "
                "WHERE premium < 0 AND status NOT IN ('CLOSED', 'VOID')"
            )
            conn.commit()
            return cur.rowcount

    def mark_pending_close(
        self,
        trade_id: int,
        *,
        pnl: float = None,
        close_order_id: str = None,
        reason: str = None,
    ) -> None:
        """
        Two-phase close — step 1 (and optional step 1b).

        Call with no extra args *before* submitting the close order: sets
        status=PENDING_CLOSE so a crash between the broker call and the DB
        update never leaves the position silently open.

        Call again *after* the broker accepts the order, passing ``pnl`` and
        ``close_order_id``, to record the estimated P&L and the close order
        reference.  The reconciler uses these values when it later confirms
        the fill (legs gone from Alpaca), so it doesn't have to write a $0
        placeholder.

        Call ``confirm_close`` on success or ``reopen_position`` on failure.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status   = 'PENDING_CLOSE',
                       pnl      = COALESCE(?, pnl),
                       order_id = COALESCE(?, order_id),
                       close_order_id = COALESCE(?, close_order_id),
                       close_reason   = COALESCE(?, close_reason),
                       status_updated_at = ?
                 WHERE id = ?
                """,
                (
                    pnl, close_order_id, close_order_id, reason,
                    datetime.datetime.now().isoformat(), trade_id,
                ),
            )
            conn.commit()
        if close_order_id:
            self.upsert_trade_order(trade_id, close_order_id, role='CLOSE', status='accepted')

    def claim_pending_close(
        self,
        trade_id: int,
        *,
        pnl: float = None,
        reason: str = None,
    ) -> bool:
        """
        Atomically claim an EXECUTED position for live close submission.

        Returns True only for the actor that successfully transitions
        EXECUTED → PENDING_CLOSE.  Concurrent callers get False and must not
        submit another broker close order.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE trades
                   SET status            = 'PENDING_CLOSE',
                       pnl               = COALESCE(?, pnl),
                       close_reason      = COALESCE(?, close_reason),
                       status_updated_at = ?
                 WHERE id = ?
                   AND status = 'EXECUTED'
                """,
                (pnl, reason, datetime.datetime.now().isoformat(), trade_id),
            )
            conn.commit()
            return cur.rowcount == 1

    def record_pending_close_order(
        self,
        trade_id: int,
        *,
        pnl: float = None,
        close_order_id: str = None,
        reason: str = None,
    ) -> bool:
        """Attach broker close order details to a claimed PENDING_CLOSE row."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE trades
                   SET pnl               = COALESCE(?, pnl),
                       order_id          = COALESCE(?, order_id),
                       close_order_id    = COALESCE(?, close_order_id),
                       close_reason      = COALESCE(?, close_reason),
                       status_updated_at = ?
                 WHERE id = ?
                   AND status = 'PENDING_CLOSE'
                """,
                (
                    pnl, close_order_id, close_order_id, reason,
                    datetime.datetime.now().isoformat(), trade_id,
                ),
            )
            conn.commit()
            recorded = cur.rowcount == 1
        if recorded and close_order_id:
            self.upsert_trade_order(trade_id, close_order_id, role='CLOSE', status='accepted')
        return recorded

    def confirm_close(
        self,
        trade_id: int,
        pnl: float,
        close_order_id: str = None,
        *,
        pnl_source: str = 'ALPACA_FILLS',
        pnl_verified: bool = True,
    ) -> None:
        """
        Two-phase close — step 2 (success path).

        Transition PENDING_CLOSE → CLOSED with confirmed P&L and order ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status   = 'CLOSED',
                       pnl      = ?,
                       order_id = COALESCE(?, order_id),
                       close_order_id = COALESCE(?, close_order_id),
                       pnl_source = ?,
                       pnl_verified = ?,
                       status_updated_at = ?
                 WHERE id = ?
                   AND status IN ('PENDING_CLOSE', 'EXECUTED', 'DRY_RUN')
                """,
                (
                    pnl, close_order_id, close_order_id,
                    pnl_source, 1 if pnl_verified else 0,
                    datetime.datetime.now().isoformat(), trade_id,
                ),
            )
            conn.commit()

    def reopen_position(self, trade_id: int) -> None:
        """
        Two-phase close — step 2 (failure path).

        Roll PENDING_CLOSE back to EXECUTED so the monitor retries next cycle.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status = 'EXECUTED',
                       pnl_source = NULL,
                       pnl_verified = 0,
                       status_updated_at = ?
                 WHERE id = ?
                   AND status = 'PENDING_CLOSE'
                """,
                (datetime.datetime.now().isoformat(), trade_id),
            )
            conn.commit()

    def close_position(
        self,
        trade_id: int,
        pnl: float,
        close_order_id: str = None,
        *,
        pnl_source: str = None,
        pnl_verified: bool = None,
    ):
        """Mark a trade CLOSED with realised P&L and optional close-order reference.

        Prefer the two-phase ``mark_pending_close`` / ``confirm_close`` pair for
        live broker submissions.  This single-step method is retained for the
        expiry-sweep path and tests that do not submit broker orders.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status   = 'CLOSED',
                       pnl      = ?,
                       order_id = COALESCE(?, order_id),
                       close_order_id = COALESCE(?, close_order_id),
                       pnl_source = COALESCE(?, pnl_source),
                       pnl_verified = COALESCE(?, pnl_verified),
                       status_updated_at = ?
                 WHERE id = ?
                """,
                (
                    pnl, close_order_id, close_order_id,
                    pnl_source,
                    None if pnl_verified is None else (1 if pnl_verified else 0),
                    datetime.datetime.now().isoformat(), trade_id,
                ),
            )
            conn.commit()
        if close_order_id:
            role = 'EXPIRE' if str(close_order_id).startswith('EXPIRED') else 'CLOSE'
            synthetic_id = (
                close_order_id
                if close_order_id not in {
                    'DRY_RUN_CLOSE', 'EXPIRED', 'EXPIRED_RECONCILED',
                    'EXTERNALLY_CLOSED', 'RECONCILED',
                }
                else f"{close_order_id}:{trade_id}"
            )
            self.upsert_trade_order(trade_id, synthetic_id, role=role, status='synthetic')

    def get_pending_close_positions(self) -> list:
        """
        Return positions stuck in PENDING_CLOSE.

        These are positions where a close order was submitted but the process
        died (or threw) before the DB could be updated to CLOSED.  The caller
        should reconcile them against the broker and either call
        ``confirm_close`` or ``reopen_position`` as appropriate.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM trades WHERE status = 'PENDING_CLOSE' ORDER BY timestamp ASC"
            )
            rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_history(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_expired_unsettled_positions(self) -> list:
        """
        Return positions that have passed their expiry date but were never
        explicitly closed (status is still EXECUTED or DRY_RUN).

        These are options that expired naturally — either worthless (full
        premium kept) or in-the-money (partial/full loss).  The expiry sweep
        in PositionMonitor.settle_expired() uses this to record final P&L.
        """
        today = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                 WHERE status IN ('EXECUTED', 'DRY_RUN')
                   AND expiry < ?
                 ORDER BY expiry ASC
                """,
                (today,),
            )
            rows = [dict(row) for row in cursor.fetchall()]

        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    def get_pending_open_positions(self) -> list:
        """Return positions with status PENDING (open order submitted, fill unconfirmed)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(
                "SELECT * FROM trades WHERE status = 'PENDING' ORDER BY timestamp ASC"
            )]
        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    def get_open_positions(self) -> list:
        """
        Return all positions still open (status EXECUTED or DRY_RUN) whose
        expiry has not yet passed.  The 'legs' field is returned as a dict.
        """
        today = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                 WHERE status IN ('EXECUTED', 'DRY_RUN')
                   AND expiry >= ?
                 ORDER BY timestamp ASC
                """,
                (today,),
            )
            rows = [dict(row) for row in cursor.fetchall()]

        # Deserialise legs JSON for callers
        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    def get_all_executed_positions(self) -> list:
        """
        Return ALL EXECUTED positions regardless of expiry date.

        Used by the reconciler to detect ghost positions whose expiry has already
        passed but were never settled in the DB (e.g. monitor was offline at expiry).
        The 'legs' field is returned as a dict.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                 WHERE status = 'EXECUTED'
                 ORDER BY timestamp ASC
                """
            )
            rows = [dict(row) for row in cursor.fetchall()]

        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    def get_false_ghost_positions(self) -> list:
        """
        Return CLOSED positions that were closed by the reconciler's ghost
        detection (order_id = 'EXTERNALLY_CLOSED') and whose expiry is still
        in the future.

        These are candidates for recovery if their legs reappear in Alpaca —
        they were likely closed due to a race condition between the agent
        writing the DB record and the reconciler's next cycle.
        The 'legs' field is returned as a dict.
        """
        today = datetime.date.today().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trades
                 WHERE status   = 'CLOSED'
                   AND order_id = 'EXTERNALLY_CLOSED'
                   AND expiry   >= ?
                 ORDER BY timestamp ASC
                """,
                (today,),
            )
            rows = [dict(row) for row in cursor.fetchall()]

        for row in rows:
            raw = row.get('legs')
            if raw and isinstance(raw, str):
                try:
                    row['legs'] = json.loads(raw)
                except Exception:
                    row['legs'] = {}
        return rows

    def reopen_false_ghost(self, trade_id: int) -> None:
        """
        Reopen a position that was incorrectly closed as EXTERNALLY_CLOSED by
        the ghost-detection race condition.

        Transitions CLOSED (order_id='EXTERNALLY_CLOSED') → EXECUTED and
        clears the pnl placeholder so the monitor can resume tracking it.
        Only acts on rows that were closed via the ghost-detection path to
        avoid accidentally reopening legitimately closed positions.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE trades
                   SET status   = 'EXECUTED',
                       pnl      = NULL,
                       order_id = NULL,
                       pnl_source = NULL,
                       pnl_verified = 0
                 WHERE id       = ?
                   AND status   = 'CLOSED'
                   AND order_id = 'EXTERNALLY_CLOSED'
                """,
                (trade_id,),
            )
            conn.commit()
