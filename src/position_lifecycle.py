"""
Position lifecycle service.

Centralises live close state transitions so monitor, CLI, and dashboard do not
race each other or bypass PENDING_CLOSE.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor

_log = logging.getLogger('optionwheel')


@dataclass
class CloseResult:
    success: bool
    submitted: bool
    already_pending: bool
    order_id: Optional[str]
    pnl: float
    status: str
    error: Optional[str] = None


class PositionLifecycleService:
    """Atomic position lifecycle operations shared by every caller."""

    def __init__(self, db: TradeDatabase, executor: AlpacaExecutor):
        self.db = db
        self.executor = executor

    def close_position(
        self,
        pos: dict,
        *,
        limit_price: float | None,
        pnl: float,
        dry_run: bool,
        reason: str,
    ) -> CloseResult:
        trade_id = int(pos['id'])

        if dry_run or pos.get('status') == 'DRY_RUN':
            self.db.close_position(
                trade_id,
                pnl,
                'DRY_RUN_CLOSE',
                pnl_source='DRY_RUN_ESTIMATE',
                pnl_verified=False,
            )
            return CloseResult(
                success=True,
                submitted=False,
                already_pending=False,
                order_id='DRY_RUN_CLOSE',
                pnl=pnl,
                status='CLOSED',
            )

        if not self.db.claim_pending_close(trade_id, pnl=pnl, reason=reason):
            current = self.db.get_trade(trade_id) or {}
            status = current.get('status', 'UNKNOWN')
            order_id = current.get('close_order_id') or current.get('order_id')
            if status == 'PENDING_CLOSE':
                _log.info(
                    "[lifecycle] Close already pending for id=%s %s %s "
                    "(order=%s); not submitting another order.",
                    trade_id, pos.get('type'), pos.get('symbol'), order_id,
                )
                return CloseResult(
                    success=True,
                    submitted=False,
                    already_pending=True,
                    order_id=order_id,
                    pnl=float(current.get('pnl') or pnl or 0.0),
                    status=status,
                )
            return CloseResult(
                success=False,
                submitted=False,
                already_pending=False,
                order_id=order_id,
                pnl=pnl,
                status=status,
                error=f"Position is {status}, not EXECUTED",
            )

        try:
            order_id = self.executor.execute_close_position(
                pos,
                limit_price=limit_price,
                dry_run=False,
                amount=int(pos.get('contracts') or 1),
            )
        except Exception as exc:
            self.db.reopen_position(trade_id)
            _log.error(
                "[lifecycle] Close order error for id=%s %s %s: %s",
                trade_id, pos.get('type'), pos.get('symbol'), exc, exc_info=True,
            )
            return CloseResult(
                success=False,
                submitted=False,
                already_pending=False,
                order_id=None,
                pnl=pnl,
                status='EXECUTED',
                error=str(exc),
            )

        if not order_id:
            self.db.reopen_position(trade_id)
            return CloseResult(
                success=False,
                submitted=False,
                already_pending=False,
                order_id=None,
                pnl=pnl,
                status='EXECUTED',
                error='Broker close order was not accepted',
            )

        self.db.record_pending_close_order(
            trade_id,
            pnl=pnl,
            close_order_id=order_id,
            reason=reason,
        )
        return CloseResult(
            success=True,
            submitted=True,
            already_pending=False,
            order_id=order_id,
            pnl=pnl,
            status='PENDING_CLOSE',
        )
