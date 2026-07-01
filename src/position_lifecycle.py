"""
Position lifecycle service.

Centralises live close state transitions so monitor, CLI, and dashboard do not
race each other or bypass PENDING_CLOSE.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass
from typing import Optional

from src.database import TradeDatabase
from src.executor import AlpacaExecutor

_log = logging.getLogger('optionwheel')

_DEFAULT_CLOSE_EXECUTION_CFG = {
    'limit_buffer_pct': 0.05,
    'limit_buffer_dollars': 0.05,
    'retry_limit_increment_pct': 0.05,
    'retry_limit_increment_dollars': 0.05,
    'market_after_same_day_attempts': 3,
}


@dataclass
class CloseResult:
    success: bool
    submitted: bool
    already_pending: bool
    order_id: Optional[str]
    pnl: float
    status: str
    error: Optional[str] = None


@dataclass(frozen=True)
class CloseOrderPlan:
    order_type: str
    limit_price: Optional[float]
    same_day_attempts: int


class PositionLifecycleService:
    """Atomic position lifecycle operations shared by every caller."""

    def __init__(self, db: TradeDatabase, executor: AlpacaExecutor):
        self.db = db
        self.executor = executor

    def _close_execution_cfg(self) -> dict:
        config = getattr(self.executor, 'config', None)
        if not isinstance(config, dict):
            return dict(_DEFAULT_CLOSE_EXECUTION_CFG)
        risk = config.get('risk_parameters', {})
        close_cfg = risk.get('close_execution', {})
        if not isinstance(close_cfg, dict):
            close_cfg = {}
        return {**_DEFAULT_CLOSE_EXECUTION_CFG, **close_cfg}

    @staticmethod
    def _timestamp_to_date(value) -> _dt.date | None:
        if not value:
            return None
        try:
            return _dt.datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    def _same_day_close_attempts(self, trade_id: int) -> int:
        today = _dt.date.today()
        attempts = 0
        for order in self.db.get_trade_orders(trade_id):
            if str(order.get('role') or '').upper() != 'CLOSE':
                continue
            for candidate in (
                order.get('submitted_at'),
                order.get('updated_at'),
                order.get('filled_at'),
            ):
                if self._timestamp_to_date(candidate) == today:
                    attempts += 1
                    break
        return attempts

    @staticmethod
    def _spread_width(pos: dict) -> float | None:
        legs = pos.get('legs') or {}
        strat = str(pos.get('type') or '').upper()
        try:
            if strat in {'PCS', 'CCS'}:
                short = float(
                    legs.get('short_strike')
                    or legs.get('short_put')
                    or legs.get('short_call')
                    or pos.get('strike')
                    or 0.0
                )
                long = float(
                    legs.get('long_strike')
                    or legs.get('long_put')
                    or legs.get('long_call')
                    or 0.0
                )
                return abs(short - long) if short > 0 and long > 0 else None
            if strat in {'IC', 'IFLY'}:
                short_put = float(legs.get('short_put') or 0.0)
                long_put = float(legs.get('long_put') or 0.0)
                short_call = float(legs.get('short_call') or 0.0)
                long_call = float(legs.get('long_call') or 0.0)
                put_width = abs(short_put - long_put) if short_put > 0 and long_put > 0 else 0.0
                call_width = abs(short_call - long_call) if short_call > 0 and long_call > 0 else 0.0
                width = max(put_width, call_width)
                return width if width > 0 else None
        except (TypeError, ValueError):
            return None
        return None

    def _marketable_limit_price(
        self,
        pos: dict,
        reference_price: float | None,
        *,
        same_day_attempts: int,
    ) -> float | None:
        if reference_price is None:
            return None
        try:
            ref = float(reference_price)
        except (TypeError, ValueError):
            return None
        cfg = self._close_execution_cfg()
        base_buffer = max(
            abs(ref) * float(cfg.get('limit_buffer_pct', 0.0) or 0.0),
            float(cfg.get('limit_buffer_dollars', 0.0) or 0.0),
        )
        retry_buffer = max(
            abs(ref) * float(cfg.get('retry_limit_increment_pct', 0.0) or 0.0),
            float(cfg.get('retry_limit_increment_dollars', 0.0) or 0.0),
        )
        limit_price = ref + base_buffer + (retry_buffer * same_day_attempts)
        width = self._spread_width(pos)
        if width is not None and width > 0:
            limit_price = min(limit_price, width)
        return max(0.01, round(limit_price, 2))

    def close_order_plan(
        self,
        pos: dict,
        *,
        reference_price: float | None,
    ) -> CloseOrderPlan:
        trade_id = int(pos['id'])
        same_day_attempts = self._same_day_close_attempts(trade_id)
        cfg = self._close_execution_cfg()
        market_after = max(1, int(cfg.get('market_after_same_day_attempts', 3) or 3))
        if same_day_attempts + 1 >= market_after:
            return CloseOrderPlan(
                order_type='market',
                limit_price=None,
                same_day_attempts=same_day_attempts,
            )
        limit_price = self._marketable_limit_price(
            pos,
            reference_price,
            same_day_attempts=same_day_attempts,
        )
        if limit_price is None:
            return CloseOrderPlan(
                order_type='market',
                limit_price=None,
                same_day_attempts=same_day_attempts,
            )
        return CloseOrderPlan(
            order_type='limit',
            limit_price=limit_price,
            same_day_attempts=same_day_attempts,
        )

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

        plan = self.close_order_plan(pos, reference_price=limit_price)

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
                limit_price=plan.limit_price,
                order_type=plan.order_type,
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

        _log.info(
            "[lifecycle] Close submission plan for id=%s %s %s: attempt=%d type=%s limit=%s",
            trade_id,
            pos.get('type'),
            pos.get('symbol'),
            plan.same_day_attempts + 1,
            plan.order_type,
            f"{plan.limit_price:.2f}" if plan.limit_price is not None else "market",
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
