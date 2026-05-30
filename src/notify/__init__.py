"""
src.notify
==========

Email notification sub-package.

``formatter.py`` — all rendering / formatting helpers

The :class:`EmailNotifier` class lives in ``src/notifier.py`` (the
top-level module); it imports rendering helpers from ``formatter.py``.
"""
from .formatter import (
    _build_mime,
    _extract_plain_body,
    _extract_token_from_msgid,
    _parse_approval_body,
    _legs_str,
    _pos_legs_str,
    _fmt_opt,
    _render_trade_executed_text,
    _render_trade_executed_html,
    _render_position_closed_text,
    _render_position_closed_html,
    _render_trade_plan_text,
    _render_trade_plan_html,
    _render_daily_risk_text,
    _render_daily_risk_html,
    _CSS_BASE,
    _RISK_STYLE,
)

__all__ = [
    '_build_mime',
    '_extract_plain_body',
    '_extract_token_from_msgid',
    '_parse_approval_body',
    '_legs_str',
    '_pos_legs_str',
    '_fmt_opt',
    '_render_trade_executed_text',
    '_render_trade_executed_html',
    '_render_position_closed_text',
    '_render_position_closed_html',
    '_render_trade_plan_text',
    '_render_trade_plan_html',
    '_render_daily_risk_text',
    '_render_daily_risk_html',
    '_CSS_BASE',
    '_RISK_STYLE',
]
