"""
src.notify
==========

Email notification sub-package.

``sender.py``    — :class:`EmailNotifier` (SMTP + IMAP)
``formatter.py`` — all rendering / formatting helpers
"""
from .sender import EmailNotifier
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
    'EmailNotifier',
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
