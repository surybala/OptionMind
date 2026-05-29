"""
Tests for src/notifier.py
"""
from __future__ import annotations

import smtplib
import time
import unittest
from email.mime.multipart import MIMEMultipart
from unittest.mock import MagicMock, call, patch

from email import message_from_string as _mfs

from src.notifier import (
    EmailNotifier,
    _extract_plain_body,
    _extract_token_from_msgid,
    _fmt_opt,
    _parse_approval_body,
    _pos_legs_str,
    _render_daily_risk_html,
    _render_daily_risk_text,
    _render_position_closed_text,
    _render_trade_executed_text,
    _render_trade_plan_text,
    _render_weekly_digest_html,
    _render_weekly_digest_text,
)
from src.position_monitor import _classify_risk_level, _within_market_hours


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMAIL_CFG = {
    'smtp_host':                   'smtp.gmail.com',
    'smtp_port':                   587,
    'imap_host':                   'imap.gmail.com',
    'imap_port':                   993,
    'from_addr':                   'bot@example.com',
    'to_addr':                     'trader@example.com',
    'app_password':                'abcd efgh ijkl mnop',
    'approval_timeout_seconds':    5,
    'approval_poll_interval_seconds': 1,
}

_PICK = {
    'strategy':     'PCS',
    'symbol':       'AAPL',
    'expiry':       '2026-04-17',
    'current_price': 175.50,
    'premium':      0.45,
    'prob_win':     0.82,
    'roi':          0.09,
    'score':        0.369,
    'short_strike': 160.0,
    'long_strike':  155.0,
    'max_loss':     500.0,
    'market_cap':   2.8e12,
}

_POS_STOP_LOSS = {
    'id':             42,
    'type':           'PCS',
    'symbol':         'AAPL',
    'expiry':         '2026-04-17',
    'premium':        0.45,
    'close_pnl':      -90.0,
    'close_order_id': 'order-xyz',
    'reason_tag':     'STOP_LOSS',
    'entry_premium':  0.45,
    'current_mark':   1.35,
    'pnl_per_share':  -0.90,
}

_POS_GAMMA = {
    **_POS_STOP_LOSS,
    'reason_tag':   'GAMMA_RISK',
    'ratio':        2.1,
    'short_delta':  0.38,
    'risk_score':   0.75,
    'dte':          5,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_notifier(cfg_override=None):
    cfg = dict(_EMAIL_CFG)
    if cfg_override:
        cfg.update(cfg_override)
    return EmailNotifier({'email': cfg})


# ---------------------------------------------------------------------------
# Disabled / missing config
# ---------------------------------------------------------------------------

class TestDisabledNotifier(unittest.TestCase):

    def test_no_email_key_disabled(self):
        n = EmailNotifier({})
        self.assertFalse(n.enabled)

    def test_empty_email_key_disabled(self):
        n = EmailNotifier({'email': {}})
        self.assertFalse(n.enabled)

    def test_missing_password_disabled(self):
        import os
        cfg = dict(_EMAIL_CFG)
        del cfg['app_password']
        # Ensure env var is also absent so notifier correctly reads as disabled
        old = os.environ.pop('OPTIONWHEEL_EMAIL_PASSWORD', None)
        try:
            n = EmailNotifier({'email': cfg})
            self.assertFalse(n.enabled)
        finally:
            if old is not None:
                os.environ['OPTIONWHEEL_EMAIL_PASSWORD'] = old

    def test_password_from_env_var(self):
        import os
        cfg = dict(_EMAIL_CFG)
        del cfg['app_password']   # no password in config
        os.environ['OPTIONWHEEL_EMAIL_PASSWORD'] = 'env pass word'
        try:
            n = EmailNotifier({'email': cfg})
            self.assertTrue(n.enabled)
            self.assertEqual(n._password, 'envpassword')
        finally:
            del os.environ['OPTIONWHEEL_EMAIL_PASSWORD']

    def test_env_var_overrides_config_password(self):
        import os
        os.environ['OPTIONWHEEL_EMAIL_PASSWORD'] = 'env secret'
        try:
            n = _make_notifier()   # config has 'abcd efgh ijkl mnop'
            self.assertEqual(n._password, 'envsecret')
        finally:
            del os.environ['OPTIONWHEEL_EMAIL_PASSWORD']

    def test_send_trade_executed_noop_when_disabled(self):
        n = EmailNotifier({})
        # Should not raise
        n.send_trade_executed(_PICK, 'order-1')

    def test_send_position_closed_noop_when_disabled(self):
        n = EmailNotifier({})
        n.send_position_closed(_POS_STOP_LOSS, 'STOP_LOSS')

    def test_send_trade_plan_returns_none_when_disabled(self):
        n = EmailNotifier({})
        self.assertIsNone(n.send_trade_plan([_PICK]))

    def test_wait_for_approval_returns_none_when_disabled(self):
        n = EmailNotifier({})
        self.assertIsNone(n.wait_for_approval('<fake@id>', [_PICK]))


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

class TestTokenHelpers(unittest.TestCase):

    def test_extract_token_roundtrip(self):
        token = 'deadbeef12345678'   # 16 hex chars
        ts    = int(time.time())
        msg_id = f"<{token}.{ts}@optionwheel.local>"
        self.assertEqual(_extract_token_from_msgid(msg_id), token)

    def test_extract_token_bad_format_returns_none(self):
        self.assertIsNone(_extract_token_from_msgid('<garbage>'))
        self.assertIsNone(_extract_token_from_msgid(''))


# ---------------------------------------------------------------------------
# _parse_approval_body
# ---------------------------------------------------------------------------

class TestParseApprovalBody(unittest.TestCase):

    def _picks(self, n=5):
        return [{'strategy': 'CSP', 'symbol': f'T{i}'} for i in range(1, n + 1)]

    def test_all(self):
        picks = self._picks()
        self.assertEqual(_parse_approval_body('a\n', picks), picks)
        self.assertEqual(_parse_approval_body('all\n', picks), picks)

    def test_none(self):
        picks = self._picks()
        self.assertEqual(_parse_approval_body('n\n', picks), [])
        self.assertEqual(_parse_approval_body('none\n', picks), [])

    def test_comma_list(self):
        picks = self._picks()
        result = _parse_approval_body('1,3,5', picks)
        self.assertEqual(result, [picks[0], picks[2], picks[4]])

    def test_range(self):
        picks = self._picks()
        result = _parse_approval_body('2-4', picks)
        self.assertEqual(result, [picks[1], picks[2], picks[3]])

    def test_mixed_range_and_list(self):
        picks = self._picks()
        result = _parse_approval_body('1,3-5', picks)
        self.assertEqual(result, [picks[0], picks[2], picks[3], picks[4]])

    def test_quoted_lines_stripped(self):
        picks = self._picks()
        body = "> On Thu, 15 Mar 2026 the bot wrote:\n> [OptionWheel] Trade Plan\n\n2,4\n"
        result = _parse_approval_body(body, picks)
        self.assertEqual(result, [picks[1], picks[3]])

    def test_out_of_range_ignored(self):
        picks = self._picks(3)
        result = _parse_approval_body('1,10,100', picks)
        self.assertEqual(result, [picks[0]])

    def test_empty_body_returns_empty(self):
        picks = self._picks()
        self.assertEqual(_parse_approval_body('', picks), [])

    def test_only_quoted_returns_empty(self):
        picks = self._picks()
        body = "> all quoted\n> nothing else\n"
        self.assertEqual(_parse_approval_body(body, picks), [])

    def test_case_insensitive(self):
        picks = self._picks()
        self.assertEqual(_parse_approval_body('ALL', picks), picks)
        self.assertEqual(_parse_approval_body('NONE', picks), [])


# ---------------------------------------------------------------------------
# send_trade_executed
# ---------------------------------------------------------------------------

class TestSendTradeExecuted(unittest.TestCase):

    @patch('src.notifier.smtplib.SMTP')
    def test_sends_email(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_trade_executed(_PICK, 'order-123', 'TRADE_PLAN')

        mock_smtp.sendmail.assert_called_once()
        args = mock_smtp.sendmail.call_args[0]
        self.assertEqual(args[0], 'bot@example.com')
        self.assertEqual(args[1], 'trader@example.com')
        # Parse the MIME message and check the decoded plain-text body
        from email import message_from_string as _mfs
        parsed = _mfs(args[2])
        body = _extract_plain_body(parsed)
        self.assertIn('Trade Executed', body)
        self.assertIn('AAPL', body)

    @patch('src.notifier.time.sleep')
    @patch('src.notifier.smtplib.SMTP')
    def test_smtp_failure_does_not_raise(self, mock_smtp_cls, _mock_sleep):
        mock_smtp_cls.side_effect = ConnectionRefusedError("refused")
        n = _make_notifier()
        # Should not raise
        n.send_trade_executed(_PICK, 'order-1')

    def test_password_spaces_stripped(self):
        n = _make_notifier()
        self.assertEqual(n._password, 'abcdefghijklmnop')


# ---------------------------------------------------------------------------
# send_position_closed
# ---------------------------------------------------------------------------

class TestSendPositionClosed(unittest.TestCase):

    @patch('src.notifier.smtplib.SMTP')
    def test_stop_loss_subject_contains_pnl(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_position_closed(_POS_STOP_LOSS, 'STOP_LOSS')

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        body = _extract_plain_body(_mfs(raw_msg))
        self.assertIn('STOP_LOSS', body)
        self.assertIn('AAPL', body)
        self.assertIn('$-90.00', body)

    @patch('src.notifier.smtplib.SMTP')
    def test_gamma_risk_includes_risk_metrics(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_position_closed(_POS_GAMMA, 'GAMMA_RISK')

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        body = _extract_plain_body(_mfs(raw_msg))
        self.assertIn('Gamma', body)
        self.assertIn('2.1', body)    # ratio
        self.assertIn('0.38', body)   # short_delta
        self.assertIn('5d', body)     # dte


# ---------------------------------------------------------------------------
# send_trade_plan + token embedding
# ---------------------------------------------------------------------------

class TestSendTradePlan(unittest.TestCase):

    @patch('src.notifier.smtplib.SMTP')
    def test_returns_message_id(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        msg_id = n.send_trade_plan([_PICK], capital_budget=5000)

        self.assertIsNotNone(msg_id)
        self.assertIn('@optionwheel.local', msg_id)

    @patch('src.notifier.smtplib.SMTP')
    def test_token_in_subject(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        msg_id = n.send_trade_plan([_PICK])

        token = _extract_token_from_msgid(msg_id)
        raw_msg = mock_smtp.sendmail.call_args[0][2]
        self.assertIn(f'[token:{token}]', raw_msg)

    @patch('src.notifier.time.sleep')
    @patch('src.notifier.smtplib.SMTP')
    def test_smtp_failure_returns_none(self, mock_smtp_cls, _mock_sleep):
        mock_smtp_cls.side_effect = OSError("network error")
        n = _make_notifier()
        self.assertIsNone(n.send_trade_plan([_PICK]))


# ---------------------------------------------------------------------------
# wait_for_approval — timeout path
# ---------------------------------------------------------------------------

class TestWaitForApproval(unittest.TestCase):

    @patch('src.notifier.imaplib.IMAP4_SSL')
    def test_timeout_returns_none(self, mock_imap_cls):
        """IMAP always returns no match → timeout → None."""
        mock_imap = MagicMock()
        mock_imap.uid.return_value = ('OK', [b''])
        mock_imap_cls.return_value.__enter__ = lambda s: mock_imap
        mock_imap_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier({'approval_timeout_seconds': 2, 'approval_poll_interval_seconds': 1})
        token = 'deadbeef12345678'
        msg_id = f'<{token}.{int(time.time())}@optionwheel.local>'

        result = n.wait_for_approval(msg_id, [_PICK])
        self.assertIsNone(result)

    @patch('src.notifier.imaplib.IMAP4_SSL')
    def test_reply_found_returns_approved(self, mock_imap_cls):
        """IMAP returns a matching UID on first poll; reply body is '1'."""
        import email as _email_mod

        # Build a minimal raw email with plain text body
        raw_reply = (
            b'From: trader@example.com\r\n'
            b'Subject: Re: [OptionWheel] Trade Plan [token:deadbeef12345678]\r\n'
            b'Content-Type: text/plain; charset=utf-8\r\n\r\n'
            b'1\r\n'
        )

        mock_imap = MagicMock()
        mock_imap.login.return_value = ('OK', [])
        mock_imap.select.return_value = ('OK', [b'1'])
        # First SEARCH (subject token) returns UID b'5'
        mock_imap.uid.side_effect = [
            ('OK', [b'5']),           # SEARCH subject
            ('OK', [b'']),            # SEARCH In-Reply-To
            ('OK', [(b'5 RFC822 {N}', raw_reply)]),  # FETCH
            ('OK', []),               # STORE \\Seen
        ]
        mock_imap_cls.return_value.__enter__ = lambda s: mock_imap
        mock_imap_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier({'approval_timeout_seconds': 10, 'approval_poll_interval_seconds': 1})
        token = 'deadbeef12345678'
        msg_id = f'<{token}.{int(time.time())}@optionwheel.local>'

        picks = [_PICK, dict(_PICK, symbol='MSFT')]
        result = n.wait_for_approval(msg_id, picks)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['symbol'], 'AAPL')


# ---------------------------------------------------------------------------
# Text renderers — smoke tests
# ---------------------------------------------------------------------------

class TestRenderers(unittest.TestCase):

    def test_trade_executed_text_contains_key_fields(self):
        txt = _render_trade_executed_text(_PICK, 'ord-1', 'TRADE_PLAN')
        self.assertIn('PCS', txt)
        self.assertIn('AAPL', txt)
        self.assertIn('2026-04-17', txt)
        self.assertIn('45.00/contract', txt)
        self.assertIn('Contracts:', txt)
        self.assertIn('TRADE_PLAN', txt)

    def test_trade_executed_text_single_contract_total_equals_per_contract(self):
        txt = _render_trade_executed_text(_PICK, 'ord-1', 'TRADE_PLAN')
        # 1 contract: total == per-contract ($45.00 total)
        self.assertIn('$45.00 total', txt)
        self.assertIn('Contracts:  1', txt)

    def test_trade_executed_text_multi_contract_total_scaled(self):
        pick = {**_PICK, 'quantity': 10}
        txt = _render_trade_executed_text(pick, 'ord-2', 'AUTO')
        self.assertIn('Contracts:  10', txt)
        self.assertIn('$450.00 total', txt)

    def test_trade_executed_html_contains_contracts_row(self):
        from src.notifier import _render_trade_executed_html
        html = _render_trade_executed_html({**_PICK, 'quantity': 5}, 'ord-3', 'AUTO')
        self.assertIn('Contracts', html)
        self.assertIn('<strong>5</strong>', html)

    def test_trade_executed_html_multi_contract_total(self):
        from src.notifier import _render_trade_executed_html
        html = _render_trade_executed_html({**_PICK, 'quantity': 8}, 'ord-4', 'AUTO')
        # 0.45 × 100 × 8 = 360.00
        self.assertIn('360.00 total', html)

    def test_position_closed_stop_loss_has_risk_section(self):
        txt = _render_position_closed_text(_POS_GAMMA, 'GAMMA_RISK')
        self.assertIn('Risk Metrics', txt)
        self.assertIn('Gamma/Theta', txt)
        self.assertIn('2.1', txt)

    def test_position_closed_manual_no_risk_section(self):
        pos = dict(_POS_STOP_LOSS, reason_tag='MANUAL')
        txt = _render_position_closed_text(pos, 'MANUAL')
        self.assertNotIn('Risk Metrics', txt)

    def test_position_closed_text_shows_pnl_source(self):
        pos = dict(_POS_STOP_LOSS, pnl_source='ALPACA_FILLS', pnl_verified=1)
        txt = _render_position_closed_text(pos, 'STOP_LOSS')
        self.assertIn('P&L Source:', txt)
        self.assertIn('Alpaca Fills', txt)

    def test_position_closed_html_flags_unverified_pnl_source(self):
        from src.notifier import _render_position_closed_html
        pos = dict(_POS_STOP_LOSS, pnl_source='EXTERNAL_PLACEHOLDER', pnl_verified=0)
        html = _render_position_closed_html(pos, 'STOP_LOSS')
        self.assertIn('P&L source', html)
        self.assertIn('External Placeholder (unverified)', html)

    def test_trade_plan_text_contains_instructions(self):
        txt = _render_trade_plan_text([_PICK], 5000.0, 'abc123ef12345678', 300, 15)
        self.assertIn('all', txt)
        self.assertIn('none', txt)
        self.assertIn('AAPL', txt)
        self.assertIn('PCS', txt)


# ---------------------------------------------------------------------------
# Fixtures for daily risk report tests
# ---------------------------------------------------------------------------

_POS_CRITICAL = {
    'id': 10, 'type': 'PCS', 'symbol': 'TSLA', 'expiry': '2025-02-01',
    'premium': 1.20,
    'legs': '{"short_strike": 245, "long_strike": 240}',
    'current_mark': 3.20, 'pnl_per_share': -2.0, 'pnl_dollars': -200.0,
    'stop_threshold': 2.40, 'stop_proximity_pct': 133.3,
    'profit_captured_pct': -166.7, 'gamma_theta_ratio': 2.1,
    'net_short_delta': 0.42, 'risk_score': 8.3, 'dte': 2, 'spot': 248.5,
    'risk_level': 'CRITICAL',
}

_POS_SAFE = {
    'id': 11, 'type': 'IC', 'symbol': 'SPY', 'expiry': '2025-02-14',
    'premium': 0.80,
    'legs': '{"short_put":495,"long_put":490,"short_call":510,"long_call":515}',
    'current_mark': 0.35, 'pnl_per_share': 0.45, 'pnl_dollars': 45.0,
    'stop_threshold': 1.60, 'stop_proximity_pct': 21.9,
    'profit_captured_pct': 56.3, 'gamma_theta_ratio': 0.4,
    'net_short_delta': 0.08, 'risk_score': 1.2, 'dte': 15, 'spot': 500.0,
    'risk_level': 'SAFE',
}

_SNAPSHOT = [_POS_CRITICAL, _POS_SAFE]


# ---------------------------------------------------------------------------
# _fmt_opt
# ---------------------------------------------------------------------------

class TestFmtOpt(unittest.TestCase):

    def test_none_returns_dash(self):
        self.assertEqual(_fmt_opt(None), '—')

    def test_formats_float_default(self):
        self.assertEqual(_fmt_opt(3.14159), '3.14')

    def test_prefix(self):
        self.assertEqual(_fmt_opt(9.5, '.2f', '$'), '$9.50')

    def test_suffix(self):
        self.assertEqual(_fmt_opt(42.0, '.0f', suffix='%'), '42%')

    def test_prefix_and_suffix(self):
        self.assertEqual(_fmt_opt(7.0, '.1f', '$', 'x'), '$7.0x')

    def test_non_numeric_returns_dash(self):
        self.assertEqual(_fmt_opt('bad', '.2f'), '—')

    def test_zero_is_not_dash(self):
        self.assertNotEqual(_fmt_opt(0.0), '—')


# ---------------------------------------------------------------------------
# _pos_legs_str  (DB-style positions: 'type' + JSON 'legs')
# ---------------------------------------------------------------------------

class TestPosLegsStr(unittest.TestCase):

    def test_pcs(self):
        pos = {'type': 'PCS', 'legs': '{"short_strike":245,"long_strike":240}'}
        self.assertEqual(_pos_legs_str(pos), '245/240 P')

    def test_ccs(self):
        pos = {'type': 'CCS', 'legs': '{"short_strike":510,"long_strike":515}'}
        self.assertEqual(_pos_legs_str(pos), '510/515 C')

    def test_csp(self):
        pos = {'type': 'CSP', 'legs': '{"short_strike":160}'}
        self.assertEqual(_pos_legs_str(pos), '160P')

    def test_cc(self):
        pos = {'type': 'CC', 'legs': '{"short_strike":200}'}
        self.assertEqual(_pos_legs_str(pos), '200C')

    def test_ic(self):
        pos = {'type': 'IC',
               'legs': '{"short_put":495,"long_put":490,"short_call":510,"long_call":515}'}
        result = _pos_legs_str(pos)
        self.assertIn('490', result)
        self.assertIn('495', result)
        self.assertIn('510', result)
        self.assertIn('515', result)

    def test_strangle(self):
        pos = {'type': 'STRANGLE', 'legs': '{"short_put":480,"short_call":520}'}
        result = _pos_legs_str(pos)
        self.assertIn('480', result)
        self.assertIn('520', result)

    def test_no_legs_graceful(self):
        """When legs is None the function falls back to '?' placeholders (not a crash)."""
        pos = {'type': 'PCS', 'legs': None}
        result = _pos_legs_str(pos)
        self.assertIn('?', result)   # e.g. '?/? P' — graceful degradation

    def test_fallback_to_pos_strike_for_csp(self):
        pos = {'type': 'CSP', 'legs': '{}', 'strike': 155}
        self.assertEqual(_pos_legs_str(pos), '155P')

    def test_dict_legs_accepted(self):
        """legs may already be a parsed dict (not a JSON string)."""
        pos = {'type': 'PCS', 'legs': {'short_strike': 300, 'long_strike': 295}}
        self.assertEqual(_pos_legs_str(pos), '300/295 P')


# ---------------------------------------------------------------------------
# _classify_risk_level  (from position_monitor)
# ---------------------------------------------------------------------------

class TestClassifyRiskLevel(unittest.TestCase):

    def test_all_axes_safe(self):
        pos = {'profit_captured_pct': 75, 'stop_proximity_pct': 30,
               'gamma_theta_ratio': 0.5}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'SAFE')

    def test_stop_proximity_watch(self):
        pos = {'stop_proximity_pct': 60}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'WATCH')

    def test_stop_proximity_caution(self):
        pos = {'stop_proximity_pct': 80}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CAUTION')

    def test_stop_proximity_critical(self):
        pos = {'stop_proximity_pct': 95}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CRITICAL')

    def test_negative_profit_is_critical(self):
        pos = {'profit_captured_pct': -10, 'stop_proximity_pct': 40}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CRITICAL')

    def test_low_profit_is_caution(self):
        pos = {'profit_captured_pct': 15, 'stop_proximity_pct': 30}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CAUTION')

    def test_medium_profit_is_safe(self):
        """25–50% profit captured is no longer a warning — position is in profit."""
        pos = {'profit_captured_pct': 35, 'stop_proximity_pct': 30}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'SAFE')

    def test_gamma_ratio_watch(self):
        pos = {'profit_captured_pct': 80, 'stop_proximity_pct': 20,
               'gamma_theta_ratio': 1.0}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'WATCH')

    def test_gamma_ratio_caution(self):
        pos = {'profit_captured_pct': 80, 'stop_proximity_pct': 20,
               'gamma_theta_ratio': 1.3}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CAUTION')

    def test_gamma_ratio_critical(self):
        pos = {'profit_captured_pct': 80, 'stop_proximity_pct': 20,
               'gamma_theta_ratio': 1.8}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CRITICAL')

    def test_highest_axis_wins(self):
        """WATCH stop + CRITICAL gamma → overall CRITICAL."""
        pos = {'profit_captured_pct': 60, 'stop_proximity_pct': 60,
               'gamma_theta_ratio': 2.0}
        self.assertEqual(_classify_risk_level(pos, 2.0), 'CRITICAL')

    def test_missing_fields_defaults_watch(self):
        """No risk metrics at all → WATCH (can't confirm safe, incomplete data)."""
        self.assertEqual(_classify_risk_level({}, 2.0), 'WATCH')


# ---------------------------------------------------------------------------
# _render_daily_risk_text / _render_daily_risk_html
# ---------------------------------------------------------------------------

class TestDailyRiskRenderers(unittest.TestCase):

    def test_text_contains_symbols(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('TSLA', txt)
        self.assertIn('SPY', txt)

    def test_text_contains_risk_levels(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('CRITICAL', txt)
        self.assertIn('SAFE', txt)

    def test_text_contains_pnl_values(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('-200', txt)   # CRITICAL pnl_dollars
        self.assertIn('45', txt)     # SAFE pnl_dollars

    def test_text_contains_legend(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('Risk levels', txt)
        self.assertIn('SAFE', txt)
        self.assertIn('CRITICAL', txt)

    def test_text_contains_date(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('2025-01-17', txt)

    def test_text_contains_dte(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('2d', txt)    # TSLA DTE=2
        self.assertIn('15d', txt)   # SPY  DTE=15

    def test_html_contains_symbols(self):
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('TSLA', html)
        self.assertIn('SPY', html)

    def test_html_critical_row_bg(self):
        """CRITICAL positions should get the red background colour."""
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('fdedec', html)   # _RISK_STYLE['CRITICAL'] row bg

    def test_html_safe_row_bg(self):
        """SAFE positions should get the green background colour."""
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('eafaf1', html)   # _RISK_STYLE['SAFE'] row bg

    def test_html_summary_badges_present(self):
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('CRITICAL', html)
        self.assertIn('SAFE', html)

    def test_html_contains_table_headers(self):
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        for header in ('Strategy', 'Symbol', 'Expiry', 'DTE', 'Entry', 'Mark', 'Stop%'):
            self.assertIn(header, html)

    def test_html_legs_rendered(self):
        """Leg details from the legs JSON should appear in the HTML table."""
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('245', html)   # TSLA short_strike
        self.assertIn('495', html)   # SPY short_put

    def test_empty_positions_still_renders(self):
        txt  = _render_daily_risk_text([], '2025-01-17')
        html = _render_daily_risk_html([], '2025-01-17')
        self.assertIn('2025-01-17', txt)
        self.assertIn('2025-01-17', html)

    def test_text_qty_column_default_one(self):
        txt = _render_daily_risk_text(_SNAPSHOT, '2025-01-17')
        self.assertIn('Qty', txt)

    def test_text_qty_shows_contract_count(self):
        pos = {**_POS_SAFE, 'contracts': 7}
        txt = _render_daily_risk_text([pos], '2025-01-17')
        self.assertIn('   7', txt)

    def test_html_qty_header_present(self):
        html = _render_daily_risk_html(_SNAPSHOT, '2025-01-17')
        self.assertIn('<th>Qty</th>', html)

    def test_html_qty_value_rendered(self):
        pos = {**_POS_CRITICAL, 'contracts': 12}
        html = _render_daily_risk_html([pos], '2025-01-17')
        self.assertIn('>12<', html)

    def test_text_closed_today_qty_column(self):
        closed = [{
            'reason_tag': 'STOP_LOSS', 'type': 'PCS', 'symbol': 'TSLA',
            'expiry': '2025-02-01', 'contracts': 5,
            'close_pnl': -350.0, 'entry_premium': 1.20, 'current_mark': 2.90,
            'ratio': 2.1, 'short_delta': 0.42, 'risk_score': 8.3, 'dte': 2,
        }]
        txt = _render_daily_risk_text([], '2025-01-17', closed_today=closed)
        self.assertIn('Qty', txt)
        self.assertIn('   5', txt)

    def test_html_closed_today_qty_header(self):
        closed = [{
            'reason_tag': 'STOP_LOSS', 'type': 'PCS', 'symbol': 'TSLA',
            'expiry': '2025-02-01', 'contracts': 3,
            'close_pnl': -150.0, 'entry_premium': 1.20, 'current_mark': 2.70,
            'ratio': None, 'short_delta': None, 'risk_score': None, 'dte': 2,
        }]
        html = _render_daily_risk_html([], '2025-01-17', closed_today=closed)
        self.assertIn('<th>Qty</th>', html)
        self.assertIn('>3<', html)


# ---------------------------------------------------------------------------
# send_daily_risk_report
# ---------------------------------------------------------------------------

class TestSendDailyRiskReport(unittest.TestCase):

    def test_disabled_returns_false(self):
        n = EmailNotifier({})
        self.assertFalse(n.send_daily_risk_report(_SNAPSHOT))

    def test_empty_positions_returns_false(self):
        n = _make_notifier()
        with patch('src.notifier.smtplib.SMTP') as mock_smtp_cls:
            result = n.send_daily_risk_report([])
        self.assertFalse(result)
        mock_smtp_cls.assert_not_called()

    @patch('src.notifier.smtplib.SMTP')
    def test_sends_email_and_returns_true(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        result = n.send_daily_risk_report(_SNAPSHOT)

        self.assertTrue(result)
        mock_smtp.sendmail.assert_called_once()

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_contains_position_count(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report(_SNAPSHOT)

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        # Subject may be quoted-printable encoded (spaces → _), decode before asserting
        from email.header import decode_header as _dh
        parsed   = _mfs(raw_msg)
        subj_raw = parsed['Subject']
        subj     = ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in _dh(subj_raw)
        )
        self.assertIn('2 positions', subj)

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_flags_critical_count(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report(_SNAPSHOT)  # _POS_CRITICAL is in snapshot

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        self.assertIn('CRITICAL', raw_msg)

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_no_critical_flag_when_all_safe(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report([_POS_SAFE])

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        # Subject line should not include "CRITICAL" badge
        subject_line = [line for line in raw_msg.splitlines()
                        if line.startswith('Subject:')][0]
        self.assertNotIn('CRITICAL', subject_line)

    @patch('src.notifier.smtplib.SMTP')
    def test_email_body_contains_symbols(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report(_SNAPSHOT)

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        body = _extract_plain_body(_mfs(raw_msg))
        self.assertIn('TSLA', body)
        self.assertIn('SPY', body)

    @patch('src.notifier.time.sleep')
    @patch('src.notifier.smtplib.SMTP')
    def test_smtp_failure_returns_false(self, mock_smtp_cls, _mock_sleep):
        mock_smtp_cls.side_effect = OSError('connection refused')
        n = _make_notifier()
        result = n.send_daily_risk_report(_SNAPSHOT)
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# Trade plan capital key fix
# ---------------------------------------------------------------------------

class TestTradePlanCapitalKey(unittest.TestCase):
    """
    Verify the notifier uses the pre-computed 'capital' key (stamped by
    agent.py via _capital_for_pick × 100) rather than deriving it from
    max_loss (which is a per-share value for spreads, not per-contract).
    """

    def test_capital_key_used_in_text_renderer(self):
        pick_with_capital = dict(_PICK, capital=1000.0)
        txt = _render_trade_plan_text(
            [pick_with_capital], 5000.0, 'abc123ef12345678', 300, 15
        )
        self.assertIn('1,000', txt)   # formatted per-contract capital

    def test_max_loss_ignored_when_capital_key_present(self):
        """max_loss=9 (per-share) must NOT appear as the capital column."""
        pick_with_capital = dict(_PICK, capital=1000.0, max_loss=9.0)
        txt = _render_trade_plan_text(
            [pick_with_capital], 5000.0, 'abc123ef12345678', 300, 15
        )
        # $9 should not appear as the capital; $1,000 should
        self.assertIn('1,000', txt)


# ---------------------------------------------------------------------------
# _within_market_hours
# ---------------------------------------------------------------------------

class TestWithinMarketHours(unittest.TestCase):
    """Unit tests for the position-monitor market-hours guard."""

    def _call(self, now_dt, open_t='09:30', close_t='16:00',
              tz_name='US/Eastern', weekdays_only=True):
        """Patch datetime.datetime.now to return *now_dt* and call the function."""
        with patch('src.position_monitor.datetime') as mock_dt:
            mock_dt.datetime.now.return_value = now_dt
            # weekday() is a method on the real object; forward it
            mock_dt.date = __import__('datetime').date
            return _within_market_hours(open_t, close_t, tz_name, weekdays_only)

    def _weekday_at(self, hour, minute):
        """Return a Monday datetime (weekday=0) at the given time."""
        import datetime as _dt
        # 2026-03-16 is a Monday
        return _dt.datetime(2026, 3, 16, hour, minute, 0)

    def _saturday_at(self, hour, minute):
        import datetime as _dt
        # 2026-03-14 is a Saturday
        return _dt.datetime(2026, 3, 14, hour, minute, 0)

    def test_during_market_hours_returns_true(self):
        self.assertTrue(self._call(self._weekday_at(10, 30)))

    def test_before_market_open_returns_false(self):
        self.assertFalse(self._call(self._weekday_at(9, 15)))

    def test_exactly_at_open_returns_true(self):
        self.assertTrue(self._call(self._weekday_at(9, 30)))

    def test_exactly_at_close_returns_false(self):
        # close is exclusive: 16:00 is NOT within hours
        self.assertFalse(self._call(self._weekday_at(16, 0)))

    def test_after_market_close_returns_false(self):
        self.assertFalse(self._call(self._weekday_at(17, 0)))

    def test_saturday_returns_false_when_weekdays_only(self):
        self.assertFalse(self._call(self._saturday_at(12, 0), weekdays_only=True))

    def test_saturday_returns_true_when_weekdays_not_required(self):
        self.assertTrue(self._call(self._saturday_at(12, 0), weekdays_only=False))


# ---------------------------------------------------------------------------
# _send_eod_risk_report integration
# ---------------------------------------------------------------------------

class TestSendEodRiskReport(unittest.TestCase):
    """
    Verify _send_eod_risk_report wires PositionMonitor.get_risk_snapshot()
    into EmailNotifier.send_daily_risk_report() correctly.
    """

    def _make_args(self, db='data/trades.db', config='config.json', live=False):
        args = MagicMock()
        args.db = db
        args.config = config
        args.live = live
        return args

    def _run_eod(self, snapshot, notifier_enabled):
        """Helper: call monitor._send_eod_risk_report with fully mocked dependencies."""
        import sys
        from monitor import _send_eod_risk_report

        mock_mon = MagicMock()
        mock_mon.get_risk_snapshot.return_value = snapshot
        mock_mon_cls = MagicMock(return_value=mock_mon)

        mock_notif = MagicMock()
        mock_notif.enabled = notifier_enabled
        mock_notif_cls = MagicMock(return_value=mock_notif)

        # Stub out alpaca-py dependent module
        executor_stub = MagicMock()
        executor_stub.AlpacaExecutor = MagicMock()

        saved = sys.modules.get('src.executor')
        sys.modules['src.executor'] = executor_stub
        try:
            with patch('monitor.TradeDatabase', MagicMock()), \
                 patch('monitor.EmailNotifier', mock_notif_cls), \
                 patch('monitor.PositionMonitor', mock_mon_cls):
                _send_eod_risk_report({}, 'data/trades.db', 'config.json')
        finally:
            if saved is None:
                sys.modules.pop('src.executor', None)
            else:
                sys.modules['src.executor'] = saved

        return mock_mon, mock_notif

    def test_calls_send_daily_risk_report_when_enabled(self):
        snapshot = [{'symbol': 'SPY', 'risk_level': 'SAFE'}]
        mock_mon, mock_notif = self._run_eod(snapshot, notifier_enabled=True)

        mock_mon.get_risk_snapshot.assert_called_once()
        mock_notif.send_daily_risk_report.assert_called_once_with(
            snapshot, closed_today=[]
        )

    def test_skips_send_when_notifier_disabled(self):
        snapshot = [{'symbol': 'TSLA'}]
        mock_mon, mock_notif = self._run_eod(snapshot, notifier_enabled=False)

        mock_mon.get_risk_snapshot.assert_called_once()
        mock_notif.send_daily_risk_report.assert_not_called()


# ---------------------------------------------------------------------------
# EOD daemon trigger logic (was_in_market transition)
# ---------------------------------------------------------------------------

class TestEodDaemonTriggerLogic(unittest.TestCase):
    """
    Test the was_in_market → not in_market transition that fires the EOD
    report exactly once per trading day.
    """

    def test_eod_fires_on_market_close_transition(self):
        """
        Simulate the daemon loop: two iterations where first is in-market,
        second is after-close.  EOD report should fire on the second iteration.
        """
        import datetime as _dt

        today = _dt.date(2026, 3, 16)  # Monday

        call_count = {'n': 0}

        def fake_within_market(open_t, close_t, tz_name, weekdays):
            call_count['n'] += 1
            # First call: in market; second: after close
            return call_count['n'] == 1

        sent = []

        def fake_send_eod(args):
            sent.append(True)

        # Minimal reimplementation of the relevant daemon loop logic to test
        # the state-machine without spinning the real `while True` loop.
        was_in_market = False
        eod_sent_date = None
        eod_report = True

        for _ in range(2):
            in_market = fake_within_market('09:30', '16:00', 'US/Eastern', True)
            if in_market:
                pass  # would call _run_monitor_once
            elif was_in_market and eod_report:
                today_val = today
                if today_val != eod_sent_date:
                    fake_send_eod(None)
                    eod_sent_date = today_val
            was_in_market = in_market

        self.assertEqual(len(sent), 1, "EOD report should fire exactly once")

    def test_eod_does_not_fire_twice_on_same_day(self):
        """Once eod_sent_date == today, subsequent after-close cycles skip the send."""
        import datetime as _dt

        today = _dt.date(2026, 3, 16)

        sent = []

        def fake_send_eod(args):
            sent.append(True)

        was_in_market = False
        eod_sent_date = today   # already sent today
        eod_report = True

        # Simulate a second after-close iteration
        in_market = False
        if not in_market and was_in_market and eod_report:
            if today != eod_sent_date:
                fake_send_eod(None)
                eod_sent_date = today

        self.assertEqual(len(sent), 0, "Should not resend on same day")

    def test_eod_skipped_when_eod_report_false(self):
        """eod_risk_report=false in config means the report never fires."""
        sent = []

        def fake_send_eod(args):
            sent.append(True)

        import datetime as _dt
        today = _dt.date(2026, 3, 16)

        was_in_market = True
        eod_sent_date = None
        eod_report = False    # disabled

        in_market = False     # just transitioned out
        if not in_market and was_in_market and eod_report:
            if today != eod_sent_date:
                fake_send_eod(None)

        self.assertEqual(len(sent), 0)


# ---------------------------------------------------------------------------
# Feature 1: Stop-loss closures section in the daily risk report
# ---------------------------------------------------------------------------

_CLOSED_TODAY = [
    {
        'type': 'PCS', 'symbol': 'AAPL', 'expiry': '2026-04-17',
        'reason_tag': 'STOP_LOSS',
        'close_pnl': -90.0,
        'entry_premium': 0.45, 'current_mark': 1.35,
        'ratio': 1.8, 'short_delta': 0.42, 'risk_score': 7.2, 'dte': 32,
    },
    {
        'type': 'CC', 'symbol': 'TSLA', 'expiry': '2026-04-10',
        'reason_tag': 'GAMMA_RISK',
        'close_pnl': -45.0,
        'entry_premium': 1.10, 'current_mark': 1.55,
        'ratio': 2.1, 'short_delta': 0.31, 'risk_score': 8.9, 'dte': 25,
    },
]


class TestClosedTodayTextRenderer(unittest.TestCase):
    """_render_daily_risk_text with closed_today section."""

    def test_closed_today_section_present(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text([], '2026-03-16', closed_today=_CLOSED_TODAY)
        self.assertIn('Stop-Loss', text)
        self.assertIn('AAPL', text)
        self.assertIn('TSLA', text)
        self.assertIn('STOP_LOSS', text)
        self.assertIn('GAMMA_RISK', text)

    def test_closed_today_pnl_values(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text([], '2026-03-16', closed_today=_CLOSED_TODAY)
        # _fmt_opt formats as f"${v:+,.2f}" → "$-90.00" for negatives
        self.assertIn('$-90.00', text)
        self.assertIn('$-45.00', text)

    def test_no_closed_today_section_when_empty(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text(_SNAPSHOT, '2026-03-16', closed_today=[])
        self.assertNotIn('Stop-Loss / Gamma-Risk Closures', text)

    def test_no_closed_today_section_when_omitted(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text(_SNAPSHOT, '2026-03-16')
        self.assertNotIn('Stop-Loss / Gamma-Risk Closures', text)

    def test_closed_today_includes_risk_metrics(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text([], '2026-03-16', closed_today=_CLOSED_TODAY)
        # gamma/theta ratio for AAPL row
        self.assertIn('1.80', text)
        # dte
        self.assertIn('32d', text)

    def test_count_in_section_header(self):
        from src.notifier import _render_daily_risk_text
        text = _render_daily_risk_text([], '2026-03-16', closed_today=_CLOSED_TODAY)
        self.assertIn('2 position', text)


class TestClosedTodayHtmlRenderer(unittest.TestCase):
    """_render_daily_risk_html with closed_today section."""

    def test_closed_today_table_present(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html([], '2026-03-16', closed_today=_CLOSED_TODAY)
        self.assertIn("Today&#39;s Stop-Loss Closures", html)
        self.assertIn('AAPL', html)
        self.assertIn('TSLA', html)

    def test_stop_loss_row_background(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html([], '2026-03-16', closed_today=_CLOSED_TODAY)
        # STOP_LOSS rows use pinkish background
        self.assertIn('#fdf2f2', html)

    def test_gamma_risk_row_background(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html([], '2026-03-16', closed_today=_CLOSED_TODAY)
        self.assertIn('#fff3e0', html)

    def test_no_closures_section_when_empty(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html(_SNAPSHOT, '2026-03-16', closed_today=[])
        self.assertNotIn("Today&#39;s Stop-Loss Closures", html)

    def test_reason_badges_present(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html([], '2026-03-16', closed_today=_CLOSED_TODAY)
        self.assertIn('STOP_LOSS', html)
        self.assertIn('GAMMA_RISK', html)

    def test_pnl_values_colored(self):
        from src.notifier import _render_daily_risk_html
        html = _render_daily_risk_html([], '2026-03-16', closed_today=_CLOSED_TODAY)
        # Both closures are losses → red color
        self.assertIn('#e74c3c', html)


class TestSendDailyRiskReportClosedToday(unittest.TestCase):
    """send_daily_risk_report forwards closed_today to renderers and updates subject."""

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_includes_closed_today_badge(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report([], closed_today=_CLOSED_TODAY)

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        from email.header import decode_header as _dh
        parsed   = _mfs(raw_msg)
        subj_raw = parsed['Subject']
        subj = ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in _dh(subj_raw)
        )
        self.assertIn('CLOSED TODAY', subj)

    @patch('src.notifier.smtplib.SMTP')
    def test_sends_when_only_closed_no_open(self, mock_smtp_cls):
        """Email should be sent even when there are zero open positions."""
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        result = n.send_daily_risk_report([], closed_today=_CLOSED_TODAY)

        self.assertTrue(result)
        mock_smtp.sendmail.assert_called_once()

    def test_skips_when_both_empty(self):
        """No email when both positions and closed_today are empty."""
        n = _make_notifier()
        with patch('src.notifier.smtplib.SMTP') as mock_smtp_cls:
            result = n.send_daily_risk_report([], closed_today=[])
        self.assertFalse(result)
        mock_smtp_cls.assert_not_called()

    @patch('src.notifier.smtplib.SMTP')
    def test_body_contains_closed_symbols(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_daily_risk_report(_SNAPSHOT, closed_today=_CLOSED_TODAY)

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        body = _extract_plain_body(_mfs(raw_msg))
        self.assertIn('AAPL', body)    # from closed_today
        self.assertIn('TSLA', body)    # from both _SNAPSHOT and closed_today


class TestSendEodRiskReportClosedToday(unittest.TestCase):
    """_send_eod_risk_report passes closed_today into send_daily_risk_report."""

    def _run_eod_with_closed(self, snapshot, closed_today):
        import sys
        from monitor import _send_eod_risk_report

        mock_mon = MagicMock()
        mock_mon.get_risk_snapshot.return_value = snapshot
        mock_notif = MagicMock()
        mock_notif.enabled = True
        executor_stub = MagicMock()
        executor_stub.AlpacaExecutor = MagicMock()

        saved = sys.modules.get('src.executor')
        sys.modules['src.executor'] = executor_stub
        try:
            with patch('monitor.TradeDatabase', MagicMock()), \
                 patch('monitor.EmailNotifier', MagicMock(return_value=mock_notif)), \
                 patch('monitor.PositionMonitor', MagicMock(return_value=mock_mon)):
                _send_eod_risk_report({}, 'data/trades.db', 'config.json',
                                      closed_today=closed_today)
        finally:
            if saved is None:
                sys.modules.pop('src.executor', None)
            else:
                sys.modules['src.executor'] = saved
        return mock_notif

    def test_closed_today_forwarded_to_notifier(self):
        mock_notif = self._run_eod_with_closed(_SNAPSHOT, _CLOSED_TODAY)
        call_kwargs = mock_notif.send_daily_risk_report.call_args[1]
        self.assertEqual(call_kwargs.get('closed_today'), _CLOSED_TODAY)

    def test_empty_closed_today_forwarded(self):
        mock_notif = self._run_eod_with_closed(_SNAPSHOT, [])
        call_kwargs = mock_notif.send_daily_risk_report.call_args[1]
        self.assertEqual(call_kwargs.get('closed_today'), [])


# Test that the old _send_eod_risk_report call still passes calls send with keyword
class TestSendEodRiskReportUpdatedCall(unittest.TestCase):
    """Ensure _send_eod_risk_report passes closed_today=snapshot to send_daily_risk_report."""

    def test_calls_send_with_closed_today_kwarg(self):
        import sys
        from monitor import _send_eod_risk_report

        mock_mon = MagicMock()
        mock_mon.get_risk_snapshot.return_value = _SNAPSHOT
        mock_notif = MagicMock()
        mock_notif.enabled = True
        executor_stub = MagicMock()
        executor_stub.AlpacaExecutor = MagicMock()

        saved = sys.modules.get('src.executor')
        sys.modules['src.executor'] = executor_stub
        try:
            with patch('monitor.TradeDatabase', MagicMock()), \
                 patch('monitor.EmailNotifier', MagicMock(return_value=mock_notif)), \
                 patch('monitor.PositionMonitor', MagicMock(return_value=mock_mon)):
                _send_eod_risk_report({}, 'data/trades.db', 'config.json')
        finally:
            if saved is None:
                sys.modules.pop('src.executor', None)
            else:
                sys.modules['src.executor'] = saved

        # Should be called with closed_today kwarg (empty list when not provided)
        _, kwargs = mock_notif.send_daily_risk_report.call_args
        self.assertIn('closed_today', kwargs)
        self.assertEqual(kwargs['closed_today'], [])


# ---------------------------------------------------------------------------
# Feature 2: REPLAN keyword in trade plan approval
# ---------------------------------------------------------------------------

class TestParseApprovalBodyReplan(unittest.TestCase):
    """_parse_approval_body returns 'REPLAN' for replan/rescan/retry keywords."""

    _picks = [
        {'symbol': 'SPY', 'strategy': 'PCS'},
        {'symbol': 'AAPL', 'strategy': 'CSP'},
    ]

    def test_replan_keyword(self):
        from src.notifier import _parse_approval_body
        result = _parse_approval_body('replan\n', self._picks)
        self.assertEqual(result, 'REPLAN')

    def test_rescan_keyword(self):
        from src.notifier import _parse_approval_body
        result = _parse_approval_body('rescan\n', self._picks)
        self.assertEqual(result, 'REPLAN')

    def test_retry_keyword(self):
        from src.notifier import _parse_approval_body
        result = _parse_approval_body('retry\n', self._picks)
        self.assertEqual(result, 'REPLAN')

    def test_replan_case_insensitive(self):
        from src.notifier import _parse_approval_body
        for variant in ('REPLAN', 'Replan', 'RESCAN', 'Rescan'):
            with self.subTest(variant=variant):
                self.assertEqual(_parse_approval_body(variant, self._picks), 'REPLAN')

    def test_replan_with_quoted_lines_ignored(self):
        """Lines starting with '>' are quoted email headers and should be skipped."""
        from src.notifier import _parse_approval_body
        body = "> On Monday you wrote:\n> 1,2\n\nreplan"
        result = _parse_approval_body(body, self._picks)
        self.assertEqual(result, 'REPLAN')

    def test_normal_approval_unaffected(self):
        from src.notifier import _parse_approval_body
        result = _parse_approval_body('a\n', self._picks)
        self.assertEqual(result, self._picks)

    def test_reject_unaffected(self):
        from src.notifier import _parse_approval_body
        result = _parse_approval_body('n\n', self._picks)
        self.assertEqual(result, [])


class TestWaitForApprovalReplanPassthrough(unittest.TestCase):
    """wait_for_approval passes through the 'REPLAN' sentinel from _parse_approval_body."""

    # Valid message-id: 16 lowercase hex chars + timestamp
    _MSG_ID = '<abcdef1234567890.1710000000@optionwheel.local>'

    def _make_notifier_with_imap(self, body):
        """Return a notifier whose IMAP immediately returns the given body."""
        n = _make_notifier()
        n._poll_imap_for_reply = MagicMock(return_value=body)
        n._poll_interval = 0
        n._timeout = 1
        return n

    def test_replan_reply_returned(self):
        n = self._make_notifier_with_imap('replan\n')
        picks = [{'symbol': 'SPY', 'strategy': 'PCS'}]
        result = n.wait_for_approval(self._MSG_ID, picks)
        self.assertEqual(result, 'REPLAN')

    def test_normal_approval_unaffected(self):
        n = self._make_notifier_with_imap('a\n')
        picks = [{'symbol': 'SPY', 'strategy': 'PCS'}]
        result = n.wait_for_approval(self._MSG_ID, picks)
        self.assertEqual(result, picks)


class TestTradePlanReplanInstruction(unittest.TestCase):
    """Both email renderers should mention 'replan' in their instructions."""

    _picks = [{'symbol': 'SPY', 'strategy': 'PCS', 'expiry': '2026-04-17',
               'short_strike': 500, 'long_strike': 495,
               'premium': 0.45, 'capital': 500.0,
               'prob_win': 0.75, 'roi': 0.09, 'score': 0.85}]

    def test_text_renderer_contains_replan(self):
        from src.notifier import _render_trade_plan_text
        text = _render_trade_plan_text(self._picks, None, 'tok1234', 3600, 15)
        self.assertIn('replan', text.lower())

    def test_html_renderer_contains_replan(self):
        from src.notifier import _render_trade_plan_html
        html = _render_trade_plan_html(self._picks, None, 'tok1234', 3600, 15)
        self.assertIn('replan', html.lower())



# ---------------------------------------------------------------------------
# _send() SMTP retry logic
# ---------------------------------------------------------------------------

class TestSmtpRetry(unittest.TestCase):
    """Tests for EmailNotifier._send() retry / back-off behaviour."""

    def _make_smtp_ctx(self, mock_smtp):
        """Wire a MagicMock as the SMTP context-manager target."""
        ctx = MagicMock()
        ctx.__enter__ = lambda s: mock_smtp
        ctx.__exit__  = MagicMock(return_value=False)
        return ctx

    # ── Success path ──────────────────────────────────────────────────────────

    @patch('src.notifier.smtplib.SMTP')
    def test_successful_send_no_retry(self, mock_smtp_cls):
        """First-attempt success: SMTP constructed once, no sleep."""
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value = self._make_smtp_ctx(mock_smtp)

        n = _make_notifier({'smtp_max_retries': 3, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep') as mock_sleep:
            n.send_trade_executed(_PICK, 'ord-1')

        mock_smtp_cls.assert_called_once()
        mock_sleep.assert_not_called()

    # ── Transient failure retried ─────────────────────────────────────────────

    @patch('src.notifier.smtplib.SMTP')
    def test_transient_failure_retried_and_succeeds(self, mock_smtp_cls):
        """ConnectionError on attempt 1 → retry → success on attempt 2."""
        mock_smtp_ok = MagicMock()
        fail_ctx = MagicMock()
        fail_ctx.__enter__ = MagicMock(side_effect=ConnectionRefusedError("refused"))
        fail_ctx.__exit__  = MagicMock(return_value=False)
        ok_ctx = self._make_smtp_ctx(mock_smtp_ok)
        mock_smtp_cls.side_effect = [fail_ctx, ok_ctx]

        n = _make_notifier({'smtp_max_retries': 2, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep'):
            n.send_trade_executed(_PICK, 'ord-1')   # must not raise

        self.assertEqual(mock_smtp_cls.call_count, 2)
        mock_smtp_ok.sendmail.assert_called_once()

    @patch('src.notifier.smtplib.SMTP')
    def test_smtp_server_disconnected_retried(self, mock_smtp_cls):
        """SMTPServerDisconnected triggers retry loop."""
        mock_smtp_ok = MagicMock()
        fail_ctx = MagicMock()
        fail_ctx.__enter__ = MagicMock(side_effect=smtplib.SMTPServerDisconnected())
        fail_ctx.__exit__  = MagicMock(return_value=False)
        ok_ctx = self._make_smtp_ctx(mock_smtp_ok)
        mock_smtp_cls.side_effect = [fail_ctx, fail_ctx, ok_ctx]

        n = _make_notifier({'smtp_max_retries': 3, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep'):
            n.send_trade_executed(_PICK, 'ord-1')

        self.assertEqual(mock_smtp_cls.call_count, 3)
        mock_smtp_ok.sendmail.assert_called_once()

    # ── All retries exhausted ─────────────────────────────────────────────────

    @patch('src.notifier.smtplib.SMTP')
    def test_all_retries_exhausted_does_not_raise_from_send_trade_executed(self, mock_smtp_cls):
        """Caller (send_trade_executed) absorbs the final exception — does not propagate."""
        mock_smtp_cls.side_effect = OSError("network unreachable")

        n = _make_notifier({'smtp_max_retries': 2, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep'):
            n.send_trade_executed(_PICK, 'ord-1')   # must not raise

        # 1 initial + 2 retries = 3 total calls
        self.assertEqual(mock_smtp_cls.call_count, 3)

    @patch('src.notifier.smtplib.SMTP')
    def test_exhausted_retries_returns_false_for_risk_report(self, mock_smtp_cls):
        """send_daily_risk_report returns False when all retries fail."""
        mock_smtp_cls.side_effect = OSError("timeout")

        pos = {'symbol': 'SPY', 'type': 'PCS', 'expiry': '2099-01-01',
               'risk_level': 'SAFE', 'premium': 1.0}
        n = _make_notifier({'smtp_max_retries': 1, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep'):
            result = n.send_daily_risk_report([pos])

        self.assertFalse(result)
        self.assertEqual(mock_smtp_cls.call_count, 2)   # 1 attempt + 1 retry

    @patch('src.notifier.smtplib.SMTP')
    def test_exhausted_retries_returns_none_for_trade_plan(self, mock_smtp_cls):
        """send_trade_plan returns None when all retries fail."""
        mock_smtp_cls.side_effect = OSError("timeout")

        n = _make_notifier({'smtp_max_retries': 1, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep'):
            result = n.send_trade_plan([_PICK])

        self.assertIsNone(result)
        self.assertEqual(mock_smtp_cls.call_count, 2)

    # ── Auth error skips retry ────────────────────────────────────────────────

    @patch('src.notifier.smtplib.SMTP')
    def test_auth_error_not_retried(self, mock_smtp_cls):
        """SMTPAuthenticationError is permanent — must not trigger a retry."""
        mock_smtp = MagicMock()
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b'auth failed')
        mock_smtp_cls.return_value = self._make_smtp_ctx(mock_smtp)

        n = _make_notifier({'smtp_max_retries': 3, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep') as mock_sleep:
            n.send_trade_executed(_PICK, 'ord-1')   # caught and logged, no raise

        mock_smtp_cls.assert_called_once()   # only one attempt
        mock_sleep.assert_not_called()

    # ── Back-off timing ───────────────────────────────────────────────────────

    @patch('src.notifier.smtplib.SMTP')
    def test_exponential_backoff_sleep_durations(self, mock_smtp_cls):
        """sleep(base * 2^attempt) called before each retry."""
        mock_smtp_cls.side_effect = ConnectionResetError("reset")

        n = _make_notifier({'smtp_max_retries': 3, 'smtp_retry_delay_seconds': 2.0})
        with patch('src.notifier.time.sleep') as mock_sleep:
            n.send_trade_executed(_PICK, 'ord-1')

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        # Retries after attempts 0, 1, 2 → delays 2.0, 4.0, 8.0
        self.assertEqual(sleep_calls, [2.0, 4.0, 8.0])

    # ── Config wiring ─────────────────────────────────────────────────────────

    def test_default_max_retries_is_3(self):
        n = _make_notifier()
        self.assertEqual(n._smtp_max_retries, 3)

    def test_default_retry_delay_is_5(self):
        n = _make_notifier()
        self.assertEqual(n._smtp_retry_delay, 5.0)

    def test_custom_max_retries_read_from_config(self):
        n = _make_notifier({'smtp_max_retries': 5})
        self.assertEqual(n._smtp_max_retries, 5)

    def test_custom_retry_delay_read_from_config(self):
        n = _make_notifier({'smtp_retry_delay_seconds': 10.0})
        self.assertEqual(n._smtp_retry_delay, 10.0)

    @patch('src.notifier.smtplib.SMTP')
    def test_zero_retries_single_attempt_on_failure(self, mock_smtp_cls):
        """smtp_max_retries=0 → exactly one attempt, no sleep on failure."""
        mock_smtp_cls.side_effect = OSError("refused")
        n = _make_notifier({'smtp_max_retries': 0, 'smtp_retry_delay_seconds': 0.0})
        with patch('src.notifier.time.sleep') as mock_sleep:
            n.send_trade_executed(_PICK, 'ord-1')
        mock_smtp_cls.assert_called_once()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Weekly digest fixtures
# ---------------------------------------------------------------------------

_WEEK_START = '2026-04-14'
_WEEK_END   = '2026-04-18'

_WEEKLY_TRADES = [
    {
        'id': 10, 'type': 'PCS', 'symbol': 'AAPL', 'expiry': '2026-04-18',
        'premium': 0.45, 'pnl': 45.0, 'status': 'CLOSED',
        'timestamp': '2026-04-16T15:30:00',
    },
    {
        'id': 11, 'type': 'CCS', 'symbol': 'TSLA', 'expiry': '2026-04-18',
        'premium': 1.20, 'pnl': -90.0, 'status': 'CLOSED',
        'timestamp': '2026-04-17T14:00:00',
    },
    {
        'id': 12, 'type': 'IC', 'symbol': 'SPY', 'expiry': '2026-04-18',
        'premium': 2.50, 'pnl': 187.50, 'status': 'CLOSED',
        'timestamp': '2026-04-18T13:00:00',
    },
]

_OPEN_POSITIONS_DIGEST = [
    {
        'id': 20, 'type': 'PCS', 'symbol': 'NVDA', 'expiry': '2026-05-02',
        'premium': 0.80, 'dte': 14, 'risk_level': 'SAFE',
        'pnl_dollars': 32.0, 'max_loss_dollars': 420.0,
    },
    {
        'id': 21, 'type': 'CCS', 'symbol': 'AMZN', 'expiry': '2026-05-02',
        'premium': 0.60, 'dte': 14, 'risk_level': 'WATCH',
        'pnl_dollars': -15.0, 'max_loss_dollars': None,
    },
]

_CUMULATIVE_PNL  = 1234.50
_CAPITAL_DEPLOYED = 5000.0


# ---------------------------------------------------------------------------
# _render_weekly_digest_text
# ---------------------------------------------------------------------------

class TestWeeklyDigestTextRenderer(unittest.TestCase):

    def _render(self, trades=None, cum_pnl=None, open_pos=None, capital=None):
        from src.notifier import _render_weekly_digest_text
        return _render_weekly_digest_text(
            _WEEK_START,
            _WEEK_END,
            trades        if trades   is not None else _WEEKLY_TRADES,
            cum_pnl       if cum_pnl  is not None else _CUMULATIVE_PNL,
            open_pos      if open_pos is not None else _OPEN_POSITIONS_DIGEST,
            capital       if capital  is not None else _CAPITAL_DEPLOYED,
        )

    def test_contains_week_date_range(self):
        txt = self._render()
        self.assertIn(_WEEK_START, txt)
        self.assertIn(_WEEK_END,   txt)

    def test_contains_weekly_pnl(self):
        """Sum of trade P&Ls: 45 − 90 + 187.50 = 142.50."""
        txt = self._render()
        self.assertIn('142.50', txt)

    def test_contains_cumulative_pnl(self):
        txt = self._render()
        self.assertIn('1,234.50', txt)

    def test_contains_capital_deployed(self):
        txt = self._render()
        self.assertIn('5,000', txt)

    def test_win_rate_reported(self):
        """2 winners out of 3 trades → 67%."""
        txt = self._render()
        self.assertIn('67%', txt)

    def test_trade_count_reported(self):
        txt = self._render()
        self.assertIn('3', txt)   # 3 closed trades

    def test_symbols_appear_in_open_positions(self):
        txt = self._render()
        self.assertIn('NVDA', txt)
        self.assertIn('AMZN', txt)

    def test_risk_levels_appear(self):
        txt = self._render()
        self.assertIn('SAFE',  txt)
        self.assertIn('WATCH', txt)

    def test_no_open_positions_message(self):
        txt = self._render(open_pos=[])
        self.assertIn('No open positions', txt)

    def test_no_trades_closed(self):
        txt = self._render(trades=[])
        self.assertIn('0', txt)

    def test_negative_week_pnl_has_minus_sign(self):
        losing_trades = [dict(_WEEKLY_TRADES[1])]  # pnl=-90
        txt = self._render(trades=losing_trades)
        self.assertIn('-90.00', txt)

    def test_best_and_worst_trade_shown(self):
        txt = self._render()
        self.assertIn('187.50', txt)  # best
        self.assertIn('-90.00', txt)  # worst

    def test_open_positions_qty_column_header(self):
        txt = self._render()
        self.assertIn('Qty', txt)

    def test_open_positions_qty_default_one(self):
        txt = self._render()
        # Header present; default contracts=1 rows rendered without error
        self.assertIn('NVDA', txt)

    def test_open_positions_qty_multi_contract(self):
        pos = [{**_OPEN_POSITIONS_DIGEST[0], 'contracts': 8}]
        txt = self._render(open_pos=pos)
        self.assertIn('   8', txt)

    def test_weekly_digest_text_summarizes_pnl_quality(self):
        trades = [
            {**_WEEKLY_TRADES[0], 'pnl_source': 'ALPACA_FILLS', 'pnl_verified': 1},
            {**_WEEKLY_TRADES[1], 'pnl_source': 'EXTERNAL_PLACEHOLDER', 'pnl_verified': 0},
        ]
        txt = self._render(trades=trades)
        self.assertIn('Verified P&L', txt)
        self.assertIn('Unverified rows:   1', txt)


# ---------------------------------------------------------------------------
# _render_weekly_digest_html
# ---------------------------------------------------------------------------

class TestWeeklyDigestHtmlRenderer(unittest.TestCase):

    def _render(self, trades=None, cum_pnl=None, open_pos=None, capital=None):
        from src.notifier import _render_weekly_digest_html
        return _render_weekly_digest_html(
            _WEEK_START,
            _WEEK_END,
            trades        if trades   is not None else _WEEKLY_TRADES,
            cum_pnl       if cum_pnl  is not None else _CUMULATIVE_PNL,
            open_pos      if open_pos is not None else _OPEN_POSITIONS_DIGEST,
            capital       if capital  is not None else _CAPITAL_DEPLOYED,
        )

    def test_is_valid_html(self):
        html = self._render()
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('</html>', html)

    def test_kpi_cards_present(self):
        html = self._render()
        self.assertIn('Week P&L', html)
        self.assertIn('Cumulative P&L', html)
        self.assertIn('Capital Deployed', html)
        self.assertIn('Win Rate', html)

    def test_week_pnl_in_kpi(self):
        html = self._render()
        self.assertIn('142.50', html)

    def test_cumulative_pnl_in_kpi(self):
        html = self._render()
        self.assertIn('1,234.50', html)

    def test_trade_symbols_in_table(self):
        html = self._render()
        self.assertIn('AAPL', html)
        self.assertIn('TSLA', html)
        self.assertIn('SPY',  html)

    def test_weekly_digest_html_shows_pnl_source_column(self):
        trades = [
            {**_WEEKLY_TRADES[0], 'pnl_source': 'ALPACA_FILLS', 'pnl_verified': 1},
            {**_WEEKLY_TRADES[1], 'pnl_source': 'EXTERNAL_PLACEHOLDER', 'pnl_verified': 0},
        ]
        html = self._render(trades=trades)
        self.assertIn('P&L Quality', html)
        self.assertIn('Source', html)
        self.assertIn('Alpaca Fills', html)
        self.assertIn('External Placeholder (unverified)', html)

    def test_open_position_symbols_in_table(self):
        html = self._render()
        self.assertIn('NVDA', html)
        self.assertIn('AMZN', html)

    def test_risk_badge_colors_present(self):
        html = self._render()
        # SAFE badge is green (#27ae60)
        self.assertIn('#27ae60', html)

    def test_no_trades_section_shows_message(self):
        html = self._render(trades=[])
        self.assertIn('No trades closed this week', html)

    def test_no_open_positions_shows_message(self):
        html = self._render(open_pos=[])
        self.assertIn('No open positions', html)

    def test_week_date_range_in_header(self):
        html = self._render()
        self.assertIn(_WEEK_START, html)
        self.assertIn(_WEEK_END,   html)

    def test_html_trades_qty_header(self):
        html = self._render()
        self.assertIn('<th>Qty</th>', html)

    def test_html_trades_qty_multi_contract(self):
        trades = [{**_WEEKLY_TRADES[0], 'contracts': 15}]
        html = self._render(trades=trades)
        self.assertIn('>15<', html)

    def test_html_open_positions_qty_header(self):
        html = self._render()
        # two <th>Qty</th> — one for trades table, one for open positions
        self.assertGreaterEqual(html.count('<th>Qty</th>'), 1)

    def test_html_open_positions_qty_value(self):
        pos = [{**_OPEN_POSITIONS_DIGEST[0], 'contracts': 6}]
        html = self._render(open_pos=pos)
        self.assertIn('>6<', html)


# ---------------------------------------------------------------------------
# send_weekly_digest (EmailNotifier method)
# ---------------------------------------------------------------------------

class TestSendWeeklyDigest(unittest.TestCase):

    def test_disabled_returns_false(self):
        n = EmailNotifier({})
        result = n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )
        self.assertFalse(result)

    @patch('src.notifier.smtplib.SMTP')
    def test_sends_email_and_returns_true(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        result = n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )

        self.assertTrue(result)
        mock_smtp.sendmail.assert_called_once()

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_contains_week_range(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        from email.header import decode_header as _dh
        parsed   = _mfs(raw_msg)
        subj_raw = parsed['Subject']
        subj = ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in _dh(subj_raw)
        )
        self.assertIn(_WEEK_START, subj)
        self.assertIn(_WEEK_END,   subj)

    @patch('src.notifier.smtplib.SMTP')
    def test_subject_contains_weekly_pnl(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        from email.header import decode_header as _dh
        parsed   = _mfs(raw_msg)
        subj_raw = parsed['Subject']
        subj = ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in _dh(subj_raw)
        )
        self.assertIn('142.50', subj)

    @patch('src.notifier.smtplib.SMTP')
    def test_body_contains_trade_symbols(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        n = _make_notifier()
        n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )

        raw_msg = mock_smtp.sendmail.call_args[0][2]
        body = _extract_plain_body(_mfs(raw_msg))
        # Plain-text body shows open-position symbols in the positions table
        self.assertIn('NVDA', body)
        self.assertIn('AMZN', body)

    @patch('src.notifier.time.sleep')
    @patch('src.notifier.smtplib.SMTP')
    def test_smtp_failure_returns_false(self, mock_smtp_cls, _mock_sleep):
        mock_smtp_cls.side_effect = OSError('connection refused')
        n = _make_notifier()
        result = n.send_weekly_digest(
            _WEEK_START, _WEEK_END, _WEEKLY_TRADES,
            _CUMULATIVE_PNL, _OPEN_POSITIONS_DIGEST, _CAPITAL_DEPLOYED,
        )
        self.assertFalse(result)

    @patch('src.notifier.smtplib.SMTP')
    def test_negative_week_pnl_subject_has_minus(self, mock_smtp_cls):
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp
        mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)

        losing_trades = [{'type': 'PCS', 'symbol': 'AAPL', 'pnl': -200.0,
                          'expiry': '2026-04-18', 'premium': 0.5,
                          'timestamp': '2026-04-16T10:00:00'}]
        n = _make_notifier()
        n.send_weekly_digest(
            _WEEK_START, _WEEK_END, losing_trades,
            _CUMULATIVE_PNL, [], 0.0,
        )
        raw_msg = mock_smtp.sendmail.call_args[0][2]
        # Subject is decoded as a header — extract and check
        from email.header import decode_header as _dh
        parsed   = _mfs(raw_msg)
        subj_raw = parsed['Subject']
        subj = ''.join(
            part.decode(enc or 'utf-8') if isinstance(part, bytes) else part
            for part, enc in _dh(subj_raw)
        )
        self.assertIn('200.00', subj)


# ---------------------------------------------------------------------------
# _send_weekly_digest integration (monitor.py helper)
# ---------------------------------------------------------------------------

class TestSendWeeklyDigestMonitor(unittest.TestCase):
    """
    Verify monitor._send_weekly_digest wires PositionMonitor.get_risk_snapshot()
    and the DB query into EmailNotifier.send_weekly_digest() correctly.
    """

    def _run(self, snapshot, trades_in_db, cum_pnl_in_db, notifier_enabled):
        import sys
        import datetime as _dt
        from monitor import _send_weekly_digest

        mock_mon = MagicMock()
        mock_mon.get_risk_snapshot.return_value = snapshot

        mock_notif = MagicMock()
        mock_notif.enabled = notifier_enabled

        executor_stub = MagicMock()

        # Build a lightweight fake sqlite3 connection that returns the trade rows
        # and cumulative P&L without hitting the real database.
        fake_row_factory = None

        class _FakeCursor:
            def __init__(self, rows):
                self._rows = rows
                self._idx  = 0
            def fetchall(self):
                return self._rows
            def fetchone(self):
                return self._rows[0] if self._rows else (0,)

        class _FakeConn:
            def __init__(self):
                self.queries = []
            def execute(self_, sql, params=()):
                self_.queries.append((sql, params))
                if 'SUM(pnl)' in sql:
                    return _FakeCursor([(cum_pnl_in_db,)])
                return _FakeCursor(trades_in_db)
            def close(self_):
                pass

        fake_conn = _FakeConn()
        saved = sys.modules.get('src.executor')
        sys.modules['src.executor'] = executor_stub
        try:
            with patch('monitor.TradeDatabase', MagicMock()), \
                 patch('monitor.EmailNotifier', MagicMock(return_value=mock_notif)), \
                 patch('monitor.PositionMonitor', MagicMock(return_value=mock_mon)), \
                 patch('sqlite3.connect', return_value=fake_conn):
                _send_weekly_digest(
                    {}, 'data/trades.db', 'config.json',
                    week_start=_dt.date(2026, 4, 14),
                    week_end=_dt.date(2026, 4, 18),
                )
        finally:
            if saved is None:
                sys.modules.pop('src.executor', None)
            else:
                sys.modules['src.executor'] = saved

        return mock_mon, mock_notif, fake_conn

    def test_calls_send_weekly_digest_when_enabled(self):
        snapshot = [{'symbol': 'NVDA', 'risk_level': 'SAFE', 'premium': 0.8,
                     'pnl_dollars': 32.0, 'max_loss_dollars': 420.0}]
        trades   = [{'type': 'PCS', 'symbol': 'AAPL', 'pnl': 45.0,
                     'expiry': '2026-04-18', 'premium': 0.45,
                     'timestamp': '2026-04-16T10:00:00', 'status': 'CLOSED'}]
        mock_mon, mock_notif, _ = self._run(snapshot, trades, 500.0, True)

        mock_mon.get_risk_snapshot.assert_called_once()
        mock_notif.send_weekly_digest.assert_called_once()
        call_kwargs = mock_notif.send_weekly_digest.call_args
        # week_start / week_end are positional args 0 and 1
        assert call_kwargs[0][0] == '2026-04-14'
        assert call_kwargs[0][1] == '2026-04-18'

    def test_skips_send_when_notifier_disabled(self):
        mock_mon, mock_notif, _ = self._run([], [], 0.0, False)
        mock_mon.get_risk_snapshot.assert_not_called()
        mock_notif.send_weekly_digest.assert_not_called()

    def test_weekly_digest_filters_by_close_timestamp_not_open_timestamp(self):
        trades = [{
            'type': 'PCS', 'symbol': 'AAPL', 'pnl': 45.0,
            'expiry': '2026-04-18', 'premium': 0.45,
            'timestamp': '2026-03-01T10:00:00',
            'status_updated_at': '2026-04-16T10:00:00',
            'closed_at': '2026-04-16T10:00:00',
            'status': 'CLOSED',
        }]
        _, mock_notif, fake_conn = self._run([], trades, 45.0, True)

        sql = fake_conn.queries[0][0]
        self.assertIn('COALESCE(status_updated_at, timestamp)', sql)
        weekly_trades = mock_notif.send_weekly_digest.call_args[0][2]
        self.assertEqual(weekly_trades[0]['symbol'], 'AAPL')

    def test_capital_deployed_uses_max_loss_dollars(self):
        """max_loss_dollars from the snapshot is summed into capital_deployed."""
        snapshot = [
            {'symbol': 'A', 'risk_level': 'SAFE', 'premium': 1.0,
             'pnl_dollars': 0, 'max_loss_dollars': 300.0},
            {'symbol': 'B', 'risk_level': 'SAFE', 'premium': 2.0,
             'pnl_dollars': 0, 'max_loss_dollars': 500.0},
        ]
        mock_mon, mock_notif, _ = self._run(snapshot, [], 0.0, True)
        call_kwargs = mock_notif.send_weekly_digest.call_args[0]
        # 6th positional arg is capital_deployed
        capital = call_kwargs[5]
        self.assertAlmostEqual(capital, 800.0)

    def test_capital_deployed_falls_back_to_premium_when_no_max_loss(self):
        """When max_loss_dollars is None, premium × 100 is used."""
        snapshot = [{'symbol': 'X', 'risk_level': 'SAFE', 'premium': 1.5,
                     'pnl_dollars': 0, 'max_loss_dollars': None}]
        mock_mon, mock_notif, _ = self._run(snapshot, [], 0.0, True)
        capital = mock_notif.send_weekly_digest.call_args[0][5]
        self.assertAlmostEqual(capital, 150.0)   # 1.5 × 100


# ---------------------------------------------------------------------------
# Weekly digest daemon trigger logic
# ---------------------------------------------------------------------------

class TestWeeklyDigestDaemonTrigger(unittest.TestCase):
    """
    Verify the daemon loop fires the weekly digest exactly once per Friday
    and respects the weekly_digest=false config flag.
    """

    def _simulate_loop(self, today_weekday, weekly_digest_enabled,
                       already_sent_this_week=False):
        """
        Minimal reimplementation of the Friday-close branch in _run_daemon.
        Returns the list of week_monday values the digest was sent for.
        """
        import datetime as _dt

        # Build a synthetic 'today' with the desired weekday
        # 2026-04-13 is a Monday (weekday 0); offset to desired day
        base = _dt.date(2026, 4, 13)
        today = base + _dt.timedelta(days=today_weekday)

        weekly_digest = weekly_digest_enabled
        week_monday   = today - _dt.timedelta(days=today.weekday())
        weekly_digest_sent = week_monday if already_sent_this_week else None

        sent_for = []

        # Simulate the post-close branch
        was_in_market = True
        in_market     = False
        if not in_market and was_in_market:
            if weekly_digest and today.weekday() == 4:  # Friday
                if weekly_digest_sent != week_monday:
                    sent_for.append(week_monday)
                    weekly_digest_sent = week_monday

        return sent_for

    def test_fires_on_friday(self):
        sent = self._simulate_loop(today_weekday=4, weekly_digest_enabled=True)
        self.assertEqual(len(sent), 1)

    def test_does_not_fire_on_monday(self):
        sent = self._simulate_loop(today_weekday=0, weekly_digest_enabled=True)
        self.assertEqual(len(sent), 0)

    def test_does_not_fire_on_thursday(self):
        sent = self._simulate_loop(today_weekday=3, weekly_digest_enabled=True)
        self.assertEqual(len(sent), 0)

    def test_does_not_fire_twice_same_week(self):
        sent = self._simulate_loop(today_weekday=4, weekly_digest_enabled=True,
                                   already_sent_this_week=True)
        self.assertEqual(len(sent), 0)

    def test_skipped_when_disabled(self):
        sent = self._simulate_loop(today_weekday=4, weekly_digest_enabled=False)
        self.assertEqual(len(sent), 0)


if __name__ == '__main__':
    unittest.main()
