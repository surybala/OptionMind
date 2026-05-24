"""
Tests for src/notify/sender.py — EmailNotifier
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock formatter module before importing sender
_fmt_mock = MagicMock()
_fmt_mock._build_mime.return_value = MagicMock()
_fmt_mock._render_trade_executed_text.return_value = 'text'
_fmt_mock._render_trade_executed_html.return_value = '<html>'
_fmt_mock._render_position_closed_text.return_value = 'text'
_fmt_mock._render_position_closed_html.return_value = '<html>'
_fmt_mock._render_trade_plan_text.return_value = 'text'
_fmt_mock._render_trade_plan_html.return_value = '<html>'
_fmt_mock._render_daily_risk_text.return_value = 'text'
_fmt_mock._render_daily_risk_html.return_value = '<html>'
_fmt_mock._extract_token_from_msgid.return_value = 'abc123'
_fmt_mock._extract_plain_body.return_value = 'APPROVE'
_fmt_mock._parse_approval_body.return_value = [{'symbol': 'SPY'}]

sys.modules['src.notify.formatter'] = _fmt_mock

from src.notify.sender import EmailNotifier


_BASE_CFG = {
    'email': {
        'smtp_host': 'smtp.example.com',
        'smtp_port': 587,
        'imap_host': 'imap.example.com',
        'imap_port': 993,
        'from_addr': 'bot@example.com',
        'to_addr': 'user@example.com',
        'app_password': 'secret',
        'approval_timeout_seconds': 60,
        'approval_poll_interval_seconds': 5,
    }
}


def _notifier(cfg=None):
    return EmailNotifier(cfg or _BASE_CFG)


class TestEmailNotifierInit(unittest.TestCase):

    def test_enabled_with_full_config(self):
        n = _notifier()
        self.assertTrue(n.enabled)

    def test_disabled_without_email_key(self):
        n = EmailNotifier({})
        self.assertFalse(n.enabled)

    def test_disabled_missing_smtp_host(self):
        cfg = {k: v for k, v in _BASE_CFG['email'].items() if k != 'smtp_host'}
        n = EmailNotifier({'email': cfg})
        self.assertFalse(n.enabled)

    def test_disabled_missing_from_addr(self):
        cfg = {k: v for k, v in _BASE_CFG['email'].items() if k != 'from_addr'}
        n = EmailNotifier({'email': cfg})
        self.assertFalse(n.enabled)

    def test_disabled_missing_to_addr(self):
        cfg = {k: v for k, v in _BASE_CFG['email'].items() if k != 'to_addr'}
        n = EmailNotifier({'email': cfg})
        self.assertFalse(n.enabled)

    def test_disabled_empty_password(self):
        cfg = dict(_BASE_CFG['email'])
        cfg['app_password'] = ''
        # Guard against leaked env var from other tests
        saved = os.environ.pop('OPTIONWHEEL_EMAIL_PASSWORD', None)
        try:
            n = EmailNotifier({'email': cfg})
            self.assertFalse(n.enabled)
        finally:
            if saved is not None:
                os.environ['OPTIONWHEEL_EMAIL_PASSWORD'] = saved

    def test_password_from_env_var(self):
        cfg = dict(_BASE_CFG['email'])
        cfg['app_password'] = ''
        with patch.dict(os.environ, {'OPTIONWHEEL_EMAIL_PASSWORD': 'envpass'}):
            n = EmailNotifier({'email': cfg})
        self.assertTrue(n.enabled)
        self.assertEqual(n._password, 'envpass')

    def test_password_spaces_stripped(self):
        cfg = dict(_BASE_CFG['email'])
        cfg['app_password'] = 'ab cd ef'
        # Guard against leaked env var from other tests
        saved = os.environ.pop('OPTIONWHEEL_EMAIL_PASSWORD', None)
        try:
            n = EmailNotifier({'email': cfg})
            self.assertEqual(n._password, 'abcdef')
        finally:
            if saved is not None:
                os.environ['OPTIONWHEEL_EMAIL_PASSWORD'] = saved

    def test_env_var_takes_priority_over_config(self):
        with patch.dict(os.environ, {'OPTIONWHEEL_EMAIL_PASSWORD': 'fromenv'}):
            n = _notifier()
        self.assertEqual(n._password, 'fromenv')

    def test_default_ports_used_when_absent(self):
        cfg = {
            'smtp_host': 'smtp.example.com',
            'from_addr': 'a@b.com',
            'to_addr': 'c@d.com',
            'app_password': 'pw',
        }
        n = EmailNotifier({'email': cfg})
        self.assertTrue(n.enabled)
        self.assertEqual(n._smtp_port, 587)
        self.assertEqual(n._imap_port, 993)

    def test_timeout_and_poll_defaults(self):
        cfg = {
            'smtp_host': 'smtp.example.com',
            'from_addr': 'a@b.com',
            'to_addr': 'c@d.com',
            'app_password': 'pw',
        }
        n = EmailNotifier({'email': cfg})
        self.assertEqual(n._timeout, 21600)
        self.assertEqual(n._poll_interval, 15)


class TestSendTradeExecuted(unittest.TestCase):

    def test_no_op_when_disabled(self):
        n = EmailNotifier({})
        # should not raise
        n.send_trade_executed({'strategy': 'PCS', 'symbol': 'SPY', 'expiry': '2026-01-17'}, 'oid123')

    def test_calls_send_when_enabled(self):
        n = _notifier()
        with patch.object(n, '_send') as mock_send:
            n.send_trade_executed(
                {'strategy': 'PCS', 'symbol': 'SPY', 'expiry': '2026-01-17'},
                'oid123',
            )
        mock_send.assert_called_once()

    def test_send_exception_does_not_raise(self):
        n = _notifier()
        with patch.object(n, '_send', side_effect=Exception('SMTP down')):
            # should silently log and not propagate
            n.send_trade_executed({'strategy': 'PCS', 'symbol': 'SPY', 'expiry': '2026-01-17'}, 'x')

    def test_custom_reason_passed(self):
        n = _notifier()
        with patch.object(n, '_send'):
            n.send_trade_executed(
                {'strategy': 'IC', 'symbol': 'QQQ', 'expiry': '2026-01-17'},
                'oid999',
                reason='AUTO',
            )
        # just verify no error


class TestSendPositionClosed(unittest.TestCase):

    def test_no_op_when_disabled(self):
        n = EmailNotifier({})
        n.send_position_closed({'type': 'PCS', 'symbol': 'SPY', 'expiry': '2026-01-17'}, 'STOP_LOSS')

    def test_calls_send_when_enabled(self):
        n = _notifier()
        pos = {'type': 'IC', 'symbol': 'SPY', 'expiry': '2026-01-17', 'close_pnl': -50.0}
        with patch.object(n, '_send') as mock_send:
            n.send_position_closed(pos, 'STOP_LOSS')
        mock_send.assert_called_once()

    def test_negative_pnl_sign_in_subject(self):
        # Test does not raise and handles negative pnl gracefully
        n = _notifier()
        pos = {'type': 'PCS', 'symbol': 'SPY', 'expiry': '2026-01-17', 'close_pnl': -100.0}
        with patch.object(n, '_send'):
            n.send_position_closed(pos, 'STOP_LOSS')

    def test_positive_pnl_sign_in_subject(self):
        n = _notifier()
        pos = {'type': 'IC', 'symbol': 'SPY', 'expiry': '2026-01-17', 'close_pnl': 75.0}
        with patch.object(n, '_send'):
            n.send_position_closed(pos, 'PROFIT_TAKE')

    def test_send_exception_does_not_raise(self):
        n = _notifier()
        with patch.object(n, '_send', side_effect=RuntimeError('network')):
            n.send_position_closed({'type': 'PCS', 'symbol': 'SPY'}, 'STOP_LOSS')


class TestSendTradePlan(unittest.TestCase):

    def test_disabled_returns_none(self):
        n = EmailNotifier({})
        result = n.send_trade_plan([{'symbol': 'SPY'}])
        self.assertIsNone(result)

    def test_enabled_returns_msg_id_string(self):
        n = _notifier()
        with patch.object(n, '_send'):
            result = n.send_trade_plan([{'symbol': 'SPY'}, {'symbol': 'QQQ'}])
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith('<'))
        self.assertIn('@optionwheel.local', result)

    def test_send_exception_returns_none(self):
        n = _notifier()
        with patch.object(n, '_send', side_effect=Exception('SMTP error')):
            result = n.send_trade_plan([{'symbol': 'SPY'}])
        self.assertIsNone(result)

    def test_capital_budget_forwarded(self):
        n = _notifier()
        with patch.object(n, '_send'):
            result = n.send_trade_plan([{'symbol': 'SPY'}], capital_budget=10000.0, deployed_capital=5000.0)
        self.assertIsNotNone(result)


class TestSendDailyRiskReport(unittest.TestCase):

    def test_disabled_returns_false(self):
        n = EmailNotifier({})
        result = n.send_daily_risk_report([{'risk_level': 'SAFE'}])
        self.assertFalse(result)

    def test_no_positions_returns_false(self):
        n = _notifier()
        result = n.send_daily_risk_report([])
        self.assertFalse(result)

    def test_with_positions_returns_true(self):
        n = _notifier()
        positions = [{'risk_level': 'SAFE'}, {'risk_level': 'CAUTION'}]
        with patch.object(n, '_send'):
            result = n.send_daily_risk_report(positions)
        self.assertTrue(result)

    def test_critical_positions_in_subject(self):
        n = _notifier()
        positions = [{'risk_level': 'CRITICAL'}]
        with patch.object(n, '_send'):
            result = n.send_daily_risk_report(positions)
        self.assertTrue(result)

    def test_closed_today_included(self):
        n = _notifier()
        positions = [{'risk_level': 'SAFE'}]
        closed = [{'symbol': 'SPY', 'close_pnl': -50}]
        with patch.object(n, '_send'):
            result = n.send_daily_risk_report(positions, closed_today=closed)
        self.assertTrue(result)

    def test_send_exception_returns_false(self):
        n = _notifier()
        with patch.object(n, '_send', side_effect=Exception('SMTP down')):
            result = n.send_daily_risk_report([{'risk_level': 'SAFE'}])
        self.assertFalse(result)

    def test_only_closed_today_no_open_positions(self):
        n = _notifier()
        closed = [{'symbol': 'SPY', 'close_pnl': 100}]
        with patch.object(n, '_send'):
            result = n.send_daily_risk_report([], closed_today=closed)
        self.assertTrue(result)


class TestSendPrivateMethod(unittest.TestCase):

    def test_send_uses_smtp_starttls(self):
        env_without_pw = {k: v for k, v in __import__('os').environ.items()
                         if k != 'OPTIONWHEEL_EMAIL_PASSWORD'}
        with patch.dict('os.environ', env_without_pw, clear=True):
            n = _notifier()
        msg = MagicMock()
        msg.as_string.return_value = 'raw email'
        # Patch smtplib.SMTP in sender's own namespace (most reliable approach)
        with patch('src.notify.sender.smtplib.SMTP') as MockSMTP:
            smtp_instance = MockSMTP.return_value.__enter__.return_value
            n._send(msg)
        smtp_instance.ehlo.assert_called()
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with('bot@example.com', 'secret')
        smtp_instance.sendmail.assert_called_once()


class TestPollImapForReply(unittest.TestCase):

    def _make_imap_mock(self, search_uids=b'', fetch_raw=None):
        """Build a mock IMAP4_SSL context manager that returns controlled data."""
        imap = MagicMock()
        # uid('SEARCH', ...) → (ok, [uid_bytes])
        imap.uid.side_effect = self._imap_uid_side_effect(search_uids, fetch_raw)
        imap.select.return_value = ('OK', [b'1'])
        imap.login.return_value = ('OK', [b'Logged in'])

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=imap)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx, imap

    @staticmethod
    def _imap_uid_side_effect(search_uids, fetch_raw):
        calls = []
        def side_effect(cmd, *args):
            calls.append(cmd)
            if cmd == 'SEARCH':
                return ('OK', [search_uids])
            if cmd == 'FETCH':
                if fetch_raw:
                    return ('OK', [(b'', fetch_raw)])
                return ('OK', [None])
            if cmd == 'STORE':
                return ('OK', [b''])
            return ('OK', [b''])
        return side_effect

    def test_returns_none_when_no_messages(self):
        n = _notifier()
        ctx, _ = self._make_imap_mock(search_uids=b'')
        with patch('imaplib.IMAP4_SSL', return_value=ctx):
            result = n._poll_imap_for_reply('token123', '<msg@optionwheel.local>')
        self.assertIsNone(result)

    def test_returns_body_when_message_found(self):
        n = _notifier()
        # Build a minimal raw email bytes
        raw = b'Subject: reply\r\n\r\nAPPROVE ALL'
        ctx, _ = self._make_imap_mock(search_uids=b'42', fetch_raw=raw)

        _fmt_mock._extract_plain_body.return_value = 'APPROVE ALL'
        with patch('imaplib.IMAP4_SSL', return_value=ctx):
            result = n._poll_imap_for_reply('token123', '<msg@optionwheel.local>')
        self.assertEqual(result, 'APPROVE ALL')


class TestWaitForApproval(unittest.TestCase):

    def test_disabled_returns_none(self):
        n = EmailNotifier({})
        result = n.wait_for_approval('<msg@local>', [{'symbol': 'SPY'}])
        self.assertIsNone(result)

    def test_no_token_returns_none(self):
        n = _notifier()
        with patch('src.notify.sender._extract_token_from_msgid', return_value=None):
            result = n.wait_for_approval('<bad_msg_id>', [{'symbol': 'SPY'}])
        self.assertIsNone(result)

    def test_returns_approved_picks_on_reply(self):
        n = _notifier()
        picks = [{'symbol': 'SPY'}, {'symbol': 'QQQ'}]
        with patch('src.notify.sender._extract_token_from_msgid', return_value='abc123'), \
             patch('src.notify.sender._parse_approval_body', return_value=[picks[0]]), \
             patch.object(n, '_poll_imap_for_reply', return_value='APPROVE 1'):
            result = n.wait_for_approval('<tok.123@optionwheel.local>', picks)
        self.assertEqual(result, [picks[0]])

    def test_returns_replan_string(self):
        n = _notifier()
        picks = [{'symbol': 'SPY'}]
        with patch('src.notify.sender._extract_token_from_msgid', return_value='abc123'), \
             patch('src.notify.sender._parse_approval_body', return_value='REPLAN'), \
             patch.object(n, '_poll_imap_for_reply', return_value='replan'):
            result = n.wait_for_approval('<tok.456@optionwheel.local>', picks)
        self.assertEqual(result, 'REPLAN')

    def test_times_out_returns_none(self):
        n = _notifier()
        n._timeout = 0          # immediate timeout
        n._poll_interval = 1
        with patch.object(n, '_poll_imap_for_reply', return_value=None):
            result = n.wait_for_approval('<tok.789@optionwheel.local>', [])
        self.assertIsNone(result)

    def test_imap_exception_does_not_propagate(self):
        n = _notifier()
        n._timeout = 0
        with patch.object(n, '_poll_imap_for_reply', side_effect=Exception('IMAP down')):
            # Should catch exception internally and eventually time out
            result = n.wait_for_approval('<tok@optionwheel.local>', [])
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
