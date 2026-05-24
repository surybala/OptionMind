"""
EmailNotifier
=============
Gmail SMTP/IMAP notifier for OptionWheel.

Sends trade-execution and position-closed notifications via SMTP.
Sends trade-plan approval requests and polls Gmail IMAP for a reply.

Disabled silently when the 'email' key is absent from config.

Zero new pip dependencies — uses only stdlib:
    smtplib, imaplib, email, secrets, re, time, datetime

Formatting helpers live in ``src/notify/formatter.py``;
this module re-exports them for backward compatibility.
"""
from __future__ import annotations

import imaplib
import logging
import secrets
import smtplib
import time
from datetime import datetime, timezone
from email import message_from_bytes
from email.mime.multipart import MIMEMultipart
from typing import Optional

from src.notify.formatter import (
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
    _render_weekly_digest_text,
    _render_weekly_digest_html,
    _CSS_BASE,
    _RISK_STYLE,
)

_log = logging.getLogger('optionwheel')


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EmailNotifier:
    """
    Sends and receives emails for OptionWheel.

    Parameters
    ----------
    config : dict
        The full config.json dict.  Reads the 'email' sub-key.
        If the key is absent or incomplete, all methods silently no-op
        and ``self.enabled`` is False.
    """

    def __init__(self, config: dict) -> None:
        import os as _os
        cfg = config.get('email') or {}

        # App password: env var takes priority over config.json so the secret
        # never needs to be committed.  Env var name: OPTIONWHEEL_EMAIL_PASSWORD
        password_raw = (
            _os.environ.get('OPTIONWHEEL_EMAIL_PASSWORD')
            or cfg.get('app_password', '')
        )

        self.enabled: bool = bool(
            cfg.get('smtp_host')
            and cfg.get('from_addr')
            and cfg.get('to_addr')
            and password_raw
        )
        if not self.enabled:
            return

        self._smtp_host      = cfg['smtp_host']
        self._smtp_port      = int(cfg.get('smtp_port', 587))
        self._imap_host      = cfg.get('imap_host', 'imap.gmail.com')
        self._imap_port      = int(cfg.get('imap_port', 993))
        self._from_addr      = cfg['from_addr']
        self._to_addr        = cfg['to_addr']
        # Strip spaces — Google shows App Passwords with spaces
        self._password       = password_raw.replace(' ', '')
        self._timeout        = int(cfg.get('approval_timeout_seconds', 21600))
        self._poll_interval  = int(cfg.get('approval_poll_interval_seconds', 15))
        # SMTP retry settings (transient network / rate-limit errors)
        self._smtp_max_retries  = int(cfg.get('smtp_max_retries', 3))
        self._smtp_retry_delay  = float(cfg.get('smtp_retry_delay_seconds', 5.0))

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def send_trade_executed(
        self,
        pick: dict,
        order_id: str,
        reason: str = 'TRADE_PLAN',
    ) -> None:
        """Send a notification that a new position was opened."""
        if not self.enabled:
            return
        try:
            strat  = pick.get('strategy', '?')
            symbol = pick.get('symbol', '?')
            expiry = pick.get('expiry', '?')
            subject = (
                f"[OptionWheel] Trade Executed \u2014 "
                f"{strat} {symbol} {expiry}"
            )
            text = _render_trade_executed_text(pick, order_id, reason)
            html = _render_trade_executed_html(pick, order_id, reason)
            self._send(_build_mime(self._from_addr, self._to_addr, subject, text, html))
            _log.info("[notifier] Trade-executed email sent: %s %s", strat, symbol)
        except Exception as exc:
            _log.warning("[notifier] send_trade_executed failed: %s", exc)

    def send_position_closed(
        self,
        pos: dict,
        reason_tag: str,
    ) -> None:
        """
        Send a notification that a position was closed by the monitor.

        ``pos`` should contain the standard DB fields plus any risk
        metrics written back by ``_check_position`` (entry_premium,
        current_mark, pnl_per_share, ratio, short_delta, risk_score, dte).
        """
        if not self.enabled:
            return
        try:
            strat  = pos.get('type', '?')
            symbol = pos.get('symbol', '?')
            expiry = pos.get('expiry', '?')
            pnl    = float(pos.get('close_pnl') or 0)
            sign   = '+' if pnl >= 0 else ''
            subject = (
                f"[OptionWheel] Position Closed ({reason_tag}) \u2014 "
                f"{strat} {symbol} {expiry}  P&L: {sign}${pnl:,.2f}"
            )
            text = _render_position_closed_text(pos, reason_tag)
            html = _render_position_closed_html(pos, reason_tag)
            self._send(_build_mime(self._from_addr, self._to_addr, subject, text, html))
            _log.info("[notifier] Position-closed email sent: %s %s (%s)",
                      strat, symbol, reason_tag)
        except Exception as exc:
            _log.warning("[notifier] send_position_closed failed: %s", exc)

    def send_trade_plan(
        self,
        picks: list[dict],
        capital_budget: Optional[float] = None,
        deployed_capital: float = 0.0,
    ) -> Optional[str]:
        """
        Email the trade plan and return the Message-ID for reply matching.

        The token is embedded in both the Message-ID and the Subject so
        ``wait_for_approval`` can locate the reply regardless of whether
        the email client preserves the In-Reply-To header.

        Returns the Message-ID string on success, None on failure.
        """
        if not self.enabled:
            return None
        try:
            token   = secrets.token_hex(8)          # 16 hex chars
            ts      = int(time.time())
            msg_id  = f"<{token}.{ts}@optionwheel.local>"
            subject = (
                f"[OptionWheel] Trade Plan Approval Required "
                f"[token:{token}]"
            )
            text = _render_trade_plan_text(picks, capital_budget, token,
                                           self._timeout, self._poll_interval,
                                           deployed_capital=deployed_capital)
            html = _render_trade_plan_html(picks, capital_budget, token,
                                           self._timeout, self._poll_interval,
                                           deployed_capital=deployed_capital)
            self._send(_build_mime(self._from_addr, self._to_addr, subject,
                                   text, html, message_id=msg_id))
            _log.info("[notifier] Trade plan email sent (token=%s, %d picks)",
                      token, len(picks))
            return msg_id
        except Exception as exc:
            _log.warning("[notifier] send_trade_plan failed: %s", exc)
            return None

    def wait_for_approval(
        self,
        message_id: str,
        picks: list[dict],
    ) -> 'Optional[list[dict] | str]':
        """
        Poll Gmail IMAP for a reply to the trade plan email.

        Returns:
          - Approved subset of picks (list) on a normal approval reply.
          - The string ``'REPLAN'`` if the user replied with replan/rescan.
          - ``None`` on timeout / IMAP failure (caller falls back to TTY).
        """
        if not self.enabled:
            return None

        token = _extract_token_from_msgid(message_id)
        if not token:
            _log.warning("[notifier] Cannot extract token from message_id — skipping email approval")
            return None

        deadline = time.monotonic() + self._timeout
        _log.info(
            "[notifier] Waiting up to %ds for email approval reply "
            "(token=%s, polling every %ds)...",
            self._timeout, token, self._poll_interval,
        )

        while time.monotonic() < deadline:
            try:
                body = self._poll_imap_for_reply(token, message_id)
                if body is not None:
                    result = _parse_approval_body(body, picks)
                    if result == 'REPLAN':
                        _log.info("[notifier] Email approval: REPLAN requested.")
                    else:
                        _log.info("[notifier] Email approval received — %d/%d picks approved",
                                  len(result), len(picks))
                    return result
            except Exception as exc:
                _log.warning("[notifier] IMAP poll error: %s", exc)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._poll_interval, remaining))

        _log.info("[notifier] Email approval timeout — caller should fall back to TTY.")
        return None

    def send_daily_risk_report(
        self,
        positions: list[dict],
        closed_today: Optional[list[dict]] = None,
    ) -> bool:
        """
        Send end-of-day risk summary email for all open positions.

        Each position should be pre-enriched by PositionMonitor.get_risk_snapshot()
        with keys: current_mark, pnl_dollars, stop_proximity_pct, profit_captured_pct,
        gamma_theta_ratio, net_short_delta, risk_score, dte, risk_level.

        ``closed_today`` is an optional list of positions that were closed by
        stop-loss triggers during the day.  They are rendered in a separate
        section so the user can understand why each closure was triggered.

        Returns True on successful send, False otherwise.
        """
        if not self.enabled:
            return False
        closed_today = closed_today or []
        if not positions and not closed_today:
            _log.info("[notifier] send_daily_risk_report: no positions, skipping.")
            return False

        today   = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        n       = len(positions)
        counts: dict[str, int] = {}
        for p in positions:
            lvl = p.get('risk_level', 'SAFE')
            counts[lvl] = counts.get(lvl, 0) + 1

        subject_parts = [f"OptionWheel — Daily Risk Report — {today} — {n} position{'s' if n != 1 else ''}"]
        if counts.get('CRITICAL', 0):
            subject_parts.append(f"⚠ {counts['CRITICAL']} CRITICAL")
        if counts.get('CAUTION', 0):
            subject_parts.append(f"{counts['CAUTION']} CAUTION")
        if closed_today:
            subject_parts.append(f"🔔 {len(closed_today)} CLOSED TODAY")
        subject = '  |  '.join(subject_parts)

        text = _render_daily_risk_text(positions, today, closed_today=closed_today)
        html = _render_daily_risk_html(positions, today, closed_today=closed_today)
        msg  = _build_mime(self._from_addr, self._to_addr, subject, text, html)
        try:
            self._send(msg)
            _log.info("[notifier] Daily risk report sent (%d open, %d closed today).",
                      n, len(closed_today))
            return True
        except Exception as exc:
            _log.error("[notifier] Failed to send daily risk report: %s", exc)
            return False

    def send_weekly_digest(
        self,
        week_start: str,
        week_end: str,
        weekly_trades: list[dict],
        cumulative_pnl: float,
        open_positions: list[dict],
        capital_deployed: float,
    ) -> bool:
        """
        Send a weekly digest email summarising the week's P&L, capital deployed,
        cumulative P&L, and open-position risk breakdown.

        Parameters
        ----------
        week_start / week_end : str   ISO date strings (YYYY-MM-DD)
        weekly_trades         : list  Closed trade rows for the week (from DB)
        cumulative_pnl        : float All-time realised P&L (sum of all closed trades)
        open_positions        : list  Pre-enriched risk snapshot from get_risk_snapshot()
        capital_deployed      : float Sum of max-loss / capital requirement for open positions

        Returns True on successful send, False otherwise.
        """
        if not self.enabled:
            return False
        weekly_pnl = sum(float(t.get('pnl') or 0) for t in weekly_trades)
        sign = '+' if weekly_pnl >= 0 else ''
        subject = (
            f"[OptionWheel] Weekly Digest — {week_start} to {week_end} — "
            f"P&L: {sign}${weekly_pnl:,.2f}  |  Cumulative: "
            f"{'+'if cumulative_pnl >= 0 else ''}${cumulative_pnl:,.2f}"
        )
        text = _render_weekly_digest_text(
            week_start, week_end, weekly_trades,
            cumulative_pnl, open_positions, capital_deployed,
        )
        html = _render_weekly_digest_html(
            week_start, week_end, weekly_trades,
            cumulative_pnl, open_positions, capital_deployed,
        )
        msg = _build_mime(self._from_addr, self._to_addr, subject, text, html)
        try:
            self._send(msg)
            _log.info(
                "[notifier] Weekly digest sent (%d closed trades, P&L %s$%.2f, "
                "%d open positions).",
                len(weekly_trades), sign, weekly_pnl, len(open_positions),
            )
            return True
        except Exception as exc:
            _log.error("[notifier] Failed to send weekly digest: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _send(self, msg: MIMEMultipart) -> None:
        """
        Deliver a pre-built MIME message via SMTP (STARTTLS) with retries.

        Retries up to ``smtp_max_retries`` times (default 3) on transient
        failures (network errors, SMTP 4xx, connection reset, timeouts) using
        exponential back-off starting at ``smtp_retry_delay_seconds`` (default
        5 s).  Permanent auth failures (SMTP 5xx / SMTPAuthenticationError)
        are re-raised immediately without retrying.

        Raises the last exception after all retries are exhausted so callers
        can log it at the appropriate level.
        """
        last_exc: Exception
        for attempt in range(self._smtp_max_retries + 1):
            try:
                with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=30) as smtp:
                    smtp.ehlo()
                    smtp.starttls()
                    smtp.ehlo()
                    smtp.login(self._from_addr, self._password)
                    smtp.sendmail(self._from_addr, self._to_addr, msg.as_string())
                return   # success
            except smtplib.SMTPAuthenticationError:
                # Wrong credentials — retrying won't help.
                raise
            except (smtplib.SMTPException, OSError, ConnectionError) as exc:
                last_exc = exc
                if attempt == self._smtp_max_retries:
                    break
                delay = self._smtp_retry_delay * (2 ** attempt)
                _log.warning(
                    "[notifier] SMTP send failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self._smtp_max_retries, delay, exc,
                )
                time.sleep(delay)
        raise last_exc

    def _poll_imap_for_reply(
        self,
        token: str,
        original_msgid: str,
    ) -> Optional[str]:
        """
        Open an IMAP connection, search for a matching reply, return its
        plain-text body.  Returns None if nothing found yet.

        Two searches are run and unioned:
        1. UNSEEN SUBJECT containing the token string
        2. UNSEEN HEADER In-Reply-To matching original_msgid

        The matched message is marked \\Seen immediately to prevent
        double-processing on the next poll.
        """
        with imaplib.IMAP4_SSL(self._imap_host, self._imap_port) as imap:
            imap.login(self._from_addr, self._password)
            imap.select('INBOX')

            uids: set[bytes] = set()

            _, data = imap.uid('SEARCH', None,
                               f'UNSEEN SUBJECT "[token:{token}]"'.encode())
            if data and data[0]:
                uids.update(uid for uid in data[0].split() if uid)

            _, data = imap.uid('SEARCH', None,
                               f'UNSEEN HEADER In-Reply-To {original_msgid}'.encode())
            if data and data[0]:
                uids.update(uid for uid in data[0].split() if uid)

            if not uids:
                return None

            # Take the highest (most recent) UID
            uid = sorted(uids, key=lambda x: int(x))[-1]

            _, msg_data = imap.uid('FETCH', uid, '(RFC822)')
            if not msg_data or not msg_data[0]:
                return None

            raw = msg_data[0][1]
            imap.uid('STORE', uid, '+FLAGS', '(\\Seen)')

            return _extract_plain_body(message_from_bytes(raw))
