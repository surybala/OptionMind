"""
notify.formatter
================

Pure formatting / rendering helpers for OptionWheel email notifications.

All functions are stateless — no network calls, no SMTP/IMAP.
``EmailNotifier`` (in ``sender.py``) calls these to build message bodies.
"""
from __future__ import annotations

import json as _json
import re
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional


# ---------------------------------------------------------------------------
# MIME builder
# ---------------------------------------------------------------------------

def _build_mime(
    from_addr: str,
    to_addr: str,
    subject: str,
    text_body: str,
    html_body: str,
    message_id: Optional[str] = None,
) -> MIMEMultipart:
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = from_addr
    msg['To']      = to_addr
    msg['Date']    = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S +0000')
    if message_id:
        msg['Message-ID'] = message_id
    msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
    msg.attach(MIMEText(html_body, 'html',  'utf-8'))
    return msg


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _extract_token_from_msgid(message_id: str) -> Optional[str]:
    """
    Recover the 16-char hex token from a Message-ID created by send_trade_plan.
    Format: ``<TOKEN16HEX.timestamp@optionwheel.local>``
    """
    m = re.search(r'<([0-9a-f]{16})\.', message_id)
    return m.group(1) if m else None


def _extract_plain_body(msg) -> str:
    """Walk a parsed email and return the first text/plain part."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                charset = part.get_content_charset() or 'utf-8'
                return part.get_payload(decode=True).decode(charset, errors='replace')
    else:
        charset = msg.get_content_charset() or 'utf-8'
        return msg.get_payload(decode=True).decode(charset, errors='replace')
    return ''


# ---------------------------------------------------------------------------
# Reply parser
# ---------------------------------------------------------------------------

def _parse_approval_body(body: str, picks: list[dict]):
    """
    Parse an email reply body using the same logic as ``_approval_gate``.

    Strips quoted lines (starting with '>') and blank lines, then uses the
    first remaining line as the selection input.

    Supports:
      ``a``/``all``           — approve all picks
      ``n``/``none``          — reject all
      ``1,3``/``1-5``         — numeric range selection
      ``replan``/``rescan``   — discard plan and rescan with fresh prices

    Returns the approved subset of picks (empty list means none approved),
    or the sentinel string ``'REPLAN'`` to request a fresh scan.
    """
    lines = body.splitlines()
    active = [ln.strip() for ln in lines
              if ln.strip() and not ln.strip().startswith('>')]
    if not active:
        return []

    raw = active[0].lower().strip()

    if raw in ('replan', 'rescan', 'retry'):
        return 'REPLAN'
    if raw in ('n', 'none'):
        return []
    if raw in ('a', 'all'):
        return list(picks)

    indices: set[int] = set()
    for part in raw.split(','):
        part = part.strip()
        if '-' in part:
            lo_s, _, hi_s = part.partition('-')
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
                indices.update(range(lo, hi + 1))
            except ValueError:
                pass
        else:
            try:
                indices.add(int(part))
            except ValueError:
                pass

    valid = sorted(n for n in indices if 1 <= n <= len(picks))
    return [picks[n - 1] for n in valid]


# ---------------------------------------------------------------------------
# Leg string helpers
# ---------------------------------------------------------------------------

def _legs_str(pick: dict) -> str:
    """Return a compact human-readable description of the option legs."""
    strat = pick.get('strategy', '')
    if strat in ('CSP', 'CC'):
        return f"{pick.get('short_strike', '?')}{'P' if strat == 'CSP' else 'C'}"
    if strat in ('PCS', 'CCS'):
        s = pick.get('short_strike', '?')
        l = pick.get('long_strike', '?')
        side = 'P' if strat == 'PCS' else 'C'
        return f"{s}/{l} {side}"
    if strat == 'STRANGLE':
        return f"{pick.get('short_put','?')}P / {pick.get('short_call','?')}C"
    if strat in ('IC', 'IFLY'):
        return (f"{pick.get('long_put','?')}/{pick.get('short_put','?')}P "
                f"{pick.get('short_call','?')}/{pick.get('long_call','?')}C")
    return '—'


def _pos_legs_str(pos: dict) -> str:
    """Compact legs description for a DB position row (uses 'type' + parsed legs JSON)."""
    strat = pos.get('type', '')
    raw   = pos.get('legs')
    legs: dict = {}
    if isinstance(raw, dict):
        legs = raw
    elif raw:
        try:
            legs = _json.loads(raw)
        except Exception:
            pass
    if strat in ('CSP', 'CC'):
        ss = legs.get('short_strike') or pos.get('strike', '?')
        return f"{ss}{'P' if strat == 'CSP' else 'C'}"
    if strat in ('PCS', 'CCS'):
        ss   = legs.get('short_strike') or legs.get('short_put') or legs.get('short_call') or pos.get('strike', '?')
        ls   = legs.get('long_strike')  or legs.get('long_put')  or legs.get('long_call')  or '?'
        side = 'P' if strat == 'PCS' else 'C'
        return f"{ss}/{ls} {side}"
    if strat == 'STRANGLE':
        return f"{legs.get('short_put','?')}P / {legs.get('short_call','?')}C"
    if strat in ('IC', 'IFLY'):
        return (f"{legs.get('long_put','?')}/{legs.get('short_put','?')}P "
                f"{legs.get('short_call','?')}/{legs.get('long_call','?')}C")
    return '—'


# ---------------------------------------------------------------------------
# Numeric formatter
# ---------------------------------------------------------------------------

def _fmt_opt(v, fmt: str = '.2f', prefix: str = '', suffix: str = '') -> str:
    """Format a numeric value or return '—' if None."""
    if v is None:
        return '—'
    try:
        return f"{prefix}{float(v):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return '—'


def _pnl_source_label(row: dict) -> str:
    source = row.get('pnl_source') or 'UNKNOWN'
    verified = bool(row.get('pnl_verified'))
    label = source.replace('_', ' ').title()
    return label if verified else f"{label} (unverified)"


# ---------------------------------------------------------------------------
# Text renderers
# ---------------------------------------------------------------------------

def _render_trade_executed_text(pick: dict, order_id: str, reason: str) -> str:
    strat     = pick.get('strategy', '?')
    symbol    = pick.get('symbol', '?')
    expiry    = pick.get('expiry', '?')
    price     = pick.get('current_price', 0.0)
    premium   = float(pick.get('premium', 0))
    qty       = int(pick.get('quantity', 1))
    prob      = float(pick.get('prob_win', 0))
    roi       = float(pick.get('roi', 0))
    score     = float(pick.get('score', 0))
    legs      = _legs_str(pick)
    ts        = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return (
        f"OptionWheel — Trade Executed\n"
        f"{'=' * 40}\n\n"
        f"Reason:     {reason}\n"
        f"Strategy:   {strat}\n"
        f"Symbol:     {symbol}  (spot ${price:.2f})\n"
        f"Expiry:     {expiry}\n"
        f"Legs:       {legs}\n"
        f"Contracts:  {qty}\n"
        f"Credit:     ${premium * 100:.2f}/contract  (${premium * 100 * qty:.2f} total)\n"
        f"Prob Win:   {prob:.1%}\n"
        f"ROI:        {roi:.1%}\n"
        f"Score:      {score:.4f}\n"
        f"Order ID:   {order_id}\n"
        f"Timestamp:  {ts} UTC\n"
    )


def _render_position_closed_text(pos: dict, reason_tag: str) -> str:
    strat    = pos.get('type', '?')
    symbol   = pos.get('symbol', '?')
    expiry   = pos.get('expiry', '?')
    pnl      = float(pos.get('close_pnl') or 0)
    sign     = '+' if pnl >= 0 else ''
    ep       = float(pos.get('entry_premium') or 0)
    cm       = float(pos.get('current_mark') or 0)
    pps      = float(pos.get('pnl_per_share') or 0)
    order_id = pos.get('close_order_id', '—')
    pnl_source = _pnl_source_label(pos)
    ts       = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        f"OptionWheel — Position Closed ({reason_tag})",
        '=' * 40,
        '',
        f"Strategy:     {strat}",
        f"Symbol:       {symbol}",
        f"Expiry:       {expiry}",
        f"P&L:          {sign}${pnl:,.2f}",
        f"Entry:        ${ep * 100:.2f}/contract",
        f"Mark @ close: ${cm * 100:.2f}/contract",
        f"P&L:          {'+' if pps >= 0 else ''}{pps * 100:.2f}/contract",
        f"P&L Source:   {pnl_source}",
        f"Order ID:     {order_id}",
        f"Timestamp:    {ts} UTC",
    ]

    if reason_tag == 'PROFIT_TAKE':
        captured = pos.get('profit_captured_pct')
        target   = pos.get('profit_take_pct')
        lines += [
            '',
            'Profit Capture',
            '-' * 30,
            f"Captured: {captured:.1f}%" if captured is not None else "Captured: n/a",
            f"Target:   {target * 100:.0f}%" if target is not None else "Target:   n/a",
        ]
    elif reason_tag in ('STOP_LOSS', 'GAMMA_RISK'):
        ratio      = pos.get('ratio')
        short_delta= pos.get('short_delta')
        risk_score = pos.get('risk_score')
        dte        = pos.get('dte')
        lines += [
            '',
            'Risk Metrics at Close',
            '-' * 30,
            f"Gamma/Theta ratio: {ratio:.4f}"     if ratio      is not None else "Gamma/Theta ratio: n/a",
            f"Short delta:       {short_delta:.4f}" if short_delta is not None else "Short delta:       n/a",
            f"Risk score:        {risk_score:.4f}" if risk_score  is not None else "Risk score:        n/a",
            f"DTE at close:      {dte}d"           if dte         is not None else "DTE at close:      n/a",
        ]

    return '\n'.join(lines) + '\n'


def _render_trade_plan_text(
    picks: list[dict],
    capital_budget: Optional[float],
    token: str,
    timeout: int,
    poll_interval: int,
    deployed_capital: float = 0.0,
) -> str:
    lines = [
        'OptionWheel — Trade Plan Approval Required',
        '=' * 40,
        '',
        'Reply to this email with one of:',
        '  a  or  all       — approve all picks',
        '  n  or  none      — reject all',
        '  1,3              — approve picks 1 and 3',
        '  1-5              — approve picks 1 through 5',
        '  2,4-7            — mixed ranges',
        '  replan           — discard this plan, rescan with current prices',
        '',
        f'Your reply will be checked within {poll_interval}s.',
        f'Timeout in {timeout}s — TTY prompt used if no reply received.',
        '',
        f"{'#':<3} {'Strategy':<10} {'Symbol':<7} {'Spot':>7} {'Expiry':<12} "
        f"{'Legs':<22} {'Credit':>8} {'Capital':>9} {'Prob':>6} {'ROI':>6} {'Score':>7}",
        '-' * 108,
    ]
    total_cap = 0.0
    for i, p in enumerate(picks, 1):
        strat   = p.get('strategy', '?')
        symbol  = p.get('symbol', '?')
        spot    = p.get('current_price')
        spot_s  = f"${float(spot):>6.2f}" if spot is not None else '      ?'
        expiry  = p.get('expiry', '?')
        legs    = _legs_str(p)
        premium = float(p.get('premium', 0))
        cap     = float(p.get('capital') or 0)
        total_cap += cap
        prob    = float(p.get('prob_win', 0))
        roi     = float(p.get('roi', 0))
        score   = float(p.get('score', 0))
        lines.append(
            f"{i:<3} {strat:<10} {symbol:<7} {spot_s} {expiry:<12} "
            f"{legs:<22} ${premium * 100:>7.2f}  ${cap:>8,.0f} {prob:>5.0%} {roi:>5.0%} {score:>7.4f}"
        )
    lines += [
        '-' * 108,
        f"{'TOTAL':<65} ${total_cap:>8,.0f}",
    ]
    if capital_budget is not None:
        remaining = capital_budget - deployed_capital - total_cap
        parts = [f"Capital budget: ${capital_budget:,.0f}"]
        if deployed_capital > 0:
            parts.append(f"Already deployed: ${deployed_capital:,.0f}")
        parts.append(f"New picks: ${total_cap:,.0f}")
        parts.append(f"Remaining: ${remaining:,.0f}")
        lines.append("  |  ".join(parts))
    lines += ['', f'Token: {token}']
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# CSS base style
# ---------------------------------------------------------------------------

_CSS_BASE = """
body{margin:0;padding:20px;background:#f4f4f4;
     font-family:system-ui,-apple-system,Arial,sans-serif}
.card{max-width:680px;margin:0 auto;background:#fff;border-radius:8px;
      overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.hdr{color:#fff;padding:20px 24px}
.hdr h1{margin:0;font-size:20px}
.hdr p{margin:4px 0 0;opacity:.8;font-size:13px}
.body{padding:24px}
.banner{border-left:4px solid;padding:12px 16px;border-radius:4px;margin-bottom:20px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{padding:10px 12px;text-align:left;background:#2c3e50;color:#fff}
td{padding:9px 12px;border-bottom:1px solid #eee}
tr:nth-child(even) td{background:#f9f9f9}
.footer{font-size:12px;color:#999;margin-top:20px}
code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:13px}
"""


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _render_trade_executed_html(pick: dict, order_id: str, reason: str) -> str:
    strat   = pick.get('strategy', '?')
    symbol  = pick.get('symbol', '?')
    expiry  = pick.get('expiry', '?')
    price   = pick.get('current_price', 0.0)
    premium = float(pick.get('premium', 0))
    qty     = int(pick.get('quantity', 1))
    prob    = float(pick.get('prob_win', 0))
    roi     = float(pick.get('roi', 0))
    score   = float(pick.get('score', 0))
    legs    = _legs_str(pick)
    ts      = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    rows = [
        ('Strategy',   f'<strong>{strat}</strong>'),
        ('Symbol',     f'{symbol}  <span style="color:#666;font-size:12px">spot ${float(price):.2f}</span>'),
        ('Expiry',     expiry),
        ('Legs',       legs),
        ('Contracts',  f'<strong>{qty}</strong>'),
        ('Credit',     f'${premium * 100:.2f}/contract &nbsp; (<strong>${premium * 100 * qty:.2f} total</strong>)'),
        ('Prob Win',   f'{prob:.1%}'),
        ('ROI',        f'{roi:.1%}'),
        ('Score',      f'{score:.4f}'),
        ('Reason',     reason),
        ('Order ID',   f'<code>{order_id}</code>'),
    ]

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>{_CSS_BASE}</style></head>
<body>
<div class="card">
  <div class="hdr" style="background:#1a1a2e">
    <h1>Trade Executed</h1>
    <p>OptionWheel Notification</p>
  </div>
  <div class="body">
    <div class="banner" style="background:#eaf7ee;border-color:#27ae60">
      <strong style="color:#27ae60">&#10003; SUCCESS</strong>
      &mdash; {strat} {symbol} order filled
      <br><span style="font-size:12px;color:#666">
        Reason: {reason} &nbsp;|&nbsp; Order: {order_id}
      </span>
    </div>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)}
    </table>
    <p class="footer">Sent by OptionWheel at {ts} UTC</p>
  </div>
</div>
</body></html>"""


def _render_position_closed_html(pos: dict, reason_tag: str) -> str:
    strat    = pos.get('type', '?')
    symbol   = pos.get('symbol', '?')
    expiry   = pos.get('expiry', '?')
    pnl      = float(pos.get('close_pnl') or 0)
    sign     = '+' if pnl >= 0 else ''
    ep       = float(pos.get('entry_premium') or 0)
    cm       = float(pos.get('current_mark') or 0)
    pps      = float(pos.get('pnl_per_share') or 0)
    order_id = pos.get('close_order_id', '—')
    pnl_source = _pnl_source_label(pos)
    ts       = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    hdr_color   = {'STOP_LOSS': '#c0392b', 'GAMMA_RISK': '#d35400', 'PROFIT_TAKE': '#27ae60'}.get(reason_tag, '#2980b9')
    pnl_bg      = '#eaf7ee' if pnl >= 0 else '#fdf2f2'
    pnl_border  = '#27ae60' if pnl >= 0 else '#e74c3c'
    pnl_color   = '#27ae60' if pnl >= 0 else '#e74c3c'

    main_rows = [
        ('Strategy',      f'<strong>{strat}</strong>'),
        ('Symbol',        symbol),
        ('Expiry',        expiry),
        ('Entry premium', f'${ep * 100:.2f}/contract'),
        ('Mark at close', f'${cm * 100:.2f}/contract'),
        ('P&L',           f'<span style="color:{pnl_color}">{("+" if pps >= 0 else "")}${pps * 100:.2f}/contract</span>'),
        ('P&L source',    pnl_source),
        ('Order ID',      f'<code>{order_id}</code>'),
    ]

    risk_table = ''
    if reason_tag == 'PROFIT_TAKE':
        captured = pos.get('profit_captured_pct')
        target   = pos.get('profit_take_pct')
        pt_rows = [
            ('Captured', f'{captured:.1f}%' if captured is not None else '<span style="color:#aaa">n/a</span>'),
            ('Target',   f'{target * 100:.0f}%' if target is not None else '<span style="color:#aaa">n/a</span>'),
        ]
        risk_table = f"""
    <h3 style="color:#27ae60;margin:24px 0 8px">Profit Capture</h3>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pt_rows)}
    </table>"""
    elif reason_tag in ('STOP_LOSS', 'GAMMA_RISK'):
        ratio      = pos.get('ratio')
        short_delta= pos.get('short_delta')
        risk_score = pos.get('risk_score')
        dte        = pos.get('dte')

        def _fmt(v, fmt='.4f'):
            return f'{v:{fmt}}' if v is not None else '<span style="color:#aaa">n/a</span>'

        risk_rows = [
            ('Gamma / Theta ratio', _fmt(ratio)),
            ('Short delta',         _fmt(short_delta)),
            ('Risk score',          _fmt(risk_score)),
            ('DTE at close',        f'{dte}d' if dte is not None else '<span style="color:#aaa">n/a</span>'),
        ]
        risk_table = f"""
    <h3 style="color:#2c3e50;margin:24px 0 8px">Risk Metrics at Close</h3>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in risk_rows)}
    </table>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>{_CSS_BASE}</style></head>
<body>
<div class="card">
  <div class="hdr" style="background:{hdr_color}">
    <h1>Position Closed &mdash; {reason_tag}</h1>
    <p>OptionWheel Notification</p>
  </div>
  <div class="body">
    <div class="banner" style="background:{pnl_bg};border-color:{pnl_border}">
      <strong>P&amp;L: {sign}${pnl:,.2f}</strong>
      <br><span style="font-size:12px;color:#666">
        Entry: ${ep * 100:.2f}/contract &nbsp;|&nbsp; Mark at close: ${cm * 100:.2f}/contract
      </span>
    </div>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in main_rows)}
    </table>
    {risk_table}
    <p class="footer">Sent by OptionWheel at {ts} UTC</p>
  </div>
</div>
</body></html>"""


def _render_trade_plan_html(
    picks: list[dict],
    capital_budget: Optional[float],
    token: str,
    timeout: int,
    poll_interval: int,
    deployed_capital: float = 0.0,
) -> str:
    ts        = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    n         = len(picks)
    total_cap = sum(float(p.get('capital') or 0) for p in picks)

    rows_html = ''
    for i, p in enumerate(picks, 1):
        strat   = p.get('strategy', '?')
        symbol  = p.get('symbol', '?')
        spot    = p.get('current_price')
        spot_s  = f"${float(spot):.2f}" if spot is not None else '—'
        expiry  = p.get('expiry', '?')
        legs    = _legs_str(p)
        premium = float(p.get('premium', 0))
        cap     = float(p.get('capital') or 0)
        prob    = float(p.get('prob_win', 0))
        roi     = float(p.get('roi', 0))
        score   = float(p.get('score', 0))
        rows_html += (
            f"<tr>"
            f"<td style='font-weight:bold;color:#2980b9'>{i}</td>"
            f"<td>{strat}</td><td>{symbol}</td>"
            f"<td style='color:#666;font-size:12px'>{spot_s}</td>"
            f"<td>{expiry}</td>"
            f"<td style='font-family:monospace'>{legs}</td>"
            f"<td>${premium * 100:.2f}</td>"
            f"<td>${cap:,.0f}</td>"
            f"<td>{prob:.0%}</td>"
            f"<td>{roi:.0%}</td>"
            f"<td>{score:.4f}</td>"
            f"</tr>"
        )

    budget_note = ''
    if capital_budget is not None:
        remaining = capital_budget - deployed_capital - total_cap
        deployed_part = (
            f"Already deployed: <strong>${deployed_capital:,.0f}</strong> &nbsp;|&nbsp; "
            if deployed_capital > 0 else ""
        )
        budget_note = (
            f"<p style='font-size:13px;color:#666;margin-top:8px'>"
            f"Capital budget: <strong>${capital_budget:,.0f}</strong> &nbsp;|&nbsp; "
            f"{deployed_part}"
            f"New picks: <strong>${total_cap:,.0f}</strong> &nbsp;|&nbsp; "
            f"Remaining: <strong>${remaining:,.0f}</strong>"
            f"</p>"
        )

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
{_CSS_BASE}
.instr{{background:#fff8e1;border:1px solid #ffc107;border-radius:6px;
        padding:16px;margin-bottom:24px}}
.instr h3{{margin:0 0 8px;color:#856404;font-size:15px}}
.instr ul{{margin:6px 0 0;padding-left:20px;font-size:14px;line-height:1.9}}
.instr p{{margin:8px 0 0;font-size:12px;color:#856404}}
th,td{{font-size:13px}}
tfoot td{{background:#ecf0f1;font-weight:bold}}
</style>
</head>
<body>
<div class="card">
  <div class="hdr" style="background:#1a1a2e">
    <h1>Trade Plan &mdash; Approval Required</h1>
    <p>{n} pick{'s' if n != 1 else ''} &nbsp;|&nbsp; {ts} UTC</p>
  </div>
  <div class="body">
    <div class="instr">
      <h3>&#9993; How to Approve</h3>
      <ul>
        <li><code>a</code> or <code>all</code> &mdash; approve all {n} picks</li>
        <li><code>n</code> or <code>none</code> &mdash; reject all</li>
        <li><code>1,3</code> &mdash; approve picks 1 and 3</li>
        <li><code>1-5</code> &mdash; approve picks 1 through 5</li>
        <li><code>2,4-7</code> &mdash; mixed ranges</li>
        <li><code>replan</code> &mdash; discard this plan and rescan with fresh prices</li>
      </ul>
      <p>Reply will be checked within {poll_interval}s.
         Timeout in {timeout}s &mdash; TTY prompt used if no reply.</p>
    </div>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Strategy</th><th>Symbol</th><th>Spot</th><th>Expiry</th>
          <th>Legs</th><th>Credit</th><th>Capital</th>
          <th>Prob</th><th>ROI</th><th>Score</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
      <tfoot>
        <tr>
          <td colspan="7">TOTAL</td>
          <td>${total_cap:,.0f}</td>
          <td colspan="3"></td>
        </tr>
      </tfoot>
    </table>
    {budget_note}
    <p class="footer">Token: {token} &nbsp;|&nbsp; Generated: {ts} UTC</p>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Weekly digest renderers
# ---------------------------------------------------------------------------

def _render_weekly_digest_text(
    week_start: str,
    week_end: str,
    weekly_trades: list[dict],
    cumulative_pnl: float,
    open_positions: list[dict],
    capital_deployed: float,
) -> str:
    """Plain-text weekly digest: P&L summary, capital, risk breakdown."""
    wins   = [t for t in weekly_trades if (t.get('pnl') or 0) > 0]
    losses = [t for t in weekly_trades if (t.get('pnl') or 0) < 0]
    weekly_pnl = sum(float(t.get('pnl') or 0) for t in weekly_trades)
    verified_pnl = sum(float(t.get('pnl') or 0) for t in weekly_trades if t.get('pnl_verified'))
    unverified_trades = [t for t in weekly_trades if not t.get('pnl_verified')]
    win_rate   = (len(wins) / len(weekly_trades) * 100) if weekly_trades else 0.0
    best  = max((float(t.get('pnl') or 0) for t in weekly_trades), default=0.0)
    worst = min((float(t.get('pnl') or 0) for t in weekly_trades), default=0.0)

    risk_counts: dict[str, int] = {}
    for p in open_positions:
        lvl = p.get('risk_level', 'SAFE')
        risk_counts[lvl] = risk_counts.get(lvl, 0) + 1

    weekly_sign = '+' if weekly_pnl >= 0 else ''
    cum_sign    = '+' if cumulative_pnl >= 0 else ''
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    lines = [
        f"OptionWheel — Weekly Digest — {week_start} to {week_end}",
        '=' * 70,
        '',
        '── P&L Summary ──────────────────────────────────────────────────────',
        f"  Week P&L:          {weekly_sign}${weekly_pnl:>10,.2f}",
        f"  Verified P&L:      ${verified_pnl:>10,.2f}",
        f"  Unverified rows:   {len(unverified_trades)}",
        f"  Cumulative P&L:    {cum_sign}${cumulative_pnl:>10,.2f}",
        f"  Closed this week:  {len(weekly_trades)} trade(s)  "
            f"({len(wins)} wins / {len(losses)} losses)  win rate {win_rate:.0f}%",
        f"  Best trade:        +${best:,.2f}"  if best  else "  Best trade:        —",
        f"  Worst trade:        ${worst:,.2f}" if worst else "  Worst trade:       —",
        '',
        '── Capital ──────────────────────────────────────────────────────────',
        f"  Deployed capital:  ${capital_deployed:>10,.0f}",
        f"  Open positions:    {len(open_positions)}",
        '',
        '── Open Position Risk Breakdown ─────────────────────────────────────',
    ]
    for lvl, emoji in [('SAFE', '🟢'), ('WATCH', '🟡'), ('CAUTION', '🟠'), ('CRITICAL', '🔴')]:
        n = risk_counts.get(lvl, 0)
        if n:
            lines.append(f"  {emoji} {lvl:<9} {n}")
    if not any(risk_counts.values()):
        lines.append("  No open positions.")

    if open_positions:
        lines += [
            '',
            f"{'#':<3} {'Risk':<9} {'Strat':<7} {'Symbol':<7} {'Expiry':<12} "
            f"{'DTE':>4} {'Qty':>4} {'Entry':>7} {'P&L$':>9}",
            '-' * 70,
        ]
        for i, p in enumerate(open_positions, 1):
            lvl    = p.get('risk_level', 'SAFE')
            emoji  = _RISK_STYLE.get(lvl, _RISK_STYLE['SAFE'])[3]
            strat  = p.get('type', '?')
            symbol = p.get('symbol', '?')
            expiry = p.get('expiry', '?')
            dte    = f"{p['dte']}d" if p.get('dte') is not None else '—'
            qty    = int(p.get('contracts') or 1)
            entry  = _fmt_opt((p.get('premium') or 0) * 100, '.2f', '$')
            pnl    = _fmt_opt(p.get('pnl_dollars'), '+,.2f', '$')
            lines.append(
                f"{i:<3} {emoji} {lvl:<7} {strat:<7} {symbol:<7} {expiry:<12} "
                f"{dte:>4} {qty:>4} {entry:>7} {pnl:>9}"
            )

    lines += ['', f"Generated {ts} UTC"]
    return '\n'.join(lines)


def _render_weekly_digest_html(
    week_start: str,
    week_end: str,
    weekly_trades: list[dict],
    cumulative_pnl: float,
    open_positions: list[dict],
    capital_deployed: float,
) -> str:
    """HTML weekly digest email."""
    wins   = [t for t in weekly_trades if (t.get('pnl') or 0) > 0]
    losses = [t for t in weekly_trades if (t.get('pnl') or 0) < 0]
    weekly_pnl = sum(float(t.get('pnl') or 0) for t in weekly_trades)
    verified_pnl = sum(float(t.get('pnl') or 0) for t in weekly_trades if t.get('pnl_verified'))
    unverified_trades = [t for t in weekly_trades if not t.get('pnl_verified')]
    win_rate   = (len(wins) / len(weekly_trades) * 100) if weekly_trades else 0.0
    best  = max((float(t.get('pnl') or 0) for t in weekly_trades), default=0.0)
    worst = min((float(t.get('pnl') or 0) for t in weekly_trades), default=0.0)
    risk_counts: dict[str, int] = {}
    for p in open_positions:
        lvl = p.get('risk_level', 'SAFE')
        risk_counts[lvl] = risk_counts.get(lvl, 0) + 1
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    def _pnl_span(v: float) -> str:
        sign  = '+' if v >= 0 else ''
        color = '#27ae60' if v >= 0 else '#e74c3c'
        return f"<span style='color:{color};font-weight:bold'>{sign}${v:,.2f}</span>"

    # ── KPI cards ────────────────────────────────────────────────────────────
    kpi_color = '#27ae60' if weekly_pnl >= 0 else '#e74c3c'
    cum_color  = '#27ae60' if cumulative_pnl >= 0 else '#e74c3c'
    kpi_sign   = '+' if weekly_pnl >= 0 else ''
    cum_sign   = '+' if cumulative_pnl >= 0 else ''
    kpi_html = f"""
    <div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px'>
      <div style='flex:1;min-width:140px;background:#f8f9fa;border-radius:8px;
                  padding:16px;text-align:center;border-top:4px solid {kpi_color}'>
        <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Week P&L</div>
        <div style='font-size:24px;font-weight:bold;color:{kpi_color}'>{kpi_sign}${weekly_pnl:,.2f}</div>
      </div>
      <div style='flex:1;min-width:140px;background:#f8f9fa;border-radius:8px;
                  padding:16px;text-align:center;border-top:4px solid {cum_color}'>
        <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Cumulative P&L</div>
        <div style='font-size:24px;font-weight:bold;color:{cum_color}'>{cum_sign}${cumulative_pnl:,.2f}</div>
      </div>
      <div style='flex:1;min-width:140px;background:#f8f9fa;border-radius:8px;
                  padding:16px;text-align:center;border-top:4px solid #2980b9'>
        <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Capital Deployed</div>
        <div style='font-size:24px;font-weight:bold;color:#2980b9'>${capital_deployed:,.0f}</div>
      </div>
      <div style='flex:1;min-width:140px;background:#f8f9fa;border-radius:8px;
                  padding:16px;text-align:center;border-top:4px solid #8e44ad'>
        <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>Win Rate</div>
        <div style='font-size:24px;font-weight:bold;color:#8e44ad'>{win_rate:.0f}%</div>
        <div style='font-size:11px;color:#999'>{len(wins)}W / {len(losses)}L this week</div>
      </div>
      <div style='flex:1;min-width:140px;background:#f8f9fa;border-radius:8px;
                  padding:16px;text-align:center;border-top:4px solid #555'>
        <div style='font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px'>P&L Quality</div>
        <div style='font-size:18px;font-weight:bold;color:#555'>${verified_pnl:,.2f} verified</div>
        <div style='font-size:11px;color:#999'>{len(unverified_trades)} unverified row(s)</div>
      </div>
    </div>"""

    # ── Weekly closed trades ─────────────────────────────────────────────────
    if weekly_trades:
        trade_rows = ''
        for t in sorted(
            weekly_trades,
            key=lambda x: x.get('closed_at') or x.get('status_updated_at') or x.get('timestamp', ''),
            reverse=True,
        ):
            pnl_v = float(t.get('pnl') or 0)
            qty   = int(t.get('contracts') or 1)
            closed_at = t.get('closed_at') or t.get('status_updated_at') or t.get('timestamp') or ''
            source = _pnl_source_label(t)
            src_color = '#27ae60' if t.get('pnl_verified') else '#d35400'
            trade_rows += (
                f"<tr>"
                f"<td>{closed_at[:10]}</td>"
                f"<td>{t.get('type','?')}</td>"
                f"<td>{t.get('symbol','?')}</td>"
                f"<td>{t.get('expiry','?')}</td>"
                f"<td style='text-align:right;font-weight:bold'>{qty}</td>"
                f"<td>${float(t.get('premium') or 0)*100:.2f}</td>"
                f"<td>{_pnl_span(pnl_v)}</td>"
                f"<td style='color:{src_color};font-size:12px'>{source}</td>"
                f"</tr>"
            )
        best_row  = (f"<tr><td colspan='6' style='color:#555'>Best trade</td>"
                     f"<td>{_pnl_span(best)}</td></tr>") if weekly_trades else ''
        worst_row = (f"<tr><td colspan='6' style='color:#555'>Worst trade</td>"
                     f"<td>{_pnl_span(worst)}</td></tr>") if weekly_trades else ''
        trades_html = f"""
        <h3 style='font-size:14px;color:#555;margin:20px 0 8px'>Closed This Week ({len(weekly_trades)} trades)</h3>
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead><tr style='background:#f0f0f0'>
            <th style='text-align:left;padding:6px'>Date</th>
            <th>Strategy</th><th>Symbol</th><th>Expiry</th><th>Qty</th><th>Premium</th><th>P&amp;L</th><th>Source</th>
          </tr></thead>
          <tbody>{trade_rows}</tbody>
          <tfoot style='border-top:2px solid #ddd'>
            {best_row}{worst_row}
            <tr style='font-weight:bold'>
              <td colspan='6'>Week Total</td>
              <td>{_pnl_span(weekly_pnl)}</td>
            </tr>
          </tfoot>
        </table>"""
    else:
        trades_html = "<p style='color:#888;font-size:13px'>No trades closed this week.</p>"

    # ── Open positions risk table ─────────────────────────────────────────────
    if open_positions:
        risk_rows = ''
        for p in open_positions:
            lvl   = p.get('risk_level', 'SAFE')
            style = _RISK_STYLE.get(lvl, _RISK_STYLE['SAFE'])
            bg, badge_c, _, emoji = style
            qty   = int(p.get('contracts') or 1)
            pnl_v = float(p.get('pnl_dollars') or 0)
            risk_rows += (
                f"<tr style='background:{bg}'>"
                f"<td><span style='background:{badge_c};color:#fff;border-radius:4px;"
                f"padding:2px 6px;font-size:11px'>{emoji} {lvl}</span></td>"
                f"<td>{p.get('type','?')}</td>"
                f"<td>{p.get('symbol','?')}</td>"
                f"<td>{p.get('expiry','?')}</td>"
                f"<td>{p.get('dte','—')}d</td>"
                f"<td style='text-align:right;font-weight:bold'>{qty}</td>"
                f"<td>${float(p.get('premium') or 0)*100:.2f}</td>"
                f"<td>{_pnl_span(pnl_v)}</td>"
                f"</tr>"
            )
        open_html = f"""
        <h3 style='font-size:14px;color:#555;margin:24px 0 8px'>Open Positions ({len(open_positions)})</h3>
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead><tr style='background:#f0f0f0'>
            <th style='text-align:left;padding:6px'>Risk</th>
            <th>Strategy</th><th>Symbol</th><th>Expiry</th>
            <th>DTE</th><th>Qty</th><th>Entry</th><th>Unrealized P&amp;L</th>
          </tr></thead>
          <tbody>{risk_rows}</tbody>
        </table>"""
    else:
        open_html = "<p style='color:#888;font-size:13px'>No open positions.</p>"

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
{_CSS_BASE}
th,td{{font-size:13px;padding:6px 10px;text-align:center}}
th{{text-align:center}}
td:first-child{{text-align:left}}
</style></head>
<body>
<div class="card">
  <div class="hdr" style="background:#1a1a2e">
    <h1>Weekly Digest</h1>
    <p>{week_start} &ndash; {week_end}</p>
  </div>
  <div class="body">
    {kpi_html}
    {trades_html}
    {open_html}
    <p class="footer">Generated {ts} UTC</p>
  </div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Daily risk report renderers
# ---------------------------------------------------------------------------

# Risk level display config  (level → bg_color, badge_color, label_color, emoji)
_RISK_STYLE: dict[str, tuple[str, str, str, str]] = {
    'SAFE':     ('#eafaf1', '#27ae60', '#1d6a39', '🟢'),
    'WATCH':    ('#fef9e7', '#f39c12', '#7d5a00', '🟡'),
    'CAUTION':  ('#fef0e7', '#e67e22', '#7b3a00', '🟠'),
    'CRITICAL': ('#fdedec', '#e74c3c', '#7b1111', '🔴'),
}


def _render_daily_risk_text(
    positions: list[dict],
    date_str: str,
    closed_today: Optional[list[dict]] = None,
) -> str:
    counts: dict[str, int] = {}
    for p in positions:
        lvl = p.get('risk_level', 'SAFE')
        counts[lvl] = counts.get(lvl, 0) + 1

    summary_parts = [f"{v} {k}" for k, v in counts.items() if v]
    closed_today = closed_today or []
    lines = [
        f"OptionWheel — Daily Risk Report — {date_str}",
        '=' * 70,
        f"{len(positions)} open position(s)  |  " + ('  '.join(summary_parts) if summary_parts else 'none'),
        '',
        f"{'#':<3} {'Risk':<9} {'Strat':<7} {'Symbol':<7} {'Expiry':<12} "
        f"{'DTE':>4} {'Qty':>4} {'Entry':>7} {'Mark':>7} {'P&L$':>9} {'Stop%':>6} "
        f"{'γ/θ':>6} {'Δ':>7} {'Score':>7}",
        '-' * 100,
    ]
    for i, p in enumerate(positions, 1):
        lvl    = p.get('risk_level', 'SAFE')
        emoji  = _RISK_STYLE.get(lvl, _RISK_STYLE['SAFE'])[3]
        strat  = p.get('type', '?')
        symbol = p.get('symbol', '?')
        expiry = p.get('expiry', '?')
        dte    = f"{p['dte']}d" if p.get('dte') is not None else '—'
        qty    = int(p.get('contracts') or 1)
        entry  = _fmt_opt((p.get('premium') or 0) * 100,      '.2f', '$')
        mark   = _fmt_opt((p.get('current_mark') or 0) * 100, '.2f', '$')
        pnl    = _fmt_opt(p.get('pnl_dollars'),               '+,.2f', '$')
        stop   = _fmt_opt(p.get('stop_proximity_pct'), '.0f', suffix='%')
        gr     = _fmt_opt(p.get('gamma_theta_ratio'), '.2f')
        delta  = _fmt_opt(p.get('net_short_delta'),  '.3f')
        score  = _fmt_opt(p.get('risk_score'),       '.2f')
        lines.append(
            f"{i:<3} {emoji} {lvl:<7} {strat:<7} {symbol:<7} {expiry:<12} "
            f"{dte:>4} {qty:>4} {entry:>7} {mark:>7} {pnl:>9} {stop:>6} "
            f"{gr:>6} {delta:>7} {score:>7}"
        )
    lines += [
        '-' * 100,
        '',
        'Risk levels:',
        '  🟢 SAFE     — profit ≥25% captured, stop <50%, γ/θ <0.8',
        '  🟡 WATCH    — stop 50-75%, γ/θ 0.8-1.2',
        '  🟠 CAUTION  — profit 0-25%, stop 75-90%, γ/θ 1.2-1.5',
        '  🔴 CRITICAL — losing money, stop >90%, or γ/θ >1.5',
    ]

    if closed_today:
        lines += [
            '',
            f"Stop-Loss / Gamma-Risk Closures Today ({len(closed_today)} position(s))",
            '=' * 70,
            f"{'#':<3} {'Reason':<12} {'Strat':<7} {'Symbol':<7} {'Expiry':<12} "
            f"{'Qty':>4} {'P&L $':>9} {'Entry':>7} {'Mark':>7} {'γ/θ':>6} {'Δshort':>7} "
            f"{'Score':>7} {'DTE':>4}",
            '-' * 100,
        ]
        for i, cp in enumerate(closed_today, 1):
            tag    = cp.get('reason_tag', 'STOP_LOSS')
            strat  = cp.get('type', '?')
            symbol = cp.get('symbol', '?')
            expiry = cp.get('expiry', '?')
            qty    = int(cp.get('contracts') or 1)
            pnl    = _fmt_opt(cp.get('close_pnl'),       '+,.2f', '$')
            entry  = _fmt_opt((cp.get('entry_premium') or 0) * 100,  '.2f', '$')
            mark   = _fmt_opt((cp.get('current_mark') or 0) * 100, '.2f', '$')
            gr     = _fmt_opt(cp.get('ratio'),            '.2f')
            delta  = _fmt_opt(cp.get('short_delta'),      '.3f')
            score  = _fmt_opt(cp.get('risk_score'),       '.2f')
            dte    = f"{cp['dte']}d" if cp.get('dte') is not None else '—'
            lines.append(
                f"{i:<3} {tag:<12} {strat:<7} {symbol:<7} {expiry:<12} "
                f"{qty:>4} {pnl:>9} {entry:>7} {mark:>7} {gr:>6} {delta:>7} "
                f"{score:>7} {dte:>4}"
            )
        lines.append('-' * 100)

    return '\n'.join(lines) + '\n'


def _render_daily_risk_html(
    positions: list[dict],
    date_str: str,
    closed_today: Optional[list[dict]] = None,
) -> str:
    counts: dict[str, int] = {}
    for p in positions:
        lvl = p.get('risk_level', 'SAFE')
        counts[lvl] = counts.get(lvl, 0) + 1

    # Summary badges
    badge_html = ''
    for lvl in ('CRITICAL', 'CAUTION', 'WATCH', 'SAFE'):
        c = counts.get(lvl, 0)
        if c == 0:
            continue
        _, badge_bg, _, emoji = _RISK_STYLE[lvl]
        badge_html += (
            f"<span style='background:{badge_bg};color:#fff;border-radius:12px;"
            f"padding:4px 12px;font-size:13px;font-weight:bold;margin-right:8px'>"
            f"{emoji} {c} {lvl}</span>"
        )

    # Table rows
    rows_html = ''
    for i, p in enumerate(positions, 1):
        lvl  = p.get('risk_level', 'SAFE')
        row_bg, badge_bg, txt_color, emoji = _RISK_STYLE.get(lvl, _RISK_STYLE['SAFE'])
        strat  = p.get('type', '?')
        symbol = p.get('symbol', '?')
        expiry = p.get('expiry', '?')
        legs   = _pos_legs_str(p)
        dte    = f"{p['dte']}" if p.get('dte') is not None else '—'
        qty    = int(p.get('contracts') or 1)
        entry  = _fmt_opt((p.get('premium') or 0) * 100,      '.2f', '$')
        spot   = _fmt_opt(p.get('spot'),                     '.2f', '$')
        mark   = _fmt_opt((p.get('current_mark') or 0) * 100, '.2f', '$')
        pnl_d  = p.get('pnl_dollars')
        pnl_color = '#27ae60' if (pnl_d or 0) >= 0 else '#e74c3c'
        pnl_str = _fmt_opt(pnl_d, '+,.2f', '$')
        profit_pct = _fmt_opt(p.get('profit_captured_pct'), '.0f', suffix='%')
        stop   = _fmt_opt(p.get('stop_proximity_pct'),  '.0f', suffix='%')
        gr     = _fmt_opt(p.get('gamma_theta_ratio'),   '.2f')
        delta  = _fmt_opt(p.get('net_short_delta'),     '.3f')
        score  = _fmt_opt(p.get('risk_score'),          '.2f')

        # Risk badge cell
        risk_badge = (
            f"<span style='background:{badge_bg};color:#fff;border-radius:8px;"
            f"padding:2px 8px;font-size:11px;font-weight:bold;white-space:nowrap'>"
            f"{emoji} {lvl}</span>"
        )
        rows_html += (
            f"<tr style='background:{row_bg}'>"
            f"<td style='color:#888;font-size:12px'>{i}</td>"
            f"<td>{risk_badge}</td>"
            f"<td><strong>{strat}</strong></td>"
            f"<td><strong style='color:#2980b9'>{symbol}</strong></td>"
            f"<td>{expiry}</td>"
            f"<td style='font-family:monospace;font-size:12px'>{legs}</td>"
            f"<td style='text-align:right'>{dte}</td>"
            f"<td style='text-align:right;font-weight:bold'>{qty}</td>"
            f"<td style='text-align:right;color:#888'>{spot}</td>"
            f"<td style='text-align:right'>{entry}</td>"
            f"<td style='text-align:right'>{mark}</td>"
            f"<td style='text-align:right;font-weight:bold;color:{pnl_color}'>{pnl_str}</td>"
            f"<td style='text-align:right'>{profit_pct}</td>"
            f"<td style='text-align:right'>{stop}</td>"
            f"<td style='text-align:right'>{gr}</td>"
            f"<td style='text-align:right'>{delta}</td>"
            f"<td style='text-align:right'>{score}</td>"
            f"</tr>"
        )

    legend_rows = ''
    for lvl, (row_bg, badge_bg, txt_color, emoji) in _RISK_STYLE.items():
        descs = {
            'SAFE':     'Profit &ge;25% captured &nbsp;|&nbsp; Stop proximity &lt;50% &nbsp;|&nbsp; γ/θ &lt;0.8',
            'WATCH':    'Stop proximity 50–75% &nbsp;|&nbsp; γ/θ 0.8–1.2',
            'CAUTION':  'Profit 0–25% &nbsp;|&nbsp; Stop proximity 75–90% &nbsp;|&nbsp; γ/θ 1.2–1.5',
            'CRITICAL': 'Losing money &nbsp;|&nbsp; Stop proximity &gt;90% &nbsp;|&nbsp; γ/θ &gt;1.5',
        }
        legend_rows += (
            f"<tr style='background:{row_bg}'>"
            f"<td><span style='background:{badge_bg};color:#fff;border-radius:8px;"
            f"padding:2px 8px;font-size:11px;font-weight:bold'>{emoji} {lvl}</span></td>"
            f"<td style='font-size:12px;color:{txt_color}'>{descs[lvl]}</td>"
            f"</tr>"
        )

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    n  = len(positions)
    closed_today = closed_today or []

    # ── Stop-loss closures section ──────────────────────────────────────────
    closures_html = ''
    if closed_today:
        def _sl_fmt(v, fmt='.2f', prefix=''):
            if v is None:
                return '<span style="color:#aaa">—</span>'
            try:
                return f"{prefix}{float(v):{fmt}}"
            except (TypeError, ValueError):
                return '<span style="color:#aaa">—</span>'

        cl_rows = ''
        for j, cp in enumerate(closed_today, 1):
            tag    = cp.get('reason_tag', 'STOP_LOSS')
            row_bg = '#fdf2f2' if tag == 'STOP_LOSS' else '#fff3e0'
            strat  = cp.get('type', '?')
            symbol = cp.get('symbol', '?')
            expiry = cp.get('expiry', '?')
            qty    = int(cp.get('contracts') or 1)
            pnl    = float(cp.get('close_pnl') or 0)
            pnl_c  = '#e74c3c' if pnl < 0 else '#27ae60'
            pnl_s  = f'<span style="color:{pnl_c};font-weight:bold">{("+" if pnl >= 0 else "")}${pnl:,.2f}</span>'
            entry  = _sl_fmt((cp.get('entry_premium') or 0) * 100, prefix='$')
            mark   = _sl_fmt((cp.get('current_mark') or 0) * 100,  prefix='$')
            gr     = _sl_fmt(cp.get('ratio'))
            delta  = _sl_fmt(cp.get('short_delta'), '.3f')
            score  = _sl_fmt(cp.get('risk_score'))
            dte    = f"{cp['dte']}d" if cp.get('dte') is not None else '<span style="color:#aaa">—</span>'
            tag_color = '#c0392b' if tag == 'STOP_LOSS' else '#d35400'
            tag_html  = (
                f"<span style='background:{tag_color};color:#fff;border-radius:8px;"
                f"padding:2px 7px;font-size:11px;font-weight:bold'>{tag}</span>"
            )
            cl_rows += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='color:#888;font-size:12px'>{j}</td>"
                f"<td>{tag_html}</td>"
                f"<td><strong>{strat}</strong></td>"
                f"<td><strong style='color:#2980b9'>{symbol}</strong></td>"
                f"<td>{expiry}</td>"
                f"<td style='text-align:right;font-weight:bold'>{qty}</td>"
                f"<td style='text-align:right'>{pnl_s}</td>"
                f"<td style='text-align:right'>{entry}</td>"
                f"<td style='text-align:right'>{mark}</td>"
                f"<td style='text-align:right'>{gr}</td>"
                f"<td style='text-align:right'>{delta}</td>"
                f"<td style='text-align:right'>{score}</td>"
                f"<td style='text-align:right'>{dte}</td>"
                f"</tr>"
            )
        closures_html = f"""
    <h2 style="color:#c0392b;margin-top:32px;font-size:15px;font-weight:bold">
      &#9888; Today&#39;s Stop-Loss Closures &mdash; {len(closed_today)} position(s)
    </h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Reason</th><th>Strategy</th><th>Symbol</th>
          <th>Expiry</th><th>Qty</th><th>P&amp;L $</th><th>Entry</th><th>Mark@Close</th>
          <th>&#947;/&#952;</th><th>&#916;short</th><th>Score</th><th>DTE</th>
        </tr>
      </thead>
      <tbody>{cl_rows}</tbody>
    </table>"""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
{_CSS_BASE}
th,td{{font-size:12px;padding:6px 8px}}
th{{background:#2c3e50;color:#fff;white-space:nowrap}}
tfoot td{{background:#ecf0f1;font-size:11px;color:#888}}
.legend td{{padding:5px 10px}}
</style>
</head>
<body>
<div class="card">
  <div class="hdr" style="background:#1a1a2e">
    <h1>Daily Risk Report &mdash; {date_str}</h1>
    <p>{n} open position{'s' if n != 1 else ''} &nbsp;&bull;&nbsp; {ts} UTC</p>
  </div>
  <div class="body">
    <p style="margin-bottom:14px">{badge_html}</p>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Risk</th><th>Strategy</th><th>Symbol</th>
          <th>Expiry</th><th>Legs</th><th>DTE</th><th>Qty</th><th>Spot</th>
          <th>Entry</th><th>Mark</th><th>P&amp;L $</th><th>Profit%</th>
          <th>Stop%</th><th>γ/θ</th><th>Δshort</th><th>Score</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
      <tfoot>
        <tr>
          <td colspan="17" style="text-align:left">
            Stop% = how close current mark is to stop-loss trigger (100% = triggered) &nbsp;|&nbsp;
            Profit% = fraction of entry premium captured &nbsp;|&nbsp;
            γ/θ = gamma/theta ratio (higher = more risky) &nbsp;|&nbsp;
            Δshort = net short-leg delta (absolute)
          </td>
        </tr>
      </tfoot>
    </table>
    <h3 style="font-size:13px;color:#555;margin-top:20px">Risk Level Key</h3>
    <table class="legend" style="width:auto">
      <tbody>{legend_rows}</tbody>
    </table>
    {closures_html}
    <p class="footer">Generated {ts} UTC</p>
  </div>
</div>
</body></html>"""
