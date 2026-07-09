"""
agent_display.py
================

Terminal formatting and display helpers for the trading agent.

Includes leg formatting, plan tables, open-position display, and
pick-to-leg extraction for database storage.

Extracted from agent.py to keep the orchestrator thin.
"""
from __future__ import annotations

from datetime import datetime

from src.agent_risk import capital_for_pick


# ── Formatting helpers ───────────────────────────────────────────────────────

def fmt_mcap(v) -> str:
    """Format a market-cap value as '1.2T', '45.3B', '850M', or '—'."""
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        return '—'
    if v >= 1e12:
        return f'{v/1e12:.1f}T'
    if v >= 1e9:
        return f'{v/1e9:.1f}B'
    if v >= 1e6:
        return f'{v/1e6:.0f}M'
    return f'{v:,.0f}'


def fmt_prob(v) -> str:
    """Format a model-derived probability or show n/a when unavailable."""
    try:
        if v is None:
            raise TypeError
        return f'{float(v):>5.1%}'
    except (TypeError, ValueError):
        return f"{'n/a':>6}"


def legs_from_pick(pick: dict) -> dict:
    """
    Extract leg strikes from a model candidate dict into a flat dict suitable
    for storage in the database's 'legs' JSON column.
    Used by PositionMonitor to price the position for stop-loss checks.
    Extra keys (market_cap, short_oi, short_volume) are stored alongside
    the strikes so the dashboard can display them without a live re-fetch.
    """
    strat = pick.get('strategy', '')
    if strat in ('PCS', 'CCS'):
        legs = {
            'short_strike': pick.get('short_strike'),
            'long_strike':  pick.get('long_strike'),
        }
    elif strat in ('IC', 'IFLY'):
        legs = {
            'short_put':  pick.get('short_put'),
            'long_put':   pick.get('long_put'),
            'short_call': pick.get('short_call'),
            'long_call':  pick.get('long_call'),
        }
    elif strat == 'CSP':
        legs = {'short_strike': pick.get('short_strike')}
    elif strat == 'CC':
        legs = {'short_strike': pick.get('short_strike') or pick.get('short_call')}
    elif strat == 'STRANGLE':
        legs = {
            'short_put':  pick.get('short_put'),
            'short_call': pick.get('short_call'),
        }
    else:
        legs = {}
    # Persist display metadata so the dashboard doesn't need a live re-fetch
    if pick.get('market_cap') is not None:
        legs['market_cap'] = pick['market_cap']
    if pick.get('short_oi') is not None:
        legs['short_oi'] = pick['short_oi']
    if 'short_volume' in pick:
        legs['short_volume'] = pick['short_volume']
    if pick.get('mispricing_score') is not None:
        legs['mispricing_score'] = pick['mispricing_score']
        legs['mispricing_score_basis'] = pick.get('mispricing_score_basis')
    return legs


def legs_str(pick: dict) -> str:
    """Compact leg description for the plan table."""
    strat = pick.get('strategy', '')
    if strat == 'CSP':
        return f"{pick.get('short_strike')}P"
    if strat == 'PCS':
        return f"{pick.get('short_put') or pick.get('short_strike')}/{pick.get('long_put') or pick.get('long_strike')} P"
    if strat == 'CCS':
        return f"{pick.get('short_call') or pick.get('short_strike')}/{pick.get('long_call') or pick.get('long_strike')} C"
    if strat == 'IC':
        return (f"{pick.get('long_put')}/{pick.get('short_put')} P  "
                f"{pick.get('short_call')}/{pick.get('long_call')} C")
    if strat == 'IFLY':
        return (f"{pick.get('long_put')}/{pick.get('short_put')}(ATM)"
                f"/{pick.get('long_call')}")
    if strat == 'STRANGLE':
        return f"{pick.get('short_put')}P / {pick.get('short_call')}C"
    if strat == 'CC':
        return f"{pick.get('short_strike') or pick.get('short_call')}C"
    return '?'


# ── Table displays ───────────────────────────────────────────────────────────

def print_open_positions(open_positions: list[dict], monitor) -> None:
    """
    Print a formatted table of all open positions with live mark prices and
    unrealized P&L fetched via PositionMonitor._get_current_mark().
    """
    import datetime as _dt

    today = _dt.date.today()

    print()
    print("=" * 106)
    print("  OPEN POSITIONS — CURRENT P&L")
    print(f"  As of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {len(open_positions)} position(s)")
    print("=" * 106)

    hdr  = f"  {'#':>3}  {'Strat':<6}  {'Symbol':<6}  {'Entry':<10}  {'Expiry':<10}  {'DTE':>4}  "
    hdr += f"{'Spot':>8}  {'Entry$':>8}  {'Mark$':>8}  {'Unreal P&L':>12}  {'P&L%':>7}  {'Status'}"
    print(hdr)
    print("  " + "-" * 102)

    unique_symbols = list({pos.get('symbol', '?') for pos in open_positions})
    spot_prices = monitor._data.get_spot_prices(unique_symbols)

    total_pnl = 0.0
    priced    = 0

    for i, pos in enumerate(open_positions, start=1):
        strat   = pos.get('type', '?')
        symbol  = pos.get('symbol', '?')
        entry   = (pos.get('timestamp') or '')[:10]
        expiry  = pos.get('expiry', '?')
        premium = float(pos.get('premium', 0) or 0)
        status  = pos.get('status', '?')

        try:
            exp_date = _dt.date.fromisoformat(expiry)
            dte      = (exp_date - today).days
            dte_str  = f"{dte}d"
        except Exception:
            dte_str  = "?"

        spot_val = spot_prices.get(symbol)
        spot_str = f"${spot_val:>7.2f}" if spot_val is not None else f"{'N/A':>8}"

        # Fetch current option mark
        current_mark = monitor._get_current_mark(pos)

        if current_mark is not None:
            unreal_pnl = (premium - current_mark) * 100
            pnl_pct    = (premium - current_mark) / premium * 100 if premium > 0 else 0.0
            total_pnl += unreal_pnl
            priced    += 1
            pnl_str    = f"${unreal_pnl:>+10,.2f}"
            pct_str    = f"{pnl_pct:>+6.1f}%"
            mark_str   = f"${current_mark:>7.4f}"
        else:
            pnl_str  = f"{'N/A':>11}"
            pct_str  = f"{'N/A':>7}"
            mark_str = f"{'N/A':>8}"

        entry_str = f"${premium * 100:>7.2f}"

        print(
            f"  {i:>3}. {strat:<6}  {symbol:<6}  {entry:<10}  {expiry:<10}  {dte_str:>4}  "
            f"{spot_str}  {entry_str}  {mark_str}  {pnl_str}  {pct_str}  {status}"
        )

    print("  " + "-" * 102)
    if priced > 0:
        sign = "+" if total_pnl >= 0 else ""
        print(f"  Total unrealized P&L ({priced}/{len(open_positions)} priced): "
              f"${sign}{total_pnl:,.2f}")
    else:
        print(f"  Could not price any positions (market may be closed or data unavailable).")
    print("=" * 106)
    print()


def print_plan(picks: list[dict], capital_budget=None) -> None:
    """
    Print the trading plan as a formatted table.

    Columns: #  Strategy  Symbol  Price  Expiry  Legs  Qty  Credit  Capital  Prob  ROI  Score  MCap  OI  Vol
    """
    total_capital = sum(capital_for_pick(p) * p.get('quantity', 1) for p in picks)
    total_credit  = sum(p.get('premium', 0) * 100 * p.get('quantity', 1) for p in picks)

    print()
    print("=" * 150)
    print("  OPTION WHEEL — TRADING PLAN")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(picks)} picks  |  Total premium: ${total_credit:,.2f}  |  "
          f"Total capital required: ${total_capital:,.0f}"
          + (f"  (budget: ${capital_budget:,.0f})" if capital_budget else ""))
    print("=" * 150)

    hdr = f"{'#':>3}  {'Strat':<6}  {'Symbol':<6}  {'Price':>9}  {'Expiry':<10}  {'Legs':<32}  "
    hdr += f"{'Qty':>4}  {'Credit':>7}  {'Capital':>9}  {'Prob':>6}  {'ROI':>6}  {'Score':>6}  "
    hdr += f"{'MCap':>7}  {'OI':>7}  {'Vol':>7}"
    print(hdr)
    print("-" * 150)

    for i, p in enumerate(picks, start=1):
        strat   = p.get('strategy', '?')
        symbol  = p.get('symbol', '?')
        price   = p.get('current_price')
        expiry  = p.get('expiry', '?')
        legs    = legs_str(p)
        qty     = p.get('quantity', 1)
        credit  = p.get('premium', 0) * 100 * qty   # total dollars
        capital = capital_for_pick(p) * qty           # total capital
        prob    = fmt_prob(p.get('prob_win'))
        roi     = p.get('roi', 0)
        score   = p.get('score', 0)
        mcap    = fmt_mcap(p.get('market_cap'))
        oi      = p.get('short_oi')
        vol     = p.get('short_volume', 0)

        price_str = f"${price:>7.2f}" if price is not None else f"{'N/A':>8}"
        oi_str  = f"{oi:>7,}" if oi is not None else f"{'—':>7}"
        vol_str = f"{vol:>7,}" if vol else f"{'—':>7}"

        print(
            f"{i:>3}. {strat:<6}  {symbol:<6}  {price_str}  {expiry:<10}  {legs:<32}  "
            f"{qty:>4}  ${credit:>6.2f}  ${capital:>8,.0f}  "
            f"{prob}  {roi:>5.1%}  {score:>6.3f}  "
            f"{mcap:>7}  {oi_str}  {vol_str}"
        )

    print("-" * 150)
    print(f"{'TOTAL':<72}  {'':>4}  {'':>7}  ${total_capital:>8,.0f}")
    print("=" * 150)
    print()


def print_positions_table(positions: list[dict]) -> None:
    """Print a compact table of open positions (ID, strat, symbol, expiry, premium)."""
    if not positions:
        print("  No open positions found.")
        return
    print(f"  {'ID':>4}  {'Strat':<6}  {'Symbol':<7}  {'Entry':<10}  {'Expiry':<10}  "
          f"{'Premium':>8}  Status")
    print("  " + "-" * 65)
    for p in positions:
        prem = float(p.get('premium', 0) or 0)
        print(f"  {p['id']:>4}  {p.get('type','?'):<6}  {p.get('symbol','?'):<7}  "
              f"{(p.get('timestamp') or '')[:10]:<10}  {p.get('expiry','?'):<10}  "
              f"${prem * 100:>7.2f}  {p.get('status','?')}")
    print()
