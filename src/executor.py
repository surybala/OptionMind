"""
AlpacaExecutor
==============

Thin wrapper around the Alpaca Trading API (alpaca-py) that translates
scanner pick dicts into multi-leg option orders.

Supports all enabled strategies:
  PCS   — put credit spread   (2 legs: sell put / buy lower put)
  CCS   — call credit spread  (2 legs: sell call / buy higher call)
  IC    — iron condor         (4 legs: put spread + call spread)
  IFLY  — iron butterfly      (4 legs: ATM short straddle + OTM wings)
  CSP   — cash-secured put    (1 leg: sell naked put)
  CC    — covered call        (1 leg: sell call against long stock)

The high-level entry point is execute_pick(pick, dry_run), which routes a
scanner pick dict to the correct method automatically.
"""
from __future__ import annotations

import logging
import os
import json

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, PositionIntent
from src.order_status import normalize_order_status
from src.utils import load_config

_log = logging.getLogger('optionwheel')


def _extract_held_for_orders_order_id(exc: Exception) -> str | None:
    """
    Return Alpaca's related open order ID when a close is rejected because the
    position quantity is already held for another order.
    """
    msg = str(exc)
    start = msg.find('{')
    end = msg.rfind('}')
    if start < 0 or end < start:
        return None

    try:
        payload = json.loads(msg[start:end + 1])
    except Exception:
        return None

    if str(payload.get('code')) != '40310000':
        return None
    if str(payload.get('held_for_orders', '0')) in ('', '0'):
        return None

    related = payload.get('related_orders') or []
    if not related:
        return None
    return str(related[0])


def _signed_option_qty(position) -> float | None:
    """Return signed option qty from an Alpaca position object."""
    try:
        qty = abs(float(getattr(position, 'qty', 0) or 0))
    except Exception:
        return None

    side = str(getattr(position, 'side', '') or '').lower()
    if side == 'short':
        return -qty
    if side == 'long':
        return qty

    # Some SDK versions expose signed qty directly instead of side.
    try:
        return float(getattr(position, 'qty', 0) or 0)
    except Exception:
        return None


def _load_alpaca_credentials(config: dict) -> tuple[str, str, bool]:
    """
    Resolve Alpaca credentials with the following priority:

    1. Environment variables   ALPACA_API_KEY / ALPACA_API_SECRET / ALPACA_PAPER
    2. config.json             alpaca.api_key / alpaca.api_secret / alpaca.paper

    Returns (api_key, api_secret, paper_mode).
    The winning source is printed so it is obvious at runtime which path was used.
    """
    creds = config.get('alpaca', {})

    env_key    = os.environ.get('ALPACA_API_KEY',    '').strip()
    env_secret = os.environ.get('ALPACA_API_SECRET', '').strip()
    env_paper  = os.environ.get('ALPACA_PAPER',      '').strip().lower()

    cfg_key    = creds.get('api_key',    '').strip()
    cfg_secret = creds.get('api_secret', '').strip()
    cfg_paper  = creds.get('paper', True)

    # Use env vars when both key and secret are present
    if env_key and env_secret:
        paper = cfg_paper          # start with config default
        if env_paper in ('0', 'false', 'no', 'live'):
            paper = False
        elif env_paper in ('1', 'true', 'yes', 'paper'):
            paper = True
        _log.info("[Alpaca] Credentials loaded from environment variables "
                  "(ALPACA_API_KEY / ALPACA_API_SECRET). paper=%s", paper)
        return env_key, env_secret, paper

    # Fall back to config.json
    if cfg_key and cfg_secret:
        _log.info("[Alpaca] Credentials loaded from config.json. "
                  "Tip: set ALPACA_API_KEY / ALPACA_API_SECRET env vars to avoid "
                  "storing secrets in config.json.")
    return cfg_key, cfg_secret, cfg_paper


def _osi_symbol(symbol: str, expiry: str, strike: float, option_type: str) -> str:
    """
    Build an OCC/OSI option symbol in the compact form Alpaca expects.

    Format: {symbol}{YYMMDD}{C|P}{strike*1000:08d}
    Example: AAPL260430P00150000  (AAPL $150 put expiring 2026-04-30)
             AZO260320C03820000   (AZO $3820 call expiring 2026-03-20)

    Note: Alpaca uses the compact (unpadded) OCC format — the root symbol is
    NOT padded to 6 characters with spaces.  Sending "AZO   …" (padded) results
    in an "asset not found" API error.
    """
    ymd = expiry.replace('-', '')[2:]          # '2026-04-30' → '260430'
    flag = 'C' if option_type.upper() == 'CALL' else 'P'
    strike_units = int(round(strike * 1000))   # $150.00 → 150000
    return f"{symbol}{ymd}{flag}{strike_units:08d}"


class AlpacaExecutor:
    """Places option orders via the Alpaca Trading API (alpaca-py SDK)."""

    def __init__(self, config_path="config.json"):
        self.config = load_config(config_path)
        self._api_key, self._api_secret, self._paper = _load_alpaca_credentials(self.config)
        self.client: TradingClient | None = None
        self.is_logged_in = False

    def login(self) -> bool:
        if not self._api_key or not self._api_secret:
            _log.warning(
                "Alpaca credentials missing. "
                "Set ALPACA_API_KEY / ALPACA_API_SECRET env vars, "
                "or add api_key / api_secret to config.json."
            )
            return False
        self.client = TradingClient(self._api_key, self._api_secret, paper=self._paper)
        _log.info("Alpaca client initialised (%s mode).", 'paper' if self._paper else 'live')
        self.is_logged_in = True
        return True

    # ── Strategy 1: Cash Secured Put ─────────────────────────────────────────

    def execute_sell_put(self, symbol, expiry, strike,
                         limit_price=None, amount=1, dry_run=True):
        if dry_run:
            _log.info("[DRY RUN] Selling %dx %s PUT | Exp: %s | Strike: %s", amount, symbol, expiry, strike)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest
        opt_sym = _osi_symbol(symbol, expiry, strike, 'PUT')
        order = LimitOrderRequest(
            symbol=opt_sym,
            qty=amount,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            position_intent=PositionIntent.SELL_TO_OPEN,
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca CSP order submitted for %s: %s", symbol, res.id)
        return str(res.id)

    # ── Strategy 2 & 3: Put / Call Credit Spread ──────────────────────────────

    def execute_sell_spread(self, symbol, expiry, short_strike, long_strike,
                            strategy, limit_price=None, amount=1, dry_run=True):
        """Execute a credit spread (PCS or CCS): sell the short leg, buy the long leg."""
        if dry_run:
            _log.info("[DRY RUN] %s %dx %s | Exp: %s | Short: %s / Long: %s",
                      strategy, amount, symbol, expiry, short_strike, long_strike)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass
        opt_type  = 'PUT' if strategy == 'PCS' else 'CALL'
        short_sym = _osi_symbol(symbol, expiry, short_strike, opt_type)
        long_sym  = _osi_symbol(symbol, expiry, long_strike,  opt_type)
        order = LimitOrderRequest(
            qty=amount,
            order_class=OrderClass.MLEG,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            legs=[
                OptionLegRequest(symbol=short_sym, side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=long_sym,  side=OrderSide.BUY,  ratio_qty=1,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
            ],
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca %s order submitted for %s: %s", strategy, symbol, res.id)
        return str(res.id)

    # ── Strategy 4: Iron Condor ───────────────────────────────────────────────

    def execute_sell_iron_condor(self, symbol, expiry,
                                 short_put, long_put, short_call, long_call,
                                 limit_price=None, amount=1, dry_run=True):
        """Execute an Iron Condor: four-leg combo — put spread + call spread."""
        if dry_run:
            _log.info("[DRY RUN] IC %dx %s | Exp: %s | Puts: %s/%s | Calls: %s/%s",
                      amount, symbol, expiry, long_put, short_put, short_call, long_call)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass
        order = LimitOrderRequest(
            qty=amount,
            order_class=OrderClass.MLEG,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            legs=[
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, short_put,  'PUT'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, long_put,   'PUT'),
                                 side=OrderSide.BUY,  ratio_qty=1,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, short_call, 'CALL'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, long_call,  'CALL'),
                                 side=OrderSide.BUY,  ratio_qty=1,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
            ],
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca Iron Condor order submitted for %s: %s", symbol, res.id)
        return str(res.id)

    # ── Strategy 5: Iron Butterfly ───────────────────────────────────────────

    def execute_sell_iron_butterfly(self, symbol, expiry,
                                    short_put, long_put, short_call, long_call,
                                    limit_price=None, amount=1, dry_run=True):
        """
        Execute an Iron Butterfly: sell ATM put + sell ATM call + buy OTM wings.

        Four-leg combo order:
          SELL  short_put  PUT  (ATM)
          BUY   long_put   PUT  (lower wing)
          SELL  short_call CALL (ATM, same strike as short_put)
          BUY   long_call  CALL (upper wing)
        """
        if dry_run:
            _log.info("[DRY RUN] IFLY %dx %s | Exp: %s | ATM: %s | Put wing: %s | Call wing: %s",
                      amount, symbol, expiry, short_put, long_put, long_call)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass
        order = LimitOrderRequest(
            qty=amount,
            order_class=OrderClass.MLEG,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            legs=[
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, short_put,  'PUT'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, long_put,   'PUT'),
                                 side=OrderSide.BUY,  ratio_qty=1,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, short_call, 'CALL'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, long_call,  'CALL'),
                                 side=OrderSide.BUY,  ratio_qty=1,
                                 position_intent=PositionIntent.BUY_TO_OPEN),
            ],
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca Iron Butterfly order submitted for %s: %s", symbol, res.id)
        return str(res.id)

    # ── Strategy 6: Short Strangle ────────────────────────────────────────────

    def execute_sell_strangle(self, symbol, expiry, put_strike, call_strike,
                              limit_price=None, amount=1, dry_run=True):
        """Execute a Short Strangle: sell a naked OTM put + a naked OTM call."""
        if dry_run:
            _log.info("[DRY RUN] STRANGLE %dx %s | Exp: %s | Put: %s | Call: %s",
                      amount, symbol, expiry, put_strike, call_strike)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass
        order = LimitOrderRequest(
            qty=amount,
            order_class=OrderClass.MLEG,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            legs=[
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, put_strike,  'PUT'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
                OptionLegRequest(symbol=_osi_symbol(symbol, expiry, call_strike, 'CALL'),
                                 side=OrderSide.SELL, ratio_qty=1,
                                 position_intent=PositionIntent.SELL_TO_OPEN),
            ],
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca Strangle order submitted for %s: %s", symbol, res.id)
        return str(res.id)

    # ── Strategy 7: Covered Call ──────────────────────────────────────────────

    def execute_sell_covered_call(self, symbol, expiry, strike,
                                  limit_price=None, amount=1, dry_run=True):
        """Execute a Covered Call: sell an OTM call against an assumed long stock position."""
        if dry_run:
            _log.info("[DRY RUN] CC %dx %s | Exp: %s | Strike: %s", amount, symbol, expiry, strike)
            return "DRY_RUN_ID"

        if not self.is_logged_in:
            if not self.login(): return None

        from alpaca.trading.requests import LimitOrderRequest
        opt_sym = _osi_symbol(symbol, expiry, strike, 'CALL')
        order = LimitOrderRequest(
            symbol=opt_sym,
            qty=amount,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            position_intent=PositionIntent.SELL_TO_OPEN,
        )
        res = self.client.submit_order(order)
        _log.info("Alpaca Covered Call order submitted for %s: %s", symbol, res.id)
        return str(res.id)

    # ── Universal dispatcher ───────────────────────────────────────────────────

    def execute_pick(self, pick: dict, dry_run: bool = True, amount: int = 1) -> str | None:
        """
        Route a scanner pick dict to the correct execution method.

        Returns the broker order ID on success, or None on failure.
        All live orders are submitted as DAY limit orders at the scanned premium.

        Parameters
        ----------
        pick     : scanner pick dict (must contain 'strategy', 'symbol', 'expiry', etc.)
        dry_run  : True = print intent but do NOT submit to Alpaca (default: True)
        amount   : number of contracts (default: 1)
        """
        strat  = pick.get('strategy', '')
        symbol = pick['symbol']
        expiry = pick['expiry']
        prem   = pick.get('premium')

        if strat == 'CSP':
            return self.execute_sell_put(
                symbol, expiry, pick['short_strike'],
                limit_price=prem, amount=amount, dry_run=dry_run,
            )

        if strat in ('PCS', 'CCS'):
            return self.execute_sell_spread(
                symbol, expiry,
                pick.get('short_strike') or pick.get('short_put') or pick.get('short_call'),
                pick.get('long_strike')  or pick.get('long_put')  or pick.get('long_call'),
                strat, limit_price=prem, amount=amount, dry_run=dry_run,
            )

        if strat == 'IC':
            return self.execute_sell_iron_condor(
                symbol, expiry,
                pick['short_put'], pick['long_put'],
                pick['short_call'], pick['long_call'],
                limit_price=prem, amount=amount, dry_run=dry_run,
            )

        if strat == 'IFLY':
            return self.execute_sell_iron_butterfly(
                symbol, expiry,
                pick['short_put'], pick['long_put'],
                pick['short_call'], pick['long_call'],
                limit_price=prem, amount=amount, dry_run=dry_run,
            )

        if strat == 'STRANGLE':
            return self.execute_sell_strangle(
                symbol, expiry, pick['short_put'], pick['short_call'],
                limit_price=prem, amount=amount, dry_run=dry_run,
            )

        if strat == 'CC':
            return self.execute_sell_covered_call(
                symbol, expiry,
                pick.get('short_strike') or pick.get('short_call'),
                limit_price=prem, amount=amount, dry_run=dry_run,
            )

        _log.warning("[executor] Unknown strategy '%s' — skipping %s", strat, symbol)
        return None

    # ── Pre-flight contract validation ────────────────────────────────────────

    def preflight_check_picks(
        self,
        picks: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Verify every option leg in each pick is active and tradeable on Alpaca.

        Makes one API call per unique OSI symbol in parallel (≤ 10 workers).
        Contracts not found (APIError) or with tradable=False are treated as
        inactive.

        Parameters
        ----------
        picks : list of scanner pick dicts

        Returns
        -------
        (valid_picks, filtered_picks)
            valid_picks   — picks where ALL legs are active/tradeable
            filtered_picks — picks with at least one inactive leg; each dict
                             gets an extra key '_inactive_contracts' listing
                             the bad OSI symbols for logging.

        Skips entirely (returns picks unchanged) when:
          - Alpaca credentials are absent
          - Login to Alpaca fails
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self._api_key or not self._api_secret:
            return picks, []
        if not self.is_logged_in and not self.login():
            return picks, []

        def _osi_legs(pick: dict) -> list[str]:
            """Return all OSI symbols needed to open this pick."""
            sym    = pick.get('symbol', '')
            expiry = pick.get('expiry', '')
            strat  = pick.get('strategy', '')
            legs   = []
            if strat == 'CSP':
                s = pick.get('short_strike')
                if s: legs.append(_osi_symbol(sym, expiry, s, 'PUT'))
            elif strat == 'PCS':
                ss = pick.get('short_strike') or pick.get('short_put')
                ls = pick.get('long_strike')  or pick.get('long_put')
                if ss: legs.append(_osi_symbol(sym, expiry, ss, 'PUT'))
                if ls: legs.append(_osi_symbol(sym, expiry, ls, 'PUT'))
            elif strat == 'CCS':
                ss = pick.get('short_strike') or pick.get('short_call')
                ls = pick.get('long_strike')  or pick.get('long_call')
                if ss: legs.append(_osi_symbol(sym, expiry, ss, 'CALL'))
                if ls: legs.append(_osi_symbol(sym, expiry, ls, 'CALL'))
            elif strat in ('IC', 'IFLY'):
                for k, t in [('short_put', 'PUT'), ('long_put',  'PUT'),
                              ('short_call', 'CALL'), ('long_call', 'CALL')]:
                    s = pick.get(k)
                    if s: legs.append(_osi_symbol(sym, expiry, s, t))
            elif strat == 'STRANGLE':
                sp = pick.get('short_put');  sc = pick.get('short_call')
                if sp: legs.append(_osi_symbol(sym, expiry, sp, 'PUT'))
                if sc: legs.append(_osi_symbol(sym, expiry, sc, 'CALL'))
            elif strat == 'CC':
                ss = pick.get('short_strike') or pick.get('short_call')
                if ss: legs.append(_osi_symbol(sym, expiry, ss, 'CALL'))
            return legs

        def _check(osi: str) -> bool:
            """Return True if the contract is active and tradeable on Alpaca."""
            try:
                c = self.client.get_option_contract(osi)
                return bool(getattr(c, 'tradable', False))
            except Exception:
                return False   # 404 or any API error → not tradeable

        # Build per-pick OSI leg lists and collect unique symbols to check
        pick_legs: list[tuple[dict, list[str]]] = [
            (p, _osi_legs(p)) for p in picks
        ]
        all_osi: set[str] = {osi for _, legs in pick_legs for osi in legs}

        if not all_osi:
            return picks, []

        _log.info("[preflight] Checking %d contract(s) on Alpaca ...", len(all_osi))

        tradeable: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(_check, osi): osi for osi in all_osi}
            for fut in as_completed(futures):
                tradeable[futures[fut]] = fut.result()

        valid: list[dict] = []
        filtered: list[dict] = []
        for pick, legs in pick_legs:
            bad = [o for o in legs if not tradeable.get(o, False)]
            if bad:
                filtered.append({**pick, '_inactive_contracts': bad})
            else:
                valid.append(pick)

        if filtered:
            _log.warning("[preflight] Removed %d pick(s) with inactive legs:", len(filtered))
            for p in filtered:
                _log.warning("  %-6s %-6s  %s",
                             p.get('strategy', '?'), p.get('symbol', '?'),
                             p['_inactive_contracts'])
        active_count = sum(v for v in tradeable.values())
        _log.info("[preflight] %d/%d contracts active — %d/%d picks fully tradeable.",
                  active_count, len(tradeable), len(valid), len(picks))
        return valid, filtered

    # ── Fill-price query ──────────────────────────────────────────────────────

    def get_fill_price(
        self, order_id: str, timeout_seconds: int = 60
    ) -> tuple[float | None, bool]:
        """
        Poll Alpaca until *order_id* is filled or *timeout_seconds* elapses.

        Returns ``(fill_price, canceled)`` where:
          - ``fill_price`` is the ``filled_avg_price`` (net credit for MLEG
            orders), or None if the order did not fill.
          - ``canceled`` is True when Alpaca confirmed the order will never
            fill (canceled / expired / rejected / done_for_day), meaning any
            DB record for this order should be voided.  False means either the
            order filled or the outcome is still unknown (timeout).

        Requires the executor to be logged in (live orders only).
        """
        import time as _time

        if not self.is_logged_in:
            return None, False

        _FILLED   = {'filled', 'partially_filled'}
        _TERMINAL = {'canceled', 'expired', 'rejected', 'done_for_day'}

        deadline = _time.time() + timeout_seconds
        while _time.time() < deadline:
            try:
                order = self.client.get_order_by_id(order_id)
                status = normalize_order_status(getattr(order, 'status', ''))
                if status in _FILLED:
                    price = order.filled_avg_price
                    return (float(price) if price is not None else None), False
                if status in _TERMINAL:
                    _log.warning(
                        "[executor] Order %s ended with status '%s' — voiding DB entry",
                        order_id, status,
                    )
                    return None, True   # caller should void the trade record
            except Exception as exc:
                _log.warning("[executor] get_fill_price: could not fetch order %s: %s",
                             order_id, exc)
                return None, False
            _time.sleep(5)

        # Timeout: the limit order has not filled.  Actively cancel it on Alpaca
        # so it cannot fill later at a stale price, then signal the caller to void
        # the DB record.  Leaving a timed-out DAY order alive risks the position
        # filling hours later (at a stale entry price) or the order expiring at
        # close while the DB still shows the position as EXECUTED.
        try:
            self.client.cancel_order_by_id(order_id)
            _log.warning(
                "[executor] Order %s not filled within %ds — canceled on Alpaca, "
                "DB entry will be voided.",
                order_id, timeout_seconds,
            )
        except Exception as exc:
            _log.warning(
                "[executor] Order %s not filled within %ds — cancel attempt failed "
                "(%s); DB entry will still be voided. Verify manually on Alpaca.",
                order_id, timeout_seconds, exc,
            )
        return None, True   # treat as canceled — caller voids the DB record

    def _get_option_position_qty_by_symbol(self) -> dict[str, float] | None:
        """
        Return signed option quantities keyed by OSI symbol, or None if the
        broker position snapshot is unavailable.

        Alpaca rejects ``SELL_TO_CLOSE`` when the account does not hold the
        long leg.  Filtering close legs through this snapshot prevents a stale
        DB spread from sending a close order that Alpaca interprets as
        ``SELL_TO_OPEN``.
        """
        if self.client is None:
            return None
        try:
            positions = self.client.get_all_positions()
        except Exception as exc:
            _log.warning(
                "[executor] Could not fetch Alpaca positions before close (%s); "
                "submitting close order from DB legs.",
                exc,
            )
            return None

        if not isinstance(positions, (list, tuple, set)):
            # Unit tests often use a bare MagicMock client.  Fall back to the
            # historical DB-leg behaviour unless we have a concrete snapshot.
            return None

        result: dict[str, float] = {}
        for p in positions:
            ac = str(getattr(p, 'asset_class', '') or '').lower()
            sym = str(getattr(p, 'symbol', '') or '')
            if not sym or ('option' not in ac and len(sym) < 15):
                continue
            qty = _signed_option_qty(p)
            if qty is not None and qty != 0:
                result[sym] = qty
        return result

    # ── Close (buy-to-close) dispatcher ───────────────────────────────────────

    def execute_close_position(
        self,
        pos: dict,
        limit_price: float = None,
        order_type: str = 'limit',
        amount: int = 1,
        dry_run: bool = True,
    ) -> str | None:
        """
        Close an open position by submitting buy-to-close / sell-to-close orders.

        *pos* is a database row dict as returned by TradeDatabase.get_open_positions().
        Required keys: symbol, expiry, type, legs (dict).

        The closing order reverses every leg:
          short (SELL_TO_OPEN) → BUY_TO_CLOSE
          long  (BUY_TO_OPEN)  → SELL_TO_CLOSE

        Returns the broker order ID on success, or None on failure.
        """
        import json as _json

        symbol = pos.get('symbol', '')
        expiry = pos.get('expiry', '')
        strat  = pos.get('type',   '')
        legs   = pos.get('legs') or {}
        if isinstance(legs, str):
            try:
                legs = _json.loads(legs)
            except Exception:
                legs = {}

        order_type = str(order_type or 'limit').strip().lower()
        if order_type not in {'limit', 'market'}:
            _log.warning(
                "[executor] Unsupported close order_type '%s' for %s %s; falling back to limit.",
                order_type, strat, symbol,
            )
            order_type = 'limit'

        if dry_run:
            _log.info("[DRY RUN] CLOSE %s %s %s  legs=%s  type=%s  limit=$%s",
                      strat, symbol, expiry, legs, order_type, limit_price or 'market')
            return "DRY_RUN_CLOSE"

        if not self.is_logged_in:
            if not self.login():
                return None

        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, OptionLegRequest
        from alpaca.trading.enums import OrderClass

        # Use `is not None` (not truthiness) so that a mark of 0.0 isn't
        # silently dropped — a falsy 0.0 would produce limit_price=None which
        # Alpaca rejects.  Also floor to $0.01 (minimum tick) because a
        # $0.00 limit order will never fill.
        lp = max(0.01, round(limit_price, 2)) if limit_price is not None else None

        broker_qty = self._get_option_position_qty_by_symbol()
        multi_leg_close = strat in {'PCS', 'CCS', 'IC', 'IFLY', 'STRANGLE'}

        def _held(symbol_: str, expected_side: str) -> bool:
            if broker_qty is None:
                return True
            qty = broker_qty.get(symbol_)
            if qty is None:
                return False
            return qty < 0 if expected_side == 'short' else qty > 0

        def _add_leg(specs: list[dict], option_symbol: str, side, intent, expected_side: str) -> None:
            if multi_leg_close:
                # Alpaca's position snapshot can omit hedge legs for open MLEG
                # spreads.  Keep the full DB-defined close order intact instead
                # of silently degrading a spread close into a mismatched
                # single-leg order with the spread's net debit limit.
                if broker_qty is not None and not _held(option_symbol, expected_side):
                    _log.warning(
                        "[executor] Close %s %s: Alpaca position snapshot does "
                        "not show the expected %s leg %s; submitting the full "
                        "multi-leg close from DB legs.",
                        strat, symbol, expected_side, option_symbol,
                    )
                specs.append({
                    'symbol': option_symbol,
                    'side': side,
                    'intent': intent,
                    'expected_side': expected_side,
                })
                return

            if not _held(option_symbol, expected_side):
                _log.warning(
                    "[executor] Close %s %s: skipping %s leg %s because Alpaca "
                    "does not show a matching %s position.",
                    strat, symbol, expected_side, option_symbol, expected_side,
                )
                return
            specs.append({
                'symbol': option_symbol,
                'side': side,
                'intent': intent,
                'expected_side': expected_side,
            })

        def _build_close_order(specs: list[dict]):
            if not specs:
                _log.warning(
                    "[executor] Close %s %s: no matching Alpaca option legs found; "
                    "not submitting a close order.",
                    strat, symbol,
                )
                return None
            request_cls = MarketOrderRequest if order_type == 'market' else LimitOrderRequest
            if len(specs) == 1:
                leg = specs[0]
                kwargs = dict(
                    symbol=leg['symbol'],
                    qty=amount,
                    side=leg['side'],
                    type=OrderType.MARKET if order_type == 'market' else OrderType.LIMIT,
                    time_in_force=TimeInForce.DAY,
                    position_intent=leg['intent'],
                )
                if order_type == 'limit':
                    kwargs['limit_price'] = lp
                return request_cls(**kwargs)
            kwargs = dict(
                order_class=OrderClass.MLEG, qty=amount,
                type=OrderType.MARKET if order_type == 'market' else OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                legs=[
                    OptionLegRequest(
                        symbol=leg['symbol'],
                        side=leg['side'],
                        ratio_qty=1,
                        position_intent=leg['intent'],
                    )
                    for leg in specs
                ],
            )
            if order_type == 'limit':
                kwargs['limit_price'] = lp
            return request_cls(**kwargs)

        try:
            if strat == 'PCS':
                ss = legs.get('short_strike') or legs.get('short_put') or pos.get('strike')
                ls = legs.get('long_strike')  or legs.get('long_put')
                if ss is None or ls is None:
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, ss, 'PUT'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                _add_leg(specs, _osi_symbol(symbol, expiry, ls, 'PUT'),
                         OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, 'long')
                order = _build_close_order(specs)
                if order is None:
                    return None

            elif strat == 'CCS':
                ss = legs.get('short_strike') or legs.get('short_call') or pos.get('strike')
                ls = legs.get('long_strike')  or legs.get('long_call')
                if ss is None or ls is None:
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, ss, 'CALL'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                _add_leg(specs, _osi_symbol(symbol, expiry, ls, 'CALL'),
                         OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, 'long')
                order = _build_close_order(specs)
                if order is None:
                    return None

            elif strat in ('IC', 'IFLY'):
                sp   = legs.get('short_put')
                lp_k = legs.get('long_put')
                sc   = legs.get('short_call')
                lc   = legs.get('long_call')
                if any(x is None for x in (sp, lp_k, sc, lc)):
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, sp, 'PUT'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                _add_leg(specs, _osi_symbol(symbol, expiry, lp_k, 'PUT'),
                         OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, 'long')
                _add_leg(specs, _osi_symbol(symbol, expiry, sc, 'CALL'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                _add_leg(specs, _osi_symbol(symbol, expiry, lc, 'CALL'),
                         OrderSide.SELL, PositionIntent.SELL_TO_CLOSE, 'long')
                order = _build_close_order(specs)
                if order is None:
                    return None

            elif strat == 'CSP':
                ss = legs.get('short_strike') or pos.get('strike')
                if ss is None:
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, ss, 'PUT'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                order = _build_close_order(specs)
                if order is None:
                    return None

            elif strat == 'CC':
                ss = legs.get('short_strike') or pos.get('strike')
                if ss is None:
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, ss, 'CALL'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                order = _build_close_order(specs)
                if order is None:
                    return None

            elif strat == 'STRANGLE':
                sp = legs.get('short_put') or pos.get('strike')
                sc = legs.get('short_call')
                if sp is None or sc is None:
                    _log.warning("[executor] STRANGLE close: missing strikes for %s "
                                 "(short_put=%s, short_call=%s)", symbol, sp, sc)
                    return None
                specs = []
                _add_leg(specs, _osi_symbol(symbol, expiry, sp, 'PUT'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                _add_leg(specs, _osi_symbol(symbol, expiry, sc, 'CALL'),
                         OrderSide.BUY, PositionIntent.BUY_TO_CLOSE, 'short')
                order = _build_close_order(specs)
                if order is None:
                    return None

            else:
                _log.warning("[executor] execute_close_position: unknown strategy '%s'", strat)
                return None

            res = self.client.submit_order(order)
            _log.info("[executor] Close order submitted for %s %s: %s", strat, symbol, res.id)
            return str(res.id)

        except Exception as exc:
            related_order_id = _extract_held_for_orders_order_id(exc)
            if related_order_id:
                _log.warning(
                    "[executor] Close order for %s %s already in flight "
                    "(related order=%s); reusing it instead of submitting another.",
                    strat, symbol, related_order_id,
                )
                return related_order_id
            _log.error("[executor] execute_close_position error for %s %s: %s",
                       strat, symbol, exc, exc_info=True)
            return None
