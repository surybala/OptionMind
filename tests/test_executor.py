import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Mock alpaca-py modules before importing executor so no real SDK is required.
_alpaca_mock = MagicMock()
for _mod in ['alpaca', 'alpaca.trading', 'alpaca.trading.client',
             'alpaca.trading.enums', 'alpaca.trading.requests']:
    sys.modules.setdefault(_mod, _alpaca_mock)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_SAMPLE_CONFIG = {
    'alpaca': {
        'api_key': 'test_key_abc',
        'api_secret': 'test_secret_xyz',
        'paper': True,
        'high_confidence_mode': False,
    }
}

_EMPTY_CREDS_CONFIG = {
    'alpaca': {'api_key': '', 'api_secret': '', 'paper': True}
}


def _make_executor(config=_SAMPLE_CONFIG):
    with patch('src.executor.load_config', return_value=config):
        from src.executor import AlpacaExecutor
        return AlpacaExecutor()


class TestExecutorDryRun(unittest.TestCase):

    def setUp(self):
        self.executor = _make_executor()

    def test_execute_sell_put_dry_run_returns_dry_run_id(self):
        result = self.executor.execute_sell_put('AAPL', '2026-04-30', 150.0, dry_run=True)
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_execute_sell_spread_dry_run_returns_dry_run_id(self):
        result = self.executor.execute_sell_spread('AAPL', '2026-04-30', 150.0, 145.0, 'PCS', dry_run=True)
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_execute_sell_put_dry_run_does_not_login(self):
        self.executor.execute_sell_put('AAPL', '2026-04-30', 150.0, dry_run=True)
        self.assertFalse(self.executor.is_logged_in)

    def test_execute_sell_spread_dry_run_does_not_login(self):
        self.executor.execute_sell_spread('AAPL', '2026-04-30', 150.0, 145.0, 'CCS', dry_run=True)
        self.assertFalse(self.executor.is_logged_in)

    def test_execute_sell_spread_pcs_strategy(self):
        result = self.executor.execute_sell_spread('SPY', '2026-04-30', 480.0, 475.0, 'PCS', dry_run=True)
        self.assertEqual(result, 'DRY_RUN_ID')

    def test_execute_sell_spread_ccs_strategy(self):
        result = self.executor.execute_sell_spread('SPY', '2026-04-30', 520.0, 525.0, 'CCS', dry_run=True)
        self.assertEqual(result, 'DRY_RUN_ID')


class TestExecutorLogin(unittest.TestCase):

    def test_login_returns_false_when_credentials_missing(self):
        executor = _make_executor(_EMPTY_CREDS_CONFIG)
        result = executor.login()
        self.assertFalse(result)
        self.assertFalse(executor.is_logged_in)

    def test_login_sets_logged_in_with_valid_credentials(self):
        executor = _make_executor(_SAMPLE_CONFIG)
        result = executor.login()
        self.assertTrue(result)
        self.assertTrue(executor.is_logged_in)

    def test_initial_state_is_not_logged_in(self):
        executor = _make_executor()
        self.assertFalse(executor.is_logged_in)


class TestExecutorLiveMode(unittest.TestCase):
    """execute_sell_put and execute_sell_spread with dry_run=False trigger login."""

    def test_sell_put_live_mode_attempts_login_if_not_logged_in(self):
        executor = _make_executor(_EMPTY_CREDS_CONFIG)
        result = executor.execute_sell_put('AAPL', '2026-04-30', 150.0, dry_run=False)
        self.assertIsNone(result)

    def test_sell_spread_live_mode_attempts_login_if_not_logged_in(self):
        executor = _make_executor(_EMPTY_CREDS_CONFIG)
        result = executor.execute_sell_spread('AAPL', '2026-04-30', 150.0, 145.0, 'PCS', dry_run=False)
        self.assertIsNone(result)


class TestOsiSymbol(unittest.TestCase):
    """Verify the OSI/OCC symbol builder produces correct strings."""

    def setUp(self):
        from src.executor import _osi_symbol
        self._osi = _osi_symbol

    def test_put_symbol_format(self):
        sym = self._osi('AAPL', '2026-04-30', 150.0, 'PUT')
        self.assertEqual(sym, 'AAPL260430P00150000')

    def test_call_symbol_format(self):
        sym = self._osi('SPY', '2026-01-17', 500.0, 'CALL')
        self.assertEqual(sym, 'SPY260117C00500000')

    def test_fractional_strike(self):
        # $152.50 → 152500 units
        sym = self._osi('TSLA', '2026-06-20', 152.5, 'PUT')
        self.assertEqual(sym, 'TSLA260620P00152500')

    def test_single_char_symbol_no_padding(self):
        # Alpaca compact format — no space padding on root
        sym = self._osi('X', '2026-03-21', 20.0, 'CALL')
        self.assertEqual(sym, 'X260321C00020000')

    def test_five_char_symbol_no_padding(self):
        sym = self._osi('GOOGL', '2026-03-21', 180.0, 'PUT')
        self.assertEqual(sym, 'GOOGL260321P00180000')

    def test_high_strike_symbol(self):
        # AZO ~$3820 — the exact case that triggered the API error
        sym = self._osi('AZO', '2026-03-20', 3820.0, 'CALL')
        self.assertEqual(sym, 'AZO260320C03820000')


class TestLimitOrderRequestMleg(unittest.TestCase):
    """
    Verify that live-mode multi-leg submissions use LimitOrderRequest with
    order_class=OrderClass.MLEG.

    Background: alpaca-py 0.43.x removed CreateMultiLegOrderRequest and
    OrderClass.COMBO.  The fix routes all multi-leg orders through
    LimitOrderRequest(order_class=OrderClass.MLEG, legs=[...]).

    Setup intercepts the mocked alpaca.trading.requests and
    alpaca.trading.enums so we can assert on the exact arguments passed to
    the request constructor without touching a real broker.
    """

    def setUp(self):
        # Logged-in executor with a mock broker client
        self.executor = _make_executor(_SAMPLE_CONFIG)
        self.executor.is_logged_in = True
        self.executor.client = MagicMock()
        self.executor.client.submit_order.return_value = MagicMock(id='order-123')

        _reqs  = sys.modules['alpaca.trading.requests']
        _enums = sys.modules['alpaca.trading.enums']

        # LimitOrderRequest — freshly replaced so call_args is clean per test
        self._LimitOrderRequest = MagicMock(return_value=MagicMock(name='limit_order'))
        _reqs.LimitOrderRequest = self._LimitOrderRequest
        self._MarketOrderRequest = MagicMock(return_value=MagicMock(name='market_order'))
        _reqs.MarketOrderRequest = self._MarketOrderRequest

        # OptionLegRequest — side_effect returns kwargs dict so leg fields
        # are inspectable without having to unwrap Mock objects
        self._OptionLegRequest = MagicMock(side_effect=lambda **kw: kw)
        _reqs.OptionLegRequest = self._OptionLegRequest

        # Unique sentinel for MLEG so assertIs comparisons are unambiguous
        self._mleg = object()
        _enums.OrderClass = MagicMock()
        _enums.OrderClass.MLEG = self._mleg

    # ── helpers ───────────────────────────────────────────────────────────────

    def _kw(self):
        """Return keyword args from the most recent order request constructor."""
        if self._MarketOrderRequest.called:
            return self._MarketOrderRequest.call_args.kwargs
        self.assertTrue(
            self._LimitOrderRequest.called,
            "LimitOrderRequest was never called — has CreateMultiLegOrderRequest crept back in?",
        )
        return self._LimitOrderRequest.call_args.kwargs

    def _legs(self):
        """Return the legs list from the most recent LimitOrderRequest() call."""
        return self._kw()['legs']

    @staticmethod
    def _osi(symbol, expiry, strike, opt):
        from src.executor import _osi_symbol
        return _osi_symbol(symbol, expiry, strike, opt)

    def _leg_by_sym(self, symbol):
        return next(l for l in self._legs() if l['symbol'] == symbol)

    def _broker_pos(self, symbol, side='short', qty='1'):
        p = MagicMock()
        p.symbol = symbol
        p.side = side
        p.qty = qty
        p.asset_class = 'us_option'
        return p

    # ── open: PCS — 2 put legs ────────────────────────────────────────────────

    def test_pcs_uses_limit_order_request(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        self.assertTrue(self._LimitOrderRequest.called)
        self.executor.client.submit_order.assert_called_once()

    def test_pcs_order_class_is_mleg(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_pcs_no_top_level_symbol_or_side(self):
        """MLEG orders must not carry a top-level symbol or side."""
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        kw = self._kw()
        self.assertNotIn('symbol', kw)
        self.assertNotIn('side', kw)

    def test_pcs_has_two_legs(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        self.assertEqual(len(self._legs()), 2)

    def test_pcs_limit_price_forwarded(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=2.25, dry_run=False
        )
        self.assertEqual(self._kw()['limit_price'], 2.25)

    def test_pcs_qty_forwarded(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS',
            limit_price=1.0, amount=3, dry_run=False
        )
        self.assertEqual(self._kw()['qty'], 3)

    def test_pcs_leg_symbols_are_put_osi(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        syms = {l['symbol'] for l in self._legs()}
        self.assertIn(self._osi('SPY', '2026-04-30', 480.0, 'PUT'), syms)
        self.assertIn(self._osi('SPY', '2026-04-30', 475.0, 'PUT'), syms)

    def test_pcs_short_put_is_sell_to_open(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 480.0, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_OPEN)

    def test_pcs_long_put_is_buy_to_open(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 480.0, 475.0, 'PCS', limit_price=1.50, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 475.0, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_OPEN)

    # ── open: CCS — 2 call legs ───────────────────────────────────────────────

    def test_ccs_order_class_is_mleg(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 520.0, 525.0, 'CCS', limit_price=1.50, dry_run=False
        )
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_ccs_has_two_legs(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 520.0, 525.0, 'CCS', limit_price=1.50, dry_run=False
        )
        self.assertEqual(len(self._legs()), 2)

    def test_ccs_leg_symbols_are_call_osi(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 520.0, 525.0, 'CCS', limit_price=1.50, dry_run=False
        )
        syms = {l['symbol'] for l in self._legs()}
        self.assertIn(self._osi('SPY', '2026-04-30', 520.0, 'CALL'), syms)
        self.assertIn(self._osi('SPY', '2026-04-30', 525.0, 'CALL'), syms)

    def test_ccs_short_call_is_sell_to_open(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 520.0, 525.0, 'CCS', limit_price=1.50, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 520.0, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_OPEN)

    def test_ccs_long_call_is_buy_to_open(self):
        self.executor.execute_sell_spread(
            'SPY', '2026-04-30', 520.0, 525.0, 'CCS', limit_price=1.50, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 525.0, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_OPEN)

    # ── open: Iron Condor — 4 legs ────────────────────────────────────────────

    def test_ic_order_class_is_mleg(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_ic_has_four_legs(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        self.assertEqual(len(self._legs()), 4)

    def test_ic_all_four_osi_symbols_present(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        syms = {l['symbol'] for l in self._legs()}
        self.assertIn(self._osi('AAPL', '2026-04-30', 85,  'PUT'),  syms)
        self.assertIn(self._osi('AAPL', '2026-04-30', 80,  'PUT'),  syms)
        self.assertIn(self._osi('AAPL', '2026-04-30', 115, 'CALL'), syms)
        self.assertIn(self._osi('AAPL', '2026-04-30', 120, 'CALL'), syms)

    def test_ic_short_put_is_sell_to_open(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 85, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_OPEN)

    def test_ic_long_put_is_buy_to_open(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 80, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_OPEN)

    def test_ic_short_call_is_sell_to_open(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 115, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_OPEN)

    def test_ic_long_call_is_buy_to_open(self):
        self.executor.execute_sell_iron_condor(
            'AAPL', '2026-04-30', 85, 80, 115, 120, limit_price=2.0, dry_run=False
        )
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 120, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_OPEN)

    # ── open: Iron Butterfly — 4 legs ─────────────────────────────────────────

    def test_ifly_order_class_is_mleg(self):
        self.executor.execute_sell_iron_butterfly(
            'AAPL', '2026-04-30', 100, 90, 100, 110, limit_price=3.0, dry_run=False
        )
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_ifly_has_four_legs(self):
        self.executor.execute_sell_iron_butterfly(
            'AAPL', '2026-04-30', 100, 90, 100, 110, limit_price=3.0, dry_run=False
        )
        self.assertEqual(len(self._legs()), 4)

    # ── open: Strangle — 2 short legs ────────────────────────────────────────

    def test_strangle_order_class_is_mleg(self):
        self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, limit_price=2.0, dry_run=False
        )
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_strangle_has_two_legs(self):
        self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, limit_price=2.0, dry_run=False
        )
        self.assertEqual(len(self._legs()), 2)

    def test_strangle_both_legs_sell_to_open(self):
        self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, limit_price=2.0, dry_run=False
        )
        from src.executor import PositionIntent
        for leg in self._legs():
            self.assertIs(
                leg['position_intent'], PositionIntent.SELL_TO_OPEN,
                f"Leg {leg['symbol']} should be SELL_TO_OPEN",
            )

    def test_strangle_leg_symbols_put_and_call(self):
        self.executor.execute_sell_strangle(
            'AAPL', '2026-04-30', 80, 120, limit_price=2.0, dry_run=False
        )
        syms = {l['symbol'] for l in self._legs()}
        self.assertIn(self._osi('AAPL', '2026-04-30', 80,  'PUT'),  syms)
        self.assertIn(self._osi('AAPL', '2026-04-30', 120, 'CALL'), syms)

    # ── close: PCS ────────────────────────────────────────────────────────────

    @staticmethod
    def _pos(strat, legs_dict, symbol='SPY', expiry='2026-04-30', strike=480.0):
        return {'symbol': symbol, 'expiry': expiry, 'type': strat,
                'legs': legs_dict, 'strike': strike}

    def test_close_pcs_uses_limit_order_request(self):
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        self.assertTrue(self._LimitOrderRequest.called)
        self.executor.client.submit_order.assert_called_once()

    def test_close_pcs_uses_market_order_request_when_requested(self):
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(
            pos,
            limit_price=None,
            order_type='market',
            dry_run=False,
        )
        self.assertTrue(self._MarketOrderRequest.called)
        self.assertFalse(self._LimitOrderRequest.called)
        self.executor.client.submit_order.assert_called_once()

    def test_close_reuses_related_order_when_qty_held_for_orders(self):
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.client.submit_order.side_effect = RuntimeError(
            '{"available":"0","code":40310000,"existing_qty":"1",'
            '"held_for_orders":"1","message":"insufficient qty available for order",'
            '"related_orders":["existing-close-123"],'
            '"symbol":"SPY260430P00480000"}'
        )

        order_id = self.executor.execute_close_position(
            pos, limit_price=0.50, dry_run=False
        )

        self.assertEqual(order_id, 'existing-close-123')

    def test_close_pcs_order_class_is_mleg(self):
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_close_pcs_has_two_legs(self):
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        self.assertEqual(len(self._legs()), 2)

    def test_close_pcs_keeps_mleg_when_long_missing_from_broker_snapshot(self):
        """Incomplete Alpaca snapshots must not downgrade a spread close."""
        short_sym = self._osi('SPY', '2026-04-30', 480.0, 'PUT')
        long_sym = self._osi('SPY', '2026-04-30', 475.0, 'PUT')
        self.executor.client.get_all_positions.return_value = [
            self._broker_pos(short_sym, side='short', qty='1'),
        ]
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})

        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)

        kw = self._kw()
        self.assertIs(kw['order_class'], self._mleg)
        self.assertEqual(len(kw['legs']), 2)
        syms = {leg['symbol'] for leg in kw['legs']}
        self.assertEqual(syms, {short_sym, long_sym})

    def test_close_pcs_short_is_buy_to_close(self):
        """Reversing PCS: the original short put is now bought to close."""
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 480.0, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_CLOSE)

    def test_close_pcs_long_is_sell_to_close(self):
        """Reversing PCS: the original long put is now sold to close."""
        pos = self._pos('PCS', {'short_strike': 480.0, 'long_strike': 475.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 475.0, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_CLOSE)

    # ── close: CCS ────────────────────────────────────────────────────────────

    def test_close_ccs_order_class_is_mleg(self):
        pos = self._pos('CCS', {'short_strike': 520.0, 'long_strike': 525.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_close_ccs_has_two_legs(self):
        pos = self._pos('CCS', {'short_strike': 520.0, 'long_strike': 525.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        self.assertEqual(len(self._legs()), 2)

    def test_close_ccs_short_is_buy_to_close(self):
        pos = self._pos('CCS', {'short_strike': 520.0, 'long_strike': 525.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 520.0, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_CLOSE)

    def test_close_ccs_long_is_sell_to_close(self):
        pos = self._pos('CCS', {'short_strike': 520.0, 'long_strike': 525.0})
        self.executor.execute_close_position(pos, limit_price=0.50, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('SPY', '2026-04-30', 525.0, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_CLOSE)

    # ── close: IC — 4 legs ───────────────────────────────────────────────────

    def test_close_ic_order_class_is_mleg(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        self.assertIs(self._kw()['order_class'], self._mleg)

    def test_close_ic_has_four_legs(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        self.assertEqual(len(self._legs()), 4)

    def test_close_ic_short_put_is_buy_to_close(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 85, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_CLOSE)

    def test_close_ic_long_put_is_sell_to_close(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 80, 'PUT'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_CLOSE)

    def test_close_ic_short_call_is_buy_to_close(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 115, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.BUY_TO_CLOSE)

    def test_close_ic_long_call_is_sell_to_close(self):
        pos = self._pos('IC', {'short_put': 85, 'long_put': 80,
                               'short_call': 115, 'long_call': 120}, symbol='AAPL')
        self.executor.execute_close_position(pos, limit_price=1.0, dry_run=False)
        from src.executor import PositionIntent
        leg = self._leg_by_sym(self._osi('AAPL', '2026-04-30', 120, 'CALL'))
        self.assertIs(leg['position_intent'], PositionIntent.SELL_TO_CLOSE)


class TestPreflightCheckPicks(unittest.TestCase):
    """Tests for AlpacaExecutor.preflight_check_picks()."""

    def setUp(self):
        self.executor = _make_executor()
        # Simulate a logged-in state so preflight doesn't try to call login()
        self.executor.is_logged_in = True
        self.executor.client = MagicMock()

    def _active_contract(self, tradable=True):
        c = MagicMock()
        c.tradable = tradable
        return c

    def _csp_pick(self, symbol='SPY', expiry='2026-04-30', short_strike=480.0):
        return {'strategy': 'CSP', 'symbol': symbol, 'expiry': expiry,
                'short_strike': short_strike}

    def _pcs_pick(self, symbol='SPY', expiry='2026-04-30', ss=480.0, ls=475.0):
        return {'strategy': 'PCS', 'symbol': symbol, 'expiry': expiry,
                'short_put': ss, 'long_put': ls}

    def _ccs_pick(self, symbol='AAPL', expiry='2026-04-30', ss=200.0, ls=205.0):
        return {'strategy': 'CCS', 'symbol': symbol, 'expiry': expiry,
                'short_call': ss, 'long_call': ls}

    def _ic_pick(self, symbol='QQQ', expiry='2026-04-30'):
        return {'strategy': 'IC', 'symbol': symbol, 'expiry': expiry,
                'short_put': 400.0, 'long_put': 395.0,
                'short_call': 450.0, 'long_call': 455.0}

    def _strangle_pick(self, symbol='TSLA', expiry='2026-04-30'):
        return {'strategy': 'STRANGLE', 'symbol': symbol, 'expiry': expiry,
                'short_put': 150.0, 'short_call': 250.0}

    # ── Skips when credentials are missing ───────────────────────────────────

    def test_no_credentials_returns_all_picks_unchanged(self):
        exc = _make_executor(_EMPTY_CREDS_CONFIG)
        picks = [self._csp_pick(), self._pcs_pick()]
        valid, filtered = exc.preflight_check_picks(picks)
        self.assertEqual(valid, picks)
        self.assertEqual(filtered, [])

    # ── All contracts active ──────────────────────────────────────────────────

    def test_all_active_returns_all_valid(self):
        self.executor.client.get_option_contract.return_value = self._active_contract(True)
        picks = [self._csp_pick(), self._pcs_pick()]
        valid, filtered = self.executor.preflight_check_picks(picks)
        self.assertEqual(len(valid), 2)
        self.assertEqual(filtered, [])

    def test_all_active_ic_four_legs_checked(self):
        self.executor.client.get_option_contract.return_value = self._active_contract(True)
        valid, filtered = self.executor.preflight_check_picks([self._ic_pick()])
        self.assertEqual(len(valid), 1)
        self.assertEqual(filtered, [])
        # IC has 4 unique legs → 4 calls
        self.assertEqual(self.executor.client.get_option_contract.call_count, 4)

    def test_strangle_two_legs_checked(self):
        self.executor.client.get_option_contract.return_value = self._active_contract(True)
        valid, filtered = self.executor.preflight_check_picks([self._strangle_pick()])
        self.assertEqual(len(valid), 1)
        self.assertEqual(filtered, [])
        self.assertEqual(self.executor.client.get_option_contract.call_count, 2)

    # ── One leg inactive / raises ─────────────────────────────────────────────

    def test_inactive_tradable_flag_filters_pick(self):
        """tradable=False on any leg → pick is removed."""
        self.executor.client.get_option_contract.return_value = self._active_contract(False)
        valid, filtered = self.executor.preflight_check_picks([self._csp_pick()])
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(filtered), 1)
        self.assertIn('_inactive_contracts', filtered[0])

    def test_api_error_on_contract_filters_pick(self):
        """APIError (e.g. 404) on any leg → pick is removed."""
        self.executor.client.get_option_contract.side_effect = Exception("contract not active")
        valid, filtered = self.executor.preflight_check_picks([self._csp_pick()])
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(filtered), 1)

    def test_partial_failure_filters_only_bad_picks(self):
        """Good picks pass; picks with an inactive leg are filtered."""
        good_pick  = self._csp_pick(symbol='SPY',  short_strike=480.0)
        bad_pick   = self._ccs_pick(symbol='AAPL', ss=200.0, ls=205.0)

        from src.executor import _osi_symbol
        good_osi = _osi_symbol('SPY',  good_pick['expiry'], 480.0, 'PUT')
        bad_osi1 = _osi_symbol('AAPL', bad_pick['expiry'],  200.0, 'CALL')
        bad_osi2 = _osi_symbol('AAPL', bad_pick['expiry'],  205.0, 'CALL')

        def _side_effect(osi):
            if osi in (bad_osi1, bad_osi2):
                raise Exception("not found")
            return self._active_contract(True)

        self.executor.client.get_option_contract.side_effect = _side_effect
        valid, filtered = self.executor.preflight_check_picks([good_pick, bad_pick])
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]['symbol'], 'SPY')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['symbol'], 'AAPL')

    # ── Empty input / no legs ─────────────────────────────────────────────────

    def test_empty_picks_returns_empty(self):
        valid, filtered = self.executor.preflight_check_picks([])
        self.assertEqual(valid, [])
        self.assertEqual(filtered, [])

    def test_pick_with_no_legs_passes_through(self):
        """An unknown strategy with no OSI legs passes (nothing to check)."""
        mystery_pick = {'strategy': 'UNKNOWN', 'symbol': 'XYZ', 'expiry': '2026-04-30'}
        valid, filtered = self.executor.preflight_check_picks([mystery_pick])
        self.assertEqual(len(valid), 1)
        self.assertEqual(filtered, [])
        self.executor.client.get_option_contract.assert_not_called()

    # ── Inactive contracts list is populated ─────────────────────────────────

    def test_inactive_contracts_key_lists_bad_osi_symbols(self):
        from src.executor import _osi_symbol
        pick   = self._pcs_pick(ss=480.0, ls=475.0)
        bad_ss = _osi_symbol('SPY', pick['expiry'], 480.0, 'PUT')
        bad_ls = _osi_symbol('SPY', pick['expiry'], 475.0, 'PUT')
        self.executor.client.get_option_contract.side_effect = Exception("not found")
        _, filtered = self.executor.preflight_check_picks([pick])
        self.assertIn(bad_ss, filtered[0]['_inactive_contracts'])
        self.assertIn(bad_ls, filtered[0]['_inactive_contracts'])


if __name__ == '__main__':
    unittest.main()
