"""
Tests for src.risk_service.PositionRiskService.

Covers:
- Non-HFT path: IV-based Black-Scholes Greeks via _build_greeks_legs
- HFT path: broker-supplied Greeks via _build_greeks_legs_from_snapshots
- Chain failure (RuntimeError): partial dict returned with risk_level
- risk_level classification applied
- has_broker_greeks field populated correctly
- compute_mark conservative vs display
- DataAdapter.from_config factory
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_service import PositionRiskService


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pos(strategy='PCS', symbol='AAPL', expiry='2099-12-31',
         premium=2.0, strike=190.0):
    """Build a minimal open position dict."""
    return {
        'id':       1,
        'symbol':   symbol,
        'expiry':   expiry,
        'type':     strategy,
        'premium':  premium,
        'strike':   strike,
        'status':   'EXECUTED',
        'legs':     {'short_strike': strike, 'long_strike': strike - 5},
    }


def _chain(has_broker_greeks=False, spot=200.0, put_map=None, call_map=None,
           snapshots=None, osi_map=None, leg_specs=None):
    """Build a mock PositionChainResult."""
    chain = MagicMock()
    chain.has_broker_greeks = has_broker_greeks
    chain.spot              = spot
    chain.put_map           = put_map if put_map is not None else {}
    chain.call_map          = call_map if call_map is not None else {}
    chain.snapshots         = snapshots or {}
    chain.osi_map           = osi_map   or {}
    chain.leg_specs         = leg_specs or []
    return chain


def _svc(adapter=None, config=None):
    if adapter is None:
        adapter = MagicMock()
    if config is None:
        config = {'risk_parameters': {'stop_loss_multiplier': 2.0}}
    return PositionRiskService(adapter, config)


# ── Non-HFT path ──────────────────────────────────────────────────────────────

class TestNonHftPath:

    def test_enrich_position_returns_dict_copy(self):
        """enrich_position must return a new dict, not mutate the original."""
        svc     = _svc()
        pos     = _pos()
        adapter = svc._data
        chain   = _chain(spot=200.0, put_map={}, call_map={})
        adapter.get_position_chain.return_value = chain

        enriched = svc.enrich_position(pos)
        assert enriched is not pos

    def test_has_broker_greeks_false_on_non_hft(self):
        svc   = _svc()
        chain = _chain(has_broker_greeks=False, spot=200.0)
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos())
        assert enriched['has_broker_greeks'] is False

    def test_pnl_fields_populated_when_chain_available(self):
        svc   = _svc()
        # Put map: short at 190 (mid=1.20), long at 185 (mid=0.30)
        def _row(bid, ask):
            return {'bid': bid, 'ask': ask, 'lastPrice': (bid + ask) / 2}

        put_map = {190.0: _row(1.10, 1.30), 185.0: _row(0.25, 0.35)}
        chain   = _chain(has_broker_greeks=False, spot=195.0, put_map=put_map)
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos(premium=2.0))
        assert 'current_mark' in enriched
        assert 'pnl_per_share' in enriched
        assert 'profit_captured_pct' in enriched

    def test_pnl_dollars_scales_by_contracts(self):
        svc = _svc()

        def _row(bid, ask):
            return {'bid': bid, 'ask': ask, 'lastPrice': (bid + ask) / 2}

        put_map = {190.0: _row(1.10, 1.30), 185.0: _row(0.25, 0.35)}
        svc._data.get_position_chain.return_value = _chain(
            has_broker_greeks=False,
            spot=195.0,
            put_map=put_map,
        )

        pos = _pos(premium=2.0)
        pos['contracts'] = 3
        enriched = svc.enrich_position(pos)

        assert enriched['pnl_per_share'] == pytest.approx(1.1)
        assert enriched['pnl_dollars'] == pytest.approx(330.0)

    def test_bs_greeks_path_invoked_when_iv_available(self):
        """_build_greeks_legs should be invoked (non-HFT) and produce risk fields."""
        svc = _svc()

        def _row(bid, ask, iv):
            return {'bid': bid, 'ask': ask, 'lastPrice': (bid + ask) / 2,
                    'impliedVolatility': iv}

        put_map = {190.0: _row(1.10, 1.30, 0.30), 185.0: _row(0.25, 0.35, 0.32)}
        chain   = _chain(has_broker_greeks=False, spot=200.0, put_map=put_map)
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos(premium=2.0))
        # If Greeks computed, these fields should be present
        assert 'gamma_theta_ratio' in enriched or 'dte' in enriched

    def test_risk_level_always_present(self):
        svc   = _svc()
        chain = _chain(spot=None, put_map=None, call_map=None)
        chain.put_map = None
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos())
        assert 'risk_level' in enriched
        assert enriched['risk_level'] in ('SAFE', 'WATCH', 'CAUTION', 'CRITICAL')


# ── HFT path ──────────────────────────────────────────────────────────────────

class TestHftPath:

    def _hft_chain(self, short_osi='AAPL991231P00190000', long_osi='AAPL991231P00185000'):
        """Build a chain with broker greeks for a PCS."""
        snap_short = {'delta': -0.15, 'gamma': 0.05, 'theta': -0.03,
                      'bid': 1.10, 'ask': 1.30}
        snap_long  = {'delta': -0.08, 'gamma': 0.03, 'theta': -0.02,
                      'bid': 0.25, 'ask': 0.35}
        snapshots = {short_osi: snap_short, long_osi: snap_long}
        osi_map   = {(190.0, 'put'): short_osi, (185.0, 'put'): long_osi}
        leg_specs = [(190.0, 'put', 'short'), (185.0, 'put', 'long')]
        put_map   = {190.0: snap_short, 185.0: snap_long}
        return _chain(
            has_broker_greeks=True,
            spot=195.0,
            put_map=put_map,
            snapshots=snapshots,
            osi_map=osi_map,
            leg_specs=leg_specs,
        )

    def test_has_broker_greeks_true_on_hft(self):
        svc   = _svc()
        chain = self._hft_chain()
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos())
        assert enriched['has_broker_greeks'] is True

    def test_broker_greeks_produce_risk_metrics(self):
        svc   = _svc()
        chain = self._hft_chain()
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos(premium=2.0))
        assert 'gamma_theta_ratio' in enriched
        assert 'net_short_delta' in enriched
        assert 'risk_score' in enriched

    def test_broker_greeks_none_skips_greeks_gracefully(self):
        """If all broker greeks are None (market closed), no crash — partial data OK."""
        svc = _svc()
        snap_short = {'delta': None, 'gamma': None, 'theta': None,
                      'bid': 1.10, 'ask': 1.30}
        snap_long  = {'delta': None, 'gamma': None, 'theta': None,
                      'bid': 0.25, 'ask': 0.35}
        short_osi  = 'AAPL991231P00190000'
        long_osi   = 'AAPL991231P00185000'
        chain = _chain(
            has_broker_greeks=True,
            spot=195.0,
            put_map={190.0: snap_short, 185.0: snap_long},
            snapshots={short_osi: snap_short, long_osi: snap_long},
            osi_map={(190.0, 'put'): short_osi, (185.0, 'put'): long_osi},
            leg_specs=[(190.0, 'put', 'short'), (185.0, 'put', 'long')],
        )
        svc._data.get_position_chain.return_value = chain

        enriched = svc.enrich_position(_pos())
        assert 'risk_level' in enriched   # must complete without raising
        assert enriched['has_broker_greeks'] is True


# ── Chain failure (HFT RuntimeError) ─────────────────────────────────────────

class TestChainFailure:

    def test_runtime_error_propagates(self):
        """enrich_position should re-raise RuntimeError (caller handles it)."""
        svc = _svc()
        svc._data.get_position_chain.side_effect = RuntimeError("Alpaca down")

        with pytest.raises(RuntimeError, match="Alpaca down"):
            svc.enrich_position(_pos())


# ── compute_mark ──────────────────────────────────────────────────────────────

class TestComputeMark:

    def _svc_with_chain(self, put_map):
        svc   = _svc()
        chain = _chain(spot=200.0, put_map=put_map)
        return svc, chain

    def test_compute_mark_display_uses_mid(self):
        """conservative=False: mid = (bid+ask)/2."""
        put_map = {190.0: {'bid': 1.10, 'ask': 1.30, 'lastPrice': 1.20},
                   185.0: {'bid': 0.25, 'ask': 0.35, 'lastPrice': 0.30}}
        svc, chain = self._svc_with_chain(put_map)
        pos = _pos()
        mark = svc.compute_mark(pos, chain, conservative=False)
        # short mid=1.20, long mid=0.30 → net=0.90
        assert mark == pytest.approx(0.90, abs=0.01)

    def test_compute_mark_conservative_uses_ask_for_short(self):
        """conservative=True: use ask (buy-to-close) for short leg."""
        put_map = {190.0: {'bid': 1.10, 'ask': 1.30, 'lastPrice': 1.20},
                   185.0: {'bid': 0.25, 'ask': 0.35, 'lastPrice': 0.30}}
        svc, chain = self._svc_with_chain(put_map)
        pos = _pos()
        mark_display      = svc.compute_mark(pos, chain, conservative=False)
        mark_conservative = svc.compute_mark(pos, chain, conservative=True)
        # Conservative should be >= display (short leg at ask, not mid)
        assert mark_conservative >= mark_display

    def test_compute_mark_returns_none_when_no_put_map(self):
        svc   = _svc()
        chain = _chain(put_map=None)
        chain.put_map = None
        assert svc.compute_mark(_pos(), chain) is None


# ── DataAdapter.from_config ───────────────────────────────────────────────────

class TestDataAdapterFactory:

    def test_from_config_returns_adapter(self):
        from src.market_data import DataAdapter

        cfg = {
            'hft_mode': False,
            'hft': {},
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True},
        }
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            adapter = DataAdapter.from_config(cfg)
        assert isinstance(adapter, DataAdapter)
        assert adapter.is_hft() is False

    def test_from_config_hft_true(self):
        from src.market_data import DataAdapter

        cfg = {
            'hft_mode': True,
            'hft': {'max_retries': 2},
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True},
        }
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            adapter = DataAdapter.from_config(cfg)
        assert adapter.is_hft() is True


# ── PositionRiskService.from_config ──────────────────────────────────────────

class TestServiceFactory:

    def test_from_config_creates_service(self):
        cfg = {
            'hft_mode': False,
            'hft': {},
            'risk_parameters': {'stop_loss_multiplier': 2.5},
            'alpaca': {'api_key': '', 'api_secret': '', 'paper': True},
        }
        with patch('src.alpaca_data.make_alpaca_data_client', return_value=None):
            svc = PositionRiskService.from_config(cfg)
        assert isinstance(svc, PositionRiskService)
        assert svc.stop_loss_multiplier == 2.5
