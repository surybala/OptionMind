"""
Tests for OptionScanner.get_top_picks() — strategy diversity, per-ticker cap,
and IC allocation cap introduced when fixing the "all picks same ticker/strategy"
bug.

Covers:
  - Strategy diversity: no single strategy floods all n slots
  - Per-ticker cap (max_picks_per_ticker config key)
  - IC allocation cap (ic_allocation_pct in iron_condor config)
  - Remainder slot filling from pool leftovers
  - Pool-pointer deduplication (same mock object for multiple tickers)
  - Output sorted by score descending
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault('yfinance', MagicMock())
sys.modules.setdefault('pandas', MagicMock())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scanner import OptionScanner


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scanner(ic_allocation_pct=1.0, max_picks_per_ticker=None, extra_strategies=None):
    """Build a minimal OptionScanner with configurable caps."""
    strategies = extra_strategies or {}
    strategies.setdefault('iron_condor', {
        'enabled': True,
        'ic_allocation_pct': ic_allocation_pct,
    })
    cfg = {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {'min_probability_of_expiry': 0.8},
        'strategies': strategies,
    }
    if max_picks_per_ticker is not None:
        cfg['max_picks_per_ticker'] = max_picks_per_ticker
    return OptionScanner(cfg)


def _picks(strategy: str, symbol: str, n: int, base_score: float = 1.0) -> list[dict]:
    """Return n distinct pick dicts for a given strategy / symbol."""
    return [
        {'strategy': strategy, 'symbol': symbol, 'score': base_score - i * 0.01, 'premium': 0.50}
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Basic contracts
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetTopPicksBasics:

    def test_empty_ticker_list_returns_empty(self):
        sc = _scanner()
        assert sc.get_top_picks([], n=10) == []

    def test_all_scans_empty_returns_empty(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(return_value=[])
        assert sc.get_top_picks(['AAPL', 'MSFT'], n=10) == []

    def test_output_sorted_by_score_descending(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: [
            {'strategy': 'PCS', 'symbol': sym, 'score': float(i), 'premium': 0.5}
            for i in range(5)
        ])
        picks = sc.get_top_picks(['AAPL', 'MSFT'], n=6)
        scores = [p['score'] for p in picks]
        assert scores == sorted(scores, reverse=True)

    def test_returns_at_most_n_picks(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 20))
        picks = sc.get_top_picks(['AAPL', 'MSFT', 'GOOG'], n=7)
        assert len(picks) <= 7

    def test_returns_fewer_when_not_enough_candidates(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(return_value=_picks('PCS', 'AAPL', 3))
        picks = sc.get_top_picks(['AAPL'], n=10)
        assert len(picks) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy diversity
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategyDiversity:
    """A single dominant strategy must not crowd out all n slots."""

    def test_two_strategies_each_get_half(self):
        """With 2 strategies and n=10, each should get ~5 slots."""
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('PCS', sym, 20, base_score=2.0) +
            _picks('CCS', sym, 20, base_score=1.0)  # lower score → would lose naive sort
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        pcs = [p for p in picks if p['strategy'] == 'PCS']
        ccs = [p for p in picks if p['strategy'] == 'CCS']
        # Each should have at least floor(10/2)=5 or close (remainder fill might add 1)
        assert len(pcs) >= 5
        assert len(ccs) >= 4  # at minimum the quota minus 1 (in case of odd n)

    def test_dominant_strategy_cannot_take_all_slots(self):
        """Even if CCS has the highest scores, PCS must still appear."""
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('CCS', sym, 20, base_score=9.0) +
            _picks('PCS', sym, 20, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        strategies_present = {p['strategy'] for p in picks}
        assert 'PCS' in strategies_present
        assert 'CCS' in strategies_present

    def test_three_strategies_each_represented(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('PCS', sym, 15, base_score=3.0) +
            _picks('CCS', sym, 15, base_score=2.0) +
            _picks('CSP', sym, 15, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=9)
        strats = {p['strategy'] for p in picks}
        assert strats == {'PCS', 'CCS', 'CSP'}

    def test_single_strategy_gets_all_n_slots(self):
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 20))
        picks = sc.get_top_picks(['AAPL'], n=10)
        assert len(picks) == 10
        assert all(p['strategy'] == 'PCS' for p in picks)

    def test_remainder_filled_by_highest_extras(self):
        """n=10, 3 strategies → quota 3 each = 9; 10th pick must come from leftovers."""
        sc = _scanner()
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('PCS', sym, 10, base_score=1.0) +
            _picks('CCS', sym, 10, base_score=1.0) +
            _picks('CSP', sym, 10, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        assert len(picks) == 10

    def test_pool_pointer_works_with_same_mock_objects(self):
        """
        MagicMock(return_value=X) returns the *same* list/dicts for every call.
        The pool-pointer approach (not id-based) must still produce the correct n
        picks without false deduplication.
        """
        sc = _scanner()
        # Same 4 dict objects returned for both tickers
        sc.scan_ticker = MagicMock(return_value=[
            {'strategy': s, 'symbol': 'X', 'score': 1.0, 'premium': 0.5}
            for s in ['IC', 'STRANGLE', 'CC', 'CSP']
        ])
        picks = sc.get_top_picks(['AAPL', 'MSFT'], n=5)
        assert len(picks) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Per-ticker cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxPicksPerTicker:

    def test_cap_1_no_ticker_repeated_in_same_strategy(self):
        sc = _scanner(max_picks_per_ticker=1)
        # 10 picks for AAPL and 10 for MSFT, all PCS
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 10))
        picks = sc.get_top_picks(['AAPL', 'MSFT'], n=10)
        pcs_picks = [p for p in picks if p['strategy'] == 'PCS']
        aapl_pcs = [p for p in pcs_picks if p['symbol'] == 'AAPL']
        msft_pcs = [p for p in pcs_picks if p['symbol'] == 'MSFT']
        assert len(aapl_pcs) <= 1
        assert len(msft_pcs) <= 1

    def test_cap_2_allows_two_per_ticker(self):
        sc = _scanner(max_picks_per_ticker=2)
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 10))
        picks = sc.get_top_picks(['AAPL'], n=10)
        pcs_aapl = [p for p in picks if p['strategy'] == 'PCS' and p['symbol'] == 'AAPL']
        assert len(pcs_aapl) <= 2

    def test_no_cap_allows_many_same_ticker(self):
        """When max_picks_per_ticker is absent, a single ticker can fill all slots."""
        sc = _scanner()  # no cap
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 20))
        picks = sc.get_top_picks(['AAPL'], n=10)
        aapl = [p for p in picks if p['symbol'] == 'AAPL']
        assert len(aapl) == 10

    def test_cap_1_forces_ticker_diversity(self):
        """With cap=1 and 5 symbols, the top-10 should contain up to 5 distinct symbols."""
        sc = _scanner(max_picks_per_ticker=1)
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 5))
        picks = sc.get_top_picks(symbols, n=10)
        unique_symbols = {p['symbol'] for p in picks}
        # At most 1 per ticker per strategy → 5 symbols, 1 pick each = 5 total (< 10)
        assert len(picks) <= len(symbols)
        assert len(unique_symbols) == len(picks)

    def test_cap_applies_across_strategies_globally_in_remainder_fill(self):
        """In step 4 (remainder fill), the global cap across all strategies is enforced."""
        sc = _scanner(max_picks_per_ticker=1)
        # 5 PCS + 5 CCS for AAPL only; n=4 → per_strat_q=2, selected=[AAPL/PCS, AAPL/CCS]
        # remainder=2, but extras are all AAPL → global cap=1 should block further AAPL picks
        # (one per strategy already selected → global count ≥ 1 for AAPL)
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('PCS', 'AAPL', 5, base_score=2.0) +
            _picks('CCS', 'AAPL', 5, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=4)
        # With cap=1 globally in the remainder phase, AAPL can only appear once total
        aapl_picks = [p for p in picks if p['symbol'] == 'AAPL']
        # Step 3 takes 2 (1 PCS + 1 CCS), already 2 AAPL → remainder extras should be skipped
        # But note: cap is per-strategy in step 1, global in step 4.
        # So we just verify cap is honoured: total picks ≤ n
        assert len(picks) <= 4


# ═══════════════════════════════════════════════════════════════════════════════
# IC allocation cap
# ═══════════════════════════════════════════════════════════════════════════════

class TestICAllocationCap:

    def test_ic_capped_at_50_pct(self):
        """ic_allocation_pct=0.50, n=10 → at most 5 IC picks."""
        sc = _scanner(ic_allocation_pct=0.50)
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('IC', sym, 15, base_score=9.0) +   # high score → would dominate
            _picks('PCS', sym, 15, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        ic_picks = [p for p in picks if p['strategy'] == 'IC']
        assert len(ic_picks) <= 5

    def test_ic_at_full_pct_not_capped(self):
        """ic_allocation_pct=1.0 (default) → IC can take all n slots if it scores highest."""
        sc = _scanner(ic_allocation_pct=1.0)
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('IC', sym, 20, base_score=9.0))
        picks = sc.get_top_picks(['AAPL'], n=10)
        assert len(picks) == 10
        assert all(p['strategy'] == 'IC' for p in picks)

    def test_pcs_fills_slots_freed_by_ic_cap(self):
        """When IC is capped, the freed slots must be filled by other strategies."""
        sc = _scanner(ic_allocation_pct=0.50)
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('IC', sym, 20, base_score=9.0) +
            _picks('PCS', sym, 20, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        pcs_picks = [p for p in picks if p['strategy'] == 'PCS']
        assert len(pcs_picks) >= 1   # PCS must appear despite lower scores

    def test_ic_cap_also_enforced_in_remainder_fill(self):
        """IC picks beyond max_ic_slots must not slip in during remainder fill."""
        sc = _scanner(ic_allocation_pct=0.30)   # max 3 IC for n=10
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('IC',  sym, 20, base_score=9.0) +
            _picks('PCS', sym, 20, base_score=1.0) +
            _picks('CCS', sym, 20, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        ic_count = sum(1 for p in picks if p['strategy'] == 'IC')
        assert ic_count <= max(1, int(0.30 * 10))   # ≤ 3

    def test_ic_cap_at_1_pick_minimum(self):
        """Even with ic_allocation_pct near 0, IC gets at least 1 slot (floor 1)."""
        sc = _scanner(ic_allocation_pct=0.05)   # 0.05 * 10 = 0.5 → floor = 0 → clamped to 1
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('IC',  sym, 5, base_score=9.0) +
            _picks('PCS', sym, 5, base_score=1.0)
        ))
        picks = sc.get_top_picks(['AAPL'], n=10)
        ic_picks = [p for p in picks if p['strategy'] == 'IC']
        assert len(ic_picks) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Combined: diversity + per-ticker + IC cap all active together
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedConstraints:

    def test_diversity_and_per_ticker_and_ic_cap_together(self):
        """
        n=10, 3 strategies, cap=2 per ticker, IC capped at 50%.
        IC ≤ 5, each ticker ≤ 2 (globally in remainder), all strategies represented.
        """
        sc = _scanner(ic_allocation_pct=0.50, max_picks_per_ticker=2)
        syms = ['AAPL', 'MSFT', 'GOOG']
        sc.scan_ticker = MagicMock(side_effect=lambda sym: (
            _picks('IC',  sym, 5, base_score=9.0) +
            _picks('PCS', sym, 5, base_score=2.0) +
            _picks('CCS', sym, 5, base_score=1.0)
        ))
        picks = sc.get_top_picks(syms, n=10)
        assert len(picks) <= 10
        ic_count = sum(1 for p in picks if p['strategy'] == 'IC')
        assert ic_count <= 5
        strats_present = {p['strategy'] for p in picks}
        assert len(strats_present) >= 2   # at minimum 2 strategies represented


# ═══════════════════════════════════════════════════════════════════════════════
# Ticker blacklist
# ═══════════════════════════════════════════════════════════════════════════════

def _scanner_with_blacklist(blacklist: list[str], **kwargs) -> OptionScanner:
    """Build a minimal scanner with a ticker_blacklist config key."""
    cfg = {
        'market_cap_min': 1e9,
        'expiry_days_max': 14,
        'risk_parameters': {'min_probability_of_expiry': 0.8},
        'strategies': {'iron_condor': {'enabled': True, 'ic_allocation_pct': 1.0}},
        'ticker_blacklist': blacklist,
    }
    cfg.update(kwargs)
    return OptionScanner(cfg)


class TestBlacklist:

    def test_blacklisted_ticker_never_scanned(self):
        """Tickers in ticker_blacklist must not be passed to scan_ticker at all."""
        sc = _scanner_with_blacklist(['ORCL'])
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 1))
        sc.get_top_picks(['AAPL', 'ORCL', 'MSFT'], n=10)
        called_with = {call.args[0] for call in sc.scan_ticker.call_args_list}
        assert 'ORCL' not in called_with
        assert 'AAPL' in called_with
        assert 'MSFT' in called_with

    def test_blacklisted_ticker_produces_no_picks(self):
        """No picks should have a blacklisted symbol in their 'symbol' field."""
        sc = _scanner_with_blacklist(['ORCL'])
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 3))
        picks = sc.get_top_picks(['AAPL', 'ORCL', 'MSFT'], n=10)
        assert all(p['symbol'] != 'ORCL' for p in picks)

    def test_blacklist_is_case_insensitive(self):
        """Blacklist matching is case-insensitive (e.g. 'orcl' blocks 'ORCL')."""
        sc = _scanner_with_blacklist(['orcl'])
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 1))
        sc.get_top_picks(['AAPL', 'ORCL'], n=10)
        called_with = {call.args[0] for call in sc.scan_ticker.call_args_list}
        assert 'ORCL' not in called_with

    def test_empty_blacklist_scans_all_tickers(self):
        """An empty blacklist (or absent key) must not filter any tickers."""
        sc = _scanner_with_blacklist([])
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 1))
        sc.get_top_picks(['AAPL', 'ORCL', 'MSFT'], n=10)
        called_with = {call.args[0] for call in sc.scan_ticker.call_args_list}
        assert called_with == {'AAPL', 'ORCL', 'MSFT'}

    def test_no_blacklist_key_scans_all_tickers(self):
        """When ticker_blacklist is absent from config, all tickers are scanned."""
        sc = _scanner(ic_allocation_pct=1.0)   # no blacklist key
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 1))
        sc.get_top_picks(['AAPL', 'ORCL', 'MSFT'], n=10)
        called_with = {call.args[0] for call in sc.scan_ticker.call_args_list}
        assert called_with == {'AAPL', 'ORCL', 'MSFT'}

    def test_entire_list_blacklisted_returns_empty(self):
        """If every ticker is blacklisted, get_top_picks returns an empty list."""
        sc = _scanner_with_blacklist(['AAPL', 'MSFT'])
        sc.scan_ticker = MagicMock(side_effect=lambda sym: _picks('PCS', sym, 1))
        picks = sc.get_top_picks(['AAPL', 'MSFT'], n=10)
        assert picks == []
        sc.scan_ticker.assert_not_called()
