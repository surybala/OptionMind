"""Tests for select_top_picks_with_scanner_controls — both selection modes.

equal_diversity mode: existing behavior, tested thoroughly in test_scanner_get_top_picks.py.
model_ranked mode:   new ML-aware path tested here.
"""
from __future__ import annotations

import pytest

from src.pick_selection import select_top_picks_with_scanner_controls


def _p(strategy: str, symbol: str, score: float) -> dict:
    return {"strategy": strategy, "symbol": symbol, "score": score}


def _cfg(
    mode: str = "model_ranked",
    ic_allocation_pct: float = 1.0,
    strategy_caps: dict | None = None,
    regime_allocation: dict | None = None,
) -> dict:
    cfg: dict = {
        "pick_selection": {"mode": mode},
        "strategies": {
            "iron_condor": {"enabled": True, "ic_allocation_pct": ic_allocation_pct}
        },
    }
    if strategy_caps is not None:
        cfg["pick_selection"]["strategy_caps"] = strategy_caps
    if regime_allocation is not None:
        cfg["pick_selection"]["regime_allocation"] = regime_allocation
    return cfg


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestModelRankedEdgeCases:
    def test_empty_candidates_returns_empty(self):
        assert select_top_picks_with_scanner_controls([], n=5, config=_cfg()) == []

    def test_n_zero_returns_empty(self):
        picks = [_p("PCS", "SPY", 1.0)]
        assert select_top_picks_with_scanner_controls(picks, n=0, config=_cfg()) == []

    def test_fewer_candidates_than_n(self):
        picks = [_p("PCS", "SPY", 1.0), _p("CCS", "AAPL", 0.5)]
        result = select_top_picks_with_scanner_controls(picks, n=10, config=_cfg())
        assert len(result) == 2

    def test_returns_at_most_n(self):
        picks = [_p("PCS", "SPY", float(i)) for i in range(20)]
        result = select_top_picks_with_scanner_controls(picks, n=7, config=_cfg())
        assert len(result) == 7


# ── Core model_ranked behavior ────────────────────────────────────────────────


class TestModelRankedTopN:
    def test_dominant_strategy_fills_all_slots(self):
        """Unlike equal_diversity, the best strategy can take all n slots."""
        picks = (
            [_p("PCS", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("CCS", "AAPL", 1.0 - i * 0.1) for i in range(10)]
        )
        result = select_top_picks_with_scanner_controls(picks, n=5, config=_cfg())
        # All 5 should be PCS (highest score) — CCS gets no guaranteed floor
        assert all(p["strategy"] == "PCS" for p in result)

    def test_output_sorted_by_score_descending(self):
        picks = [_p("PCS", "SPY", float(i)) for i in range(10)]
        result = select_top_picks_with_scanner_controls(picks, n=5, config=_cfg())
        scores = [p["score"] for p in result]
        assert scores == sorted(scores, reverse=True)

    def test_mixed_strategies_ranked_by_score(self):
        picks = [
            _p("CCS", "SPY", 5.0),
            _p("PCS", "AAPL", 4.0),
            _p("IC", "MSFT", 3.0),
            _p("CCS", "GOOG", 2.0),
            _p("PCS", "META", 1.0),
        ]
        result = select_top_picks_with_scanner_controls(picks, n=3, config=_cfg())
        assert [p["score"] for p in result] == [5.0, 4.0, 3.0]
        assert [p["strategy"] for p in result] == ["CCS", "PCS", "IC"]

    def test_uses_model_score_key_when_score_absent(self):
        picks = [
            {"strategy": "PCS", "symbol": "SPY", "model_score": 0.9},
            {"strategy": "CCS", "symbol": "AAPL", "model_score": 0.5},
        ]
        result = select_top_picks_with_scanner_controls(picks, n=2, config=_cfg())
        assert result[0]["model_score"] == 0.9


# ── IC cap in model_ranked ────────────────────────────────────────────────────


class TestModelRankedICCap:
    def test_ic_cap_enforced(self):
        """IC picks beyond floor(ic_pct * n) must be skipped even at high score."""
        picks = (
            [_p("IC", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", "AAPL", 1.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(ic_allocation_pct=0.30)  # max 3 IC for n=10
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        ic_count = sum(1 for p in result if p["strategy"] == "IC")
        assert ic_count <= max(1, int(0.30 * 10))

    def test_ic_uncapped_at_full_pct(self):
        """ic_allocation_pct=1.0 → IC can fill all n slots."""
        picks = [_p("IC", "SPY", float(i)) for i in range(15)]
        result = select_top_picks_with_scanner_controls(picks, n=10, config=_cfg(ic_allocation_pct=1.0))
        assert len(result) == 10
        assert all(p["strategy"] == "IC" for p in result)

    def test_pcs_fills_slots_freed_by_ic_cap(self):
        """Slots freed by IC cap must go to next-best picks regardless of strategy."""
        picks = (
            [_p("IC", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", "AAPL", 2.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(ic_allocation_pct=0.30)
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        pcs_count = sum(1 for p in result if p["strategy"] == "PCS")
        assert pcs_count >= 1


# ── Same-ticker ranking behavior in model_ranked ─────────────────────────────


class TestModelRankedTickerConcentration:
    def test_same_ticker_can_fill_all_slots_when_scores_win(self):
        picks = [_p("PCS", "SPY", float(i)) for i in range(15)]
        result = select_top_picks_with_scanner_controls(picks, n=10, config=_cfg())
        assert len(result) == 10
        assert all(p["symbol"] == "SPY" for p in result)

    def test_second_symbol_does_not_force_diversity_when_scores_are_lower(self):
        picks = (
            [_p("PCS", "SPY", 9.0 - i * 0.1) for i in range(5)]
            + [_p("PCS", "AAPL", 5.0 - i * 0.1) for i in range(5)]
        )
        result = select_top_picks_with_scanner_controls(picks, n=5, config=_cfg())
        assert [pick["symbol"] for pick in result] == ["SPY"] * 5


# ── strategy_caps in model_ranked ─────────────────────────────────────────────


class TestModelRankedStrategyCaps:
    def test_pcs_capped_at_fraction(self):
        """strategy_caps.PCS: 0.4 → PCS may take at most 4 of 10 slots."""
        picks = (
            [_p("PCS", f"T{i}", 9.0 - i * 0.1) for i in range(10)]
            + [_p("CCS", f"U{i}", 1.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(strategy_caps={"PCS": 0.4})
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        pcs_count = sum(1 for p in result if p["strategy"] == "PCS")
        assert pcs_count <= max(1, int(0.4 * 10))

    def test_capped_strategy_does_not_crowd_out_others(self):
        picks = (
            [_p("PCS", f"T{i}", 9.0 - i * 0.1) for i in range(10)]
            + [_p("CCS", f"U{i}", 1.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(strategy_caps={"PCS": 0.5})
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        ccs_count = sum(1 for p in result if p["strategy"] == "CCS")
        assert ccs_count >= 5  # freed slots taken by CCS

    def test_ic_strategy_cap_overrides_ic_allocation_pct(self):
        """strategy_caps.IC and ic_allocation_pct should both cap IC independently."""
        picks = (
            [_p("IC", f"T{i}", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", f"U{i}", 1.0 - i * 0.1) for i in range(10)]
        )
        # ic_allocation_pct=0.50 → 5 IC; strategy_caps.IC=0.20 → 2 IC; min wins
        cfg = _cfg(ic_allocation_pct=0.50, strategy_caps={"IC": 0.20})
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        ic_count = sum(1 for p in result if p["strategy"] == "IC")
        # ic_pct path allows 5, but strategy_cap allows only 2
        assert ic_count <= 2

    def test_unknown_strategy_not_affected_by_other_caps(self):
        picks = [_p("STRANGLE", f"T{i}", float(i)) for i in range(10)]
        cfg = _cfg(strategy_caps={"PCS": 0.3, "CCS": 0.3})
        result = select_top_picks_with_scanner_controls(picks, n=5, config=cfg)
        assert len(result) == 5
        assert all(p["strategy"] == "STRANGLE" for p in result)


class TestModelRankedRegimeAllocation:
    def test_orange_regime_allocation_does_not_override_ranked_scores(self):
        picks = (
            [_p("PCS", f"P{i}", 10.0 - i * 0.1) for i in range(10)]
            + [_p("CCS", f"C{i}", 4.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(
            regime_allocation={
                "enabled": True,
                "regimes": {
                    "ORANGE": {
                        "PCS": {"max_fraction": 0.4},
                        "CCS": {"min_fraction": 0.4},
                    }
                },
            }
        )

        result = select_top_picks_with_scanner_controls(
            picks,
            n=10,
            config=cfg,
            regime_label="ORANGE",
        )

        assert all(p["strategy"] == "PCS" for p in result)

    def test_green_regime_allocation_does_not_zero_out_higher_scored_ccs(self):
        picks = (
            [_p("CCS", f"C{i}", 10.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", f"P{i}", 4.0 - i * 0.1) for i in range(10)]
        )
        cfg = _cfg(
            regime_allocation={
                "enabled": True,
                "regimes": {
                    "GREEN": {
                        "CCS": {"max_fraction": 0.0},
                        "PCS": {"min_fraction": 0.4},
                    }
                },
            }
        )

        result = select_top_picks_with_scanner_controls(
            picks,
            n=5,
            config=cfg,
            regime_label="GREEN",
        )

        assert all(p["strategy"] == "CCS" for p in result)


# ── Mode fallback behavior ────────────────────────────────────────────────────


class TestModeDefaults:
    def test_no_config_uses_equal_diversity(self):
        """Without config the old equal_diversity path runs unchanged."""
        picks = (
            [_p("CCS", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", "AAPL", 1.0 - i * 0.1) for i in range(10)]
        )
        result = select_top_picks_with_scanner_controls(picks, n=10, config=None)
        pcs_count = sum(1 for p in result if p["strategy"] == "PCS")
        # equal_diversity guarantees PCS a floor slot
        assert pcs_count >= 1

    def test_equal_diversity_mode_explicit(self):
        picks = (
            [_p("CCS", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", "AAPL", 1.0 - i * 0.1) for i in range(10)]
        )
        cfg = {"pick_selection": {"mode": "equal_diversity"}}
        result = select_top_picks_with_scanner_controls(picks, n=10, config=cfg)
        pcs_count = sum(1 for p in result if p["strategy"] == "PCS")
        assert pcs_count >= 1

    def test_model_ranked_does_not_guarantee_floor(self):
        """model_ranked gives no floor to lower-scored strategies."""
        picks = (
            [_p("CCS", "SPY", 9.0 - i * 0.1) for i in range(10)]
            + [_p("PCS", "AAPL", 1.0 - i * 0.1) for i in range(10)]
        )
        result = select_top_picks_with_scanner_controls(picks, n=5, config=_cfg())
        pcs_count = sum(1 for p in result if p["strategy"] == "PCS")
        # CCS scores dominate → PCS may get 0 slots
        assert pcs_count == 0
