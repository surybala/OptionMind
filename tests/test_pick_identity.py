from src.pick_identity import (
    pick_contract_signature,
    position_contract_signature,
    symbol_strategy_key,
)


def test_symbol_strategy_key_normalizes_picks_and_positions():
    assert symbol_strategy_key({"symbol": "spy", "strategy": "pcs"}) == ("SPY", "PCS")
    assert symbol_strategy_key({"symbol": "spy", "type": "pcs"}) == ("SPY", "PCS")


def test_pick_and_position_signatures_match_for_same_contract():
    pick = {
        "symbol": "SPY",
        "strategy": "PCS",
        "expiry": "2025-01-17",
        "short_strike": 600.0,
        "long_strike": 595.0,
    }
    position = {
        "symbol": "SPY",
        "type": "PCS",
        "expiry": "2025-01-17",
        "legs": {"short_strike": 600.0, "long_strike": 595.0},
    }

    assert pick_contract_signature(pick) == position_contract_signature(position)


def test_contract_signature_changes_for_distinct_ladder():
    base = {
        "symbol": "SPY",
        "strategy": "PCS",
        "expiry": "2025-01-17",
        "short_strike": 600.0,
        "long_strike": 595.0,
    }
    later_expiry = dict(base, expiry="2025-01-24")
    wider_ladder = dict(base, long_strike=590.0)

    assert pick_contract_signature(base) != pick_contract_signature(later_expiry)
    assert pick_contract_signature(base) != pick_contract_signature(wider_ladder)
