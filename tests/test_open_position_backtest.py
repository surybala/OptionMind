from __future__ import annotations

import pandas as pd

from ml.models.compare_ml_vs_deterministic_open_positions import _build_snapshot_map, _simulate_open_book


def test_simulate_open_book_carries_capital_until_position_exits():
    frame = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T14:30:00Z",
                "exit_timestamp": "2025-01-03T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 100.0,
                "entry_credit": 1.0,
                "expiration": "2025-01-10",
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 100.0,
                "candidate_score": 10.0,
                "allocator_regime_label": "GREEN",
            },
            {
                "entry_timestamp": "2025-01-03T14:30:00Z",
                "exit_timestamp": "2025-01-06T19:00:00Z",
                "dte": 7,
                "underlying": "BBB",
                "strategy": "PCS",
                "underlying_close": 101.0,
                "entry_credit": 1.0,
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 150.0,
                "candidate_score": 9.0,
                "allocator_regime_label": "GREEN",
            },
            {
                "entry_timestamp": "2025-01-06T14:30:00Z",
                "exit_timestamp": "2025-01-07T19:00:00Z",
                "dte": 7,
                "underlying": "CCC",
                "strategy": "PCS",
                "underlying_close": 102.0,
                "entry_credit": 1.0,
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 200.0,
                "candidate_score": 8.0,
                "allocator_regime_label": "GREEN",
            },
        ]
    )
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    snapshots = _build_snapshot_map(frame)
    config = {
        "max_capital_per_period": 500.0,
        "max_contracts_per_pick": 1,
        "risk_parameters": {
            "directional_exposure_caps": {"enabled": False},
        },
    }

    result = _simulate_open_book(
        frame,
        config=config,
        snapshots=snapshots,
        side_name="test",
        candidate_mask=pd.Series(True, index=frame.index),
        top_n=1,
        score_column="candidate_score",
        primary_gate_label="post_gate",
    )

    selected = result["selected"]
    assert list(selected["symbol"]) == ["AAA", "CCC"]
    assert result["gate_stage_counts"]["capital_budget"] == 1


def test_simulate_open_book_allows_distinct_same_symbol_ladders():
    frame = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T14:30:00Z",
                "exit_timestamp": "2025-01-10T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 100.0,
                "entry_credit": 1.0,
                "expiration": "2025-01-10",
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 100.0,
                "candidate_score": 10.0,
                "allocator_regime_label": "GREEN",
            },
            {
                "entry_timestamp": "2025-01-03T14:30:00Z",
                "exit_timestamp": "2025-01-10T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 101.0,
                "entry_credit": 1.0,
                "short_strike": 99.0,
                "long_strike": 94.0,
                "implied_volatility": 0.25,
                "expected_pnl": 90.0,
                "candidate_score": 9.0,
                "allocator_regime_label": "GREEN",
            },
        ]
    )
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    snapshots = _build_snapshot_map(frame)
    config = {
        "max_capital_per_period": 5_000.0,
        "max_contracts_per_pick": 1,
        "risk_parameters": {
            "directional_exposure_caps": {"enabled": False},
        },
    }

    result = _simulate_open_book(
        frame,
        config=config,
        snapshots=snapshots,
        side_name="test",
        candidate_mask=pd.Series(True, index=frame.index),
        top_n=1,
        score_column="candidate_score",
        primary_gate_label="post_gate",
    )

    selected = result["selected"]
    assert list(selected["symbol"]) == ["AAA", "AAA"]
    assert result["gate_stage_counts"].get("dedup_open_position", 0) == 0


def test_simulate_open_book_blocks_exact_duplicate_contracts():
    frame = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T14:30:00Z",
                "exit_timestamp": "2025-01-10T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 100.0,
                "entry_credit": 1.0,
                "expiration": "2025-01-10",
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 100.0,
                "candidate_score": 10.0,
                "allocator_regime_label": "GREEN",
            },
            {
                "entry_timestamp": "2025-01-03T14:30:00Z",
                "exit_timestamp": "2025-01-10T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 101.0,
                "entry_credit": 1.0,
                "expiration": "2025-01-10",
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 90.0,
                "candidate_score": 9.0,
                "allocator_regime_label": "GREEN",
            },
        ]
    )
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    snapshots = _build_snapshot_map(frame)
    config = {
        "max_capital_per_period": 5_000.0,
        "max_contracts_per_pick": 1,
        "risk_parameters": {
            "directional_exposure_caps": {"enabled": False},
        },
    }

    result = _simulate_open_book(
        frame,
        config=config,
        snapshots=snapshots,
        side_name="test",
        candidate_mask=pd.Series(True, index=frame.index),
        top_n=1,
        score_column="candidate_score",
        primary_gate_label="post_gate",
    )

    selected = result["selected"]
    assert list(selected["symbol"]) == ["AAA"]
    assert result["gate_stage_counts"]["dedup_open_position"] == 1


def test_simulate_open_book_uses_ml_rank_tier_position_sizing():
    frame = pd.DataFrame(
        [
            {
                "entry_timestamp": "2025-01-02T14:30:00Z",
                "exit_timestamp": "2025-01-03T19:00:00Z",
                "dte": 7,
                "underlying": "AAA",
                "strategy": "PCS",
                "underlying_close": 100.0,
                "entry_credit": 1.0,
                "expiration": "2025-01-10",
                "short_strike": 100.0,
                "long_strike": 95.0,
                "implied_volatility": 0.25,
                "expected_pnl": 100.0,
                "candidate_score": 10.0,
                "allocator_regime_label": "GREEN",
            },
        ]
    )
    frame["entry_timestamp"] = pd.to_datetime(frame["entry_timestamp"], utc=True)
    snapshots = _build_snapshot_map(frame)
    config = {
        "max_capital_per_period": 5_000.0,
        "max_contracts_per_pick": 4,
        "risk_parameters": {
            "ml_position_sizing": {
                "enabled": True,
                "rank_tiers": [
                    {"max_rank": 1, "quantity": 4},
                    {"max_rank": 3, "quantity": 3},
                    {"max_rank": 6, "quantity": 2},
                ],
            },
            "directional_exposure_caps": {"enabled": False},
        },
    }

    result = _simulate_open_book(
        frame,
        config=config,
        snapshots=snapshots,
        side_name="test",
        candidate_mask=pd.Series(True, index=frame.index),
        top_n=1,
        score_column="candidate_score",
        primary_gate_label="post_gate",
    )

    selected = result["selected"]
    assert int(selected.iloc[0]["quantity"]) == 4
