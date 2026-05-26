import pandas as pd
import pytest

from ml.datasets.balance_candidate_dataset import BalanceConfig, _allocate_quotas, balance_candidate_frame


def test_allocate_quotas_respects_capacity_and_total():
    counts = pd.Series({"SMH": 1000, "SPY": 100, "IEF": 10})

    quotas = _allocate_quotas(
        counts,
        300,
        caps={"SMH": 120, "SPY": 120, "IEF": 120},
        max_oversample_factor=3,
    )

    assert sum(quotas.values()) == 270
    assert quotas["SMH"] <= 120
    assert quotas["SPY"] <= 120
    assert quotas["IEF"] <= 30


def test_balance_candidate_frame_reduces_dominant_underlying_share():
    rows = []
    for i in range(100):
        rows.append({
            "entry_timestamp": f"2026-01-{(i % 28) + 1:02d}",
            "underlying": "SMH",
            "market_volatility_regime": "low",
            "market_trend_regime": "uptrend",
            "expected_pnl": i,
        })
    for symbol in ("SPY", "QQQ", "DIA"):
        for i in range(10):
            rows.append({
                "entry_timestamp": f"2026-02-{(i % 28) + 1:02d}",
                "underlying": symbol,
                "market_volatility_regime": "normal",
                "market_trend_regime": "sideways",
                "expected_pnl": i,
            })

    balanced = balance_candidate_frame(
        pd.DataFrame(rows),
        BalanceConfig(target_rows=80, max_underlying_share=0.5, max_oversample_factor=3, random_seed=1),
    )

    assert len(balanced) == 80
    assert balanced["underlying"].value_counts(normalize=True).max() <= 0.5


def test_balance_candidate_frame_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        balance_candidate_frame(pd.DataFrame(), BalanceConfig())


def test_balance_config_default_oversample_factor_is_one():
    assert BalanceConfig().max_oversample_factor == 1.0


def test_balance_no_oversample_caps_sparse_underlying_at_actual_count():
    """max_oversample_factor=1.0 — a 5-row underlying stays at 5 regardless of target."""
    rows = []
    for i in range(5):
        rows.append({"entry_timestamp": f"2026-01-0{i + 1}", "underlying": "QQQ", "expected_pnl": float(i)})
    for i in range(45):
        rows.append({
            "entry_timestamp": f"2026-02-{(i % 28) + 1:02d}",
            "underlying": "SPY",
            "expected_pnl": float(i + 10),
        })
    balanced = balance_candidate_frame(
        pd.DataFrame(rows),
        BalanceConfig(target_rows=50, max_oversample_factor=1.0, random_seed=1),
    )
    qqq_rows = (balanced["underlying"] == "QQQ").sum()
    assert qqq_rows <= 5
