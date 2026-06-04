import pandas as pd
import pytest

from ml.datasets.balance_intraday_risk_dataset import (
    IntradayRiskBalanceConfig,
    balance_intraday_risk_frame,
)


def test_balance_intraday_risk_frame_reduces_dominant_underlying_share():
    rows = []
    for i in range(100):
        rows.append({
            "entry_timestamp": f"2026-01-{(i % 28) + 1:02d}T14:30:00Z",
            "state_timestamp": f"2026-01-{(i % 28) + 1:02d}T14:35:00Z",
            "underlying": "SMH",
            "market_volatility_regime": "high",
            "market_trend_regime": "uptrend",
            "intraday_exit_reason": "stop_loss",
        })
    for symbol, exit_reason in (("SPY", "profit_take"), ("QQQ", "expired"), ("DIA", "large_loss")):
        for i in range(20):
            rows.append({
                "entry_timestamp": f"2026-02-{(i % 28) + 1:02d}T14:30:00Z",
                "state_timestamp": f"2026-02-{(i % 28) + 1:02d}T14:35:00Z",
                "underlying": symbol,
                "market_volatility_regime": "normal",
                "market_trend_regime": "sideways",
                "intraday_exit_reason": exit_reason,
            })

    balanced = balance_intraday_risk_frame(
        pd.DataFrame(rows),
        IntradayRiskBalanceConfig(target_rows=80, max_underlying_share=0.5, max_oversample_factor=1.0, random_seed=1),
    )

    assert len(balanced) == 80
    assert balanced["underlying"].value_counts(normalize=True).max() <= 0.5
    assert "entry_month" not in balanced.columns


def test_balance_intraday_risk_frame_rejects_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        balance_intraday_risk_frame(pd.DataFrame(), IntradayRiskBalanceConfig())
