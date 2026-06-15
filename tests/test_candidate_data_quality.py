import pandas as pd

from ml.datasets.candidate_data_quality import (
    CandidateQualityFilterConfig,
    apply_candidate_quality_filters,
)


def test_apply_candidate_quality_filters_removes_near_width_spreads_and_tiny_max_loss():
    df = pd.DataFrame(
        [
            {
                "entry_timestamp": "2026-01-01T14:30:00+00:00",
                "credit_to_width": 0.50,
                "max_loss": 200.0,
                "option_entry_volume": 20,
                "long_option_entry_volume": 20,
                "option_entry_trade_count": 10,
                "long_option_entry_trade_count": 10,
            },
            {
                "entry_timestamp": "2026-01-02T14:30:00+00:00",
                "credit_to_width": 0.99,
                "max_loss": 2.0,
                "option_entry_volume": 20,
                "long_option_entry_volume": 20,
                "option_entry_trade_count": 10,
                "long_option_entry_trade_count": 10,
            },
            {
                "entry_timestamp": "2026-01-03T14:30:00+00:00",
                "credit_to_width": 0.70,
                "max_loss": 200.0,
                "option_entry_volume": 1,
                "long_option_entry_volume": 2,
                "option_entry_trade_count": 1,
                "long_option_entry_trade_count": 1,
            },
        ]
    )

    filtered, stats = apply_candidate_quality_filters(
        df,
        CandidateQualityFilterConfig(
            min_max_loss_dollars=25.0,
            max_credit_to_width=0.90,
            min_short_leg_volume=5.0,
            min_long_leg_volume=5.0,
            min_short_leg_trade_count=2,
            min_long_leg_trade_count=2,
        ),
    )

    assert len(filtered) == 1
    assert stats["input_rows"] == 3
    assert stats["output_rows"] == 1
    assert stats["drop_reasons"]["min_max_loss_dollars"] == 1
    assert stats["drop_reasons"]["min_short_leg_volume"] == 1
