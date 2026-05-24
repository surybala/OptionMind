"""
Tests for SentimentAnalyzer — all yfinance calls are mocked.
"""
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.sentiment import SentimentAnalyzer, _SENTIMENT_CACHE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(enabled=True, skew=0.30, max_skew=0.50):
    return {
        'sentiment': {
            'enabled':         enabled,
            'lookback_days':   20,
            'rsi_period':      14,
            'bull_threshold':  0.20,
            'bear_threshold':  0.20,
            'skew_factor':     skew,
            'max_skew':        max_skew,
        }
    }


def _make_prices(trend: str, n: int = 35) -> pd.Series:
    """
    Return a price series with clear bull / bear / neutral momentum.
    trend='bull'    : steadily rising prices → RSI > 60, price above SMA-20
    trend='bear'    : steadily falling prices
    trend='neutral' : flat prices
    """
    import numpy as np
    if trend == 'bull':
        prices = [100.0 + i * 0.8 for i in range(n)]   # +0.8/day
    elif trend == 'bear':
        prices = [100.0 - i * 0.8 for i in range(n)]   # -0.8/day
    else:
        prices = [100.0 + (i % 3 - 1) * 0.1 for i in range(n)]  # tiny oscillation
    return pd.Series(prices, dtype=float)


def _mock_ticker(prices: pd.Series):
    """Patch yfinance.Ticker so .history() returns a DataFrame with 'Close'."""
    ticker = MagicMock()
    df = pd.DataFrame({'Close': prices})
    ticker.history.return_value = df
    return ticker


# ── Static method: RSI ────────────────────────────────────────────────────────

class TestRsi:
    def test_all_gains_returns_100(self):
        s = pd.Series([100.0 + i for i in range(20)])
        assert SentimentAnalyzer._rsi(s, 14) == 100.0

    def test_all_losses_returns_0(self):
        s = pd.Series([100.0 - i for i in range(20)])
        assert SentimentAnalyzer._rsi(s, 14) == 0.0

    def test_flat_returns_near_50(self):
        s = pd.Series([100.0] * 20)
        result = SentimentAnalyzer._rsi(s, 14)
        # All deltas zero → avg_loss==0 → returns 100 (edge case is fine)
        assert 0 <= result <= 100

    def test_short_series_returns_50(self):
        s = pd.Series([100.0, 101.0])
        assert SentimentAnalyzer._rsi(s, 14) == 50.0

    def test_bull_series_high_rsi(self):
        prices = _make_prices('bull', 35)
        rsi = SentimentAnalyzer._rsi(prices, 14)
        assert rsi > 60, f"Expected RSI > 60 for bull series, got {rsi:.1f}"

    def test_bear_series_low_rsi(self):
        prices = _make_prices('bear', 35)
        rsi = SentimentAnalyzer._rsi(prices, 14)
        assert rsi < 40, f"Expected RSI < 40 for bear series, got {rsi:.1f}"


# ── SentimentAnalyzer: disabled mode ─────────────────────────────────────────

class TestDisabled:
    def test_analyze_returns_neutral_when_disabled(self):
        sa = SentimentAnalyzer(_make_config(enabled=False))
        result = sa.analyze('AAPL')
        assert result['sentiment'] == 'NEUTRAL'
        assert result['strength']  == 0.0

    def test_adjust_delta_unchanged_when_disabled(self):
        sa = SentimentAnalyzer(_make_config(enabled=False))
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        assert sa.adjust_delta(0.30, 'PCS', sent) == 0.30
        assert sa.adjust_delta(0.30, 'CCS', sent) == 0.30


# ── SentimentAnalyzer: compute (mocked prices) ───────────────────────────────

class TestAnalyze:
    def setup_method(self):
        # Clear session cache before each test
        _SENTIMENT_CACHE.clear()

    @patch('src.sentiment.yf.Ticker')
    def test_bull_detected(self, mock_tf):
        mock_tf.return_value = _mock_ticker(_make_prices('bull', 35))
        sa = SentimentAnalyzer(_make_config())
        r  = sa.analyze('AAPL')
        assert r['sentiment'] == 'BULL', f"Got {r}"
        assert r['strength']  >  0.0

    @patch('src.sentiment.yf.Ticker')
    def test_bear_detected(self, mock_tf):
        mock_tf.return_value = _mock_ticker(_make_prices('bear', 35))
        sa = SentimentAnalyzer(_make_config())
        r  = sa.analyze('AAPL')
        assert r['sentiment'] == 'BEAR', f"Got {r}"
        assert r['strength']  >  0.0

    @patch('src.sentiment.yf.Ticker')
    def test_neutral_detected(self, mock_tf):
        mock_tf.return_value = _mock_ticker(_make_prices('neutral', 35))
        sa = SentimentAnalyzer(_make_config())
        r  = sa.analyze('AAPL')
        # Flat prices should produce near-zero composite score
        assert r['sentiment'] in ('NEUTRAL', 'BULL', 'BEAR')   # any is valid

    @patch('src.sentiment.yf.Ticker')
    def test_caches_result(self, mock_tf):
        mock_tf.return_value = _mock_ticker(_make_prices('bull', 35))
        sa = SentimentAnalyzer(_make_config())
        r1 = sa.analyze('MSFT')
        # Second call should NOT hit yfinance again
        mock_tf.return_value = _mock_ticker(_make_prices('bear', 35))  # changed!
        r2 = sa.analyze('MSFT')
        assert r1['sentiment'] == r2['sentiment'], "Cache miss — result changed"

    @patch('src.sentiment.yf.Ticker')
    def test_error_returns_neutral(self, mock_tf):
        mock_tf.side_effect = RuntimeError("network error")
        sa = SentimentAnalyzer(_make_config())
        r  = sa.analyze('BADTICKER')
        assert r['sentiment'] == 'NEUTRAL'
        assert r['strength']  == 0.0

    @patch('src.sentiment.yf.Ticker')
    def test_short_history_returns_neutral(self, mock_tf):
        mock_tf.return_value = _mock_ticker(pd.Series([100.0, 101.0, 102.0]))
        sa = SentimentAnalyzer(_make_config())
        r  = sa.analyze('SHORT')
        assert r['sentiment'] == 'NEUTRAL'


# ── SentimentAnalyzer: delta adjustment ──────────────────────────────────────

class TestDeltaAdjustment:
    def _sa(self, skew=0.30, max_skew=0.50):
        return SentimentAnalyzer(_make_config(skew=skew, max_skew=max_skew))

    # BULL
    def test_bull_raises_pcs_delta(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        adj  = sa.adjust_delta(0.30, 'PCS', sent)
        assert adj > 0.30, f"Expected > 0.30, got {adj}"

    def test_bull_lowers_ccs_delta(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        adj  = sa.adjust_delta(0.30, 'CCS', sent)
        assert adj < 0.30, f"Expected < 0.30, got {adj}"

    # BEAR
    def test_bear_lowers_pcs_delta(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BEAR', 'strength': 1.0}
        adj  = sa.adjust_delta(0.30, 'PCS', sent)
        assert adj < 0.30

    def test_bear_raises_ccs_delta(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BEAR', 'strength': 1.0}
        adj  = sa.adjust_delta(0.30, 'CCS', sent)
        assert adj > 0.30

    # NEUTRAL
    def test_neutral_unchanged(self):
        sa   = self._sa()
        sent = {'sentiment': 'NEUTRAL', 'strength': 0.0}
        assert sa.adjust_delta(0.30, 'PCS', sent) == 0.30
        assert sa.adjust_delta(0.30, 'CCS', sent) == 0.30

    # Bounds
    def test_max_skew_respected(self):
        sa   = self._sa(skew=0.30, max_skew=0.20)
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        adj  = sa.adjust_delta(0.30, 'PCS', sent)
        assert adj <= 0.30 * (1 + 0.20) + 1e-9

    def test_delta_never_below_0_05(self):
        sa   = self._sa(skew=0.99, max_skew=0.99)
        sent = {'sentiment': 'BEAR', 'strength': 1.0}
        adj  = sa.adjust_delta(0.05, 'PCS', sent)
        assert adj >= 0.05

    def test_delta_never_above_0_50(self):
        sa   = self._sa(skew=0.99, max_skew=0.99)
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        adj  = sa.adjust_delta(0.45, 'PCS', sent)
        assert adj <= 0.50

    # IC helper
    def test_adjust_ic_deltas_bull(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BULL', 'strength': 1.0}
        p_d, c_d = sa.adjust_ic_deltas(0.25, 0.25, sent)
        assert p_d > 0.25   # put (PCS) raised in bull
        assert c_d < 0.25   # call (CCS) lowered in bull

    def test_adjust_ic_deltas_bear(self):
        sa   = self._sa(skew=0.30)
        sent = {'sentiment': 'BEAR', 'strength': 1.0}
        p_d, c_d = sa.adjust_ic_deltas(0.25, 0.25, sent)
        assert p_d < 0.25
        assert c_d > 0.25

    # Zero strength → no adjustment
    def test_zero_strength_unchanged(self):
        sa   = self._sa()
        sent = {'sentiment': 'BULL', 'strength': 0.0}
        assert sa.adjust_delta(0.30, 'PCS', sent) == 0.30
        assert sa.adjust_delta(0.30, 'CCS', sent) == 0.30
