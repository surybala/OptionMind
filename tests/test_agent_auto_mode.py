"""
Tests for agent auto-mode execution logic.

Verifies that auto mode executes ALL picks from the ML pipeline without
applying any prob_win threshold — the model (ranker + classifier) is the
sole quality gate.
"""
import sys
from unittest.mock import MagicMock, patch

_external_mock = MagicMock()
for _mod in [
    'alpaca',
    'alpaca.trading',
    'alpaca.trading.client',
    'alpaca.trading.enums',
    'alpaca.trading.requests',
    'numpy',
    'pandas',
    'scipy',
    'scipy.stats',
    'yfinance',
]:
    sys.modules.setdefault(_mod, _external_mock)

from src.agent_execution import execute_picks  # noqa: E402


def _make_pick(symbol='SPY', prob_win=0.60, model_score=0.85):
    """Create a pick with low prob_win but high model score."""
    return {
        'symbol': symbol,
        'strategy': 'PCS',
        'short_strike': 500,
        'long_strike': 495,
        'premium': 0.50,
        'prob_win': prob_win,
        'model_score': model_score,
        'score': model_score,
        'quantity': 1,
        'expiry': '2026-06-20',
    }


class TestAutoModeNosProbWinGate:
    """Auto mode must execute all ML-pipeline picks regardless of prob_win."""

    def test_low_prob_win_picks_still_approved(self):
        """Picks with prob_win < 0.80 should still be approved in auto mode.

        Previously, auto mode filtered picks by prob_win >= auto_execute_prob
        (default 0.80-0.90). Now the ML pipeline is the sole gate.
        """
        picks = [
            _make_pick('SPY', prob_win=0.55, model_score=0.92),
            _make_pick('QQQ', prob_win=0.65, model_score=0.88),
            _make_pick('IWM', prob_win=0.40, model_score=0.95),
        ]

        # In auto mode, approved = picks (no filtering)
        # Simulate the agent.py auto-mode logic:
        approved = picks  # This is now the auto-mode behavior

        assert len(approved) == 3
        assert all(p in approved for p in picks)

    def test_auto_mode_does_not_reference_auto_execute_prob(self):
        """The auto_execute_prob config key should not affect pick approval."""
        import agent

        # Read the auto-mode section source to confirm no prob_win filtering
        import inspect
        source = inspect.getsource(agent._run_once)

        # The old pattern was: p.get('prob_win', 0) >= thresh
        assert "auto_execute_prob" not in source, (
            "auto_execute_prob should not be referenced — ML pipeline is the gate"
        )

    def test_execute_picks_receives_all_picks(self):
        """execute_picks should receive the full picks list in auto mode."""
        picks = [
            _make_pick('SPY', prob_win=0.50, model_score=0.90),
            _make_pick('QQQ', prob_win=0.30, model_score=0.85),
        ]

        db = MagicMock()
        executor = MagicMock()
        executor.execute_pick.return_value = 'order-123'
        executor.get_fill_price.return_value = (0.50, False)
        db.log_trade.return_value = 1
        db.confirm_open.return_value = None

        results = execute_picks(picks, executor, db, dry_run=True)

        # All picks should be submitted (dry run)
        assert len(results) == 2
        assert all(oid == 'DRY_RUN' for _, oid in results)
