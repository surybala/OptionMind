"""
Tests for:
  - src/risk_rules/gamma_risk.py  (eval_gamma_trigger, GammaRiskRule)
  - src/risk_rules/pipeline.py    (RulePipeline)
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk_rules.gamma_risk import eval_gamma_trigger, GammaRiskRule
from src.risk_rules.pipeline import RulePipeline
from src.risk_rules.base import CloseSignal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk(gamma_theta_ratio=2.0, net_short_delta=0.20, risk_score=0.75):
    return {
        'gamma_theta_ratio': gamma_theta_ratio,
        'net_short_delta': net_short_delta,
        'risk_score': risk_score,
    }


def _gamma_rule(enabled=True, ratio_threshold=1.5, min_delta=0.15,
                min_profit_pct=0.25, urgent_delta=0.30):
    return GammaRiskRule(
        enabled=enabled,
        ratio_threshold=ratio_threshold,
        min_delta=min_delta,
        min_profit_pct=min_profit_pct,
        urgent_delta=urgent_delta,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# eval_gamma_trigger
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvalGammaTrigger(unittest.TestCase):

    def _call(self, risk=None, dte=10, entry_premium=1.00, pnl_per_share=0.30,
              ratio_threshold=1.5, min_delta=0.15, min_profit_pct=0.25,
              urgent_delta=0.30):
        risk = risk or _risk()
        return eval_gamma_trigger(
            risk, dte, entry_premium, pnl_per_share,
            ratio_threshold, min_delta, min_profit_pct, urgent_delta,
        )

    def test_fires_when_all_conditions_met(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.20)
        result = self._call(risk=risk, pnl_per_share=0.30, entry_premium=1.00)
        self.assertIsNotNone(result)
        reason_str, extras_str, metrics = result
        self.assertIn('ratio=', reason_str)
        self.assertIn('delta=', reason_str)

    def test_returns_none_when_ratio_below_threshold(self):
        risk = _risk(gamma_theta_ratio=1.0, net_short_delta=0.20)
        result = self._call(risk=risk)
        self.assertIsNone(result)

    def test_returns_none_when_delta_below_min(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.10)
        result = self._call(risk=risk, min_delta=0.15)
        self.assertIsNone(result)

    def test_returns_none_when_insufficient_profit(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.20)
        # pnl_per_share / entry_premium = 0.10/1.00 = 10% < 25% min, no urgent
        result = self._call(risk=risk, pnl_per_share=0.10, entry_premium=1.00)
        self.assertIsNone(result)

    def test_urgent_delta_overrides_profit_requirement(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.35)
        # pnl insufficient but urgent_delta=0.30 is exceeded
        result = self._call(risk=risk, pnl_per_share=0.05, entry_premium=1.00,
                            urgent_delta=0.30)
        self.assertIsNotNone(result)
        reason_str, _, _ = result
        self.assertIn('urgent-delta override', reason_str)

    def test_near_expiry_dte_1_non_urgent_returns_none(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.20)
        result = self._call(risk=risk, dte=1, pnl_per_share=0.30)
        self.assertIsNone(result)

    def test_near_expiry_dte_0_non_urgent_returns_none(self):
        risk = _risk(gamma_theta_ratio=5.0, net_short_delta=0.25)
        result = self._call(risk=risk, dte=0, pnl_per_share=0.50)
        self.assertIsNone(result)

    def test_near_expiry_urgent_still_fires(self):
        # DTE=1 but delta is urgent (>= urgent_delta=0.30)
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.35)
        result = self._call(risk=risk, dte=1, pnl_per_share=0.05,
                            entry_premium=1.00, urgent_delta=0.30)
        self.assertIsNotNone(result)

    def test_returns_metrics_dict(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.20, risk_score=0.8)
        result = self._call(risk=risk, dte=5, pnl_per_share=0.30)
        self.assertIsNotNone(result)
        _, _, metrics = result
        self.assertIn('ratio', metrics)
        self.assertIn('short_delta', metrics)
        self.assertIn('risk_score', metrics)
        self.assertIn('dte', metrics)
        self.assertEqual(metrics['dte'], 5)

    def test_zero_entry_premium_does_not_raise(self):
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=0.35)
        # zero entry_premium → profit_captured_pct=0.0, urgent delta fires
        result = self._call(risk=risk, entry_premium=0.0, pnl_per_share=0.30,
                            urgent_delta=0.30)
        self.assertIsNotNone(result)

    def test_net_short_delta_abs_taken(self):
        # net_short_delta is negative (e.g. from call side) — abs() should be taken
        risk = _risk(gamma_theta_ratio=2.0, net_short_delta=-0.20)
        result = self._call(risk=risk, pnl_per_share=0.30)
        self.assertIsNotNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# GammaRiskRule.evaluate
# ═══════════════════════════════════════════════════════════════════════════════

class TestGammaRiskRule(unittest.TestCase):

    def test_disabled_returns_none(self):
        rule = _gamma_rule(enabled=False)
        result = rule.evaluate(1.0, 0.5, 0.3, 100.0, {})
        self.assertIsNone(result)

    def test_no_gamma_check_result_returns_none(self):
        rule = _gamma_rule(enabled=True)
        result = rule.evaluate(1.0, 0.5, 0.3, 100.0, {})
        self.assertIsNone(result)

    def test_with_gamma_check_result_returns_close_signal(self):
        rule = _gamma_rule(enabled=True)
        gamma_result = ('ratio=2.0>=1.5, delta=0.200', 'γ/θ=2.00  Δshort=0.200', {
            'ratio': 2.0, 'short_delta': 0.20, 'risk_score': 0.75, 'dte': 10
        })
        result = rule.evaluate(1.0, 0.5, 0.3, 100.0, {}, gamma_check_result=gamma_result)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CloseSignal)
        self.assertEqual(result.reason_tag, 'GAMMA_RISK')
        self.assertEqual(result.reason_str, 'ratio=2.0>=1.5, delta=0.200')

    def test_close_signal_contains_metrics(self):
        rule = _gamma_rule(enabled=True)
        metrics = {'ratio': 1.8, 'short_delta': 0.25, 'risk_score': 0.9, 'dte': 3}
        gamma_result = ('reason', 'extras', metrics)
        result = rule.evaluate(1.0, 0.5, 0.3, 100.0, {}, gamma_check_result=gamma_result)
        self.assertEqual(result.metrics, metrics)

    def test_rule_name(self):
        rule = _gamma_rule()
        self.assertEqual(rule.name, 'GAMMA_RISK')


# ═══════════════════════════════════════════════════════════════════════════════
# RulePipeline.evaluate_all
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_rule(name='TEST', returns=None):
    rule = MagicMock()
    rule.name = name
    rule.evaluate.return_value = returns
    return rule


def _signal(tag='STOP_LOSS'):
    return CloseSignal(
        reason_tag=tag,
        reason_str='test reason',
        extras_str='extras',
        metrics={},
    )


class TestRulePipeline(unittest.TestCase):

    def test_returns_none_when_all_rules_pass(self):
        rules = [_mock_rule('A'), _mock_rule('B'), _mock_rule('C')]
        pipeline = RulePipeline(rules)
        result = pipeline.evaluate_all(1.0, 0.5, 0.3, 100.0, {})
        self.assertIsNone(result)

    def test_returns_first_triggered_signal(self):
        sig = _signal('STOP_LOSS')
        rules = [_mock_rule('A'), _mock_rule('B', returns=sig), _mock_rule('C')]
        pipeline = RulePipeline(rules)
        result = pipeline.evaluate_all(1.0, 0.5, 0.3, 100.0, {})
        self.assertIs(result, sig)

    def test_stops_at_first_trigger(self):
        sig1 = _signal('STOP_LOSS')
        sig2 = _signal('GAMMA_RISK')
        r1 = _mock_rule('A')
        r2 = _mock_rule('B', returns=sig1)
        r3 = _mock_rule('C', returns=sig2)
        pipeline = RulePipeline([r1, r2, r3])
        result = pipeline.evaluate_all(1.0, 0.5, 0.3, 100.0, {})
        self.assertEqual(result.reason_tag, 'STOP_LOSS')
        # Third rule should never be called
        r3.evaluate.assert_not_called()

    def test_empty_pipeline_returns_none(self):
        pipeline = RulePipeline([])
        result = pipeline.evaluate_all(1.0, 0.5, 0.3, 100.0, {})
        self.assertIsNone(result)

    def test_kwargs_forwarded_to_rules(self):
        rule = _mock_rule('A')
        pipeline = RulePipeline([rule])
        pipeline.evaluate_all(1.0, 0.5, 0.3, 100.0, {}, gamma_check_result='x')
        rule.evaluate.assert_called_once_with(1.0, 0.5, 0.3, 100.0, {},
                                               gamma_check_result='x')

    def test_single_rule_triggered(self):
        sig = _signal('PROFIT_TAKE')
        rule = _mock_rule('PT', returns=sig)
        pipeline = RulePipeline([rule])
        result = pipeline.evaluate_all(2.0, 1.0, 1.5, 200.0, {'id': 1})
        self.assertEqual(result.reason_tag, 'PROFIT_TAKE')


if __name__ == '__main__':
    unittest.main()
