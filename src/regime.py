from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


_SEVERITY = {
    "GREEN": 0,
    "YELLOW": 1,
    "ORANGE": 2,
    "RED": 3,
}


@dataclass(frozen=True)
class RegimeResult:
    label: str
    quantity_multiplier: float = 1.0
    top_n_multiplier: float = 1.0
    pause_new_trades: bool = False
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float | bool | None] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.metrics.get("enabled", True))


class RegimeService:
    """Classify market regime and return new-trade throttle settings."""

    def __init__(self, config: dict):
        risk = config.get("risk_parameters", {}) if isinstance(config, dict) else {}
        self._cfg = risk.get("regime_filter", {}) or {}

    def evaluate(
        self,
        *,
        vix_current: float | None = None,
        vix_history: Iterable[float] | None = None,
        spy_history: Iterable[float] | None = None,
    ) -> RegimeResult:
        if not self._cfg.get("enabled", False):
            return RegimeResult(
                "GREEN",
                reasons=["regime filter disabled"],
                metrics={"enabled": False},
            )

        vix_vals = _clean_series(vix_history)
        spy_vals = _clean_series(spy_history)
        if vix_current is None and vix_vals:
            vix_current = vix_vals[-1]

        if vix_current is None:
            if self._cfg.get("fail_closed", False):
                return self._result("RED", ["regime data unavailable"], {})
            return self._result("GREEN", ["regime data unavailable; fail-open"], {})

        label = self._label_from_vix(float(vix_current))
        reasons = [f"VIX {vix_current:.1f} -> {label}"]
        metrics: dict[str, float | bool | None] = {
            "vix": round(float(vix_current), 4),
        }

        spike_label, spike_reason, spike_metrics = self._vix_spike(vix_vals)
        if spike_label:
            label = _max_label(label, spike_label)
            reasons.append(spike_reason)
            metrics.update(spike_metrics)

        trend_label, trend_reason, trend_metrics = self._spy_trend(spy_vals, vix_vals)
        if trend_label:
            label = _max_label(label, trend_label)
            reasons.append(trend_reason)
            metrics.update(trend_metrics)

        rv_label, rv_reason, rv_metrics = self._realized_vol(spy_vals)
        if rv_label:
            label = _max_label(label, rv_label)
            reasons.append(rv_reason)
            metrics.update(rv_metrics)

        return self._result(label, reasons, metrics)

    def _label_from_vix(self, vix: float) -> str:
        cfg = self._cfg.get("vix", {}) or {}
        green_below = float(cfg.get("green_below", 18.0))
        yellow_below = float(cfg.get("yellow_below", 25.0))
        orange_below = float(cfg.get("orange_below", 32.0))
        if vix >= orange_below:
            return "RED"
        if vix >= yellow_below:
            return "ORANGE"
        if vix >= green_below:
            return "YELLOW"
        return "GREEN"

    def _vix_spike(
        self, vix_vals: Sequence[float]
    ) -> tuple[str | None, str, dict[str, float]]:
        cfg = self._cfg.get("vix_spike", {}) or {}
        metrics: dict[str, float] = {}
        if len(vix_vals) < 2:
            return None, "", metrics

        one_day = _pct_change(vix_vals[-2], vix_vals[-1])
        metrics["vix_one_day_change_pct"] = round(one_day, 4)
        red_1d = float(cfg.get("red_one_day_pct", 0.25))
        yellow_1d = float(cfg.get("one_day_pct", 0.15))
        if one_day >= red_1d:
            return "RED", f"VIX 1-day spike {one_day:.0%}", metrics

        label = "YELLOW" if one_day >= yellow_1d else None
        reason = f"VIX 1-day rise {one_day:.0%}" if label else ""

        if len(vix_vals) >= 4:
            three_day = _pct_change(vix_vals[-4], vix_vals[-1])
            metrics["vix_three_day_change_pct"] = round(three_day, 4)
            orange_3d = float(cfg.get("three_day_pct", 0.25))
            if three_day >= orange_3d:
                return "ORANGE", f"VIX 3-day rise {three_day:.0%}", metrics

        return label, reason, metrics

    def _spy_trend(
        self, spy_vals: Sequence[float], vix_vals: Sequence[float]
    ) -> tuple[str | None, str, dict[str, float | bool]]:
        cfg = self._cfg.get("trend", {}) or {}
        fast_n = int(cfg.get("sma_fast", 20))
        slow_n = int(cfg.get("sma_slow", 50))
        if len(spy_vals) < max(fast_n, slow_n):
            return None, "", {}

        spot = spy_vals[-1]
        fast = sum(spy_vals[-fast_n:]) / fast_n
        slow = sum(spy_vals[-slow_n:]) / slow_n
        below_fast = spot < fast
        below_slow = spot < slow
        vix_rising = len(vix_vals) >= 5 and vix_vals[-1] > sum(vix_vals[-5:]) / 5
        metrics = {
            "spy_close": round(spot, 4),
            "spy_sma_fast": round(fast, 4),
            "spy_sma_slow": round(slow, 4),
            "spy_below_fast": below_fast,
            "spy_below_slow": below_slow,
            "vix_rising": vix_rising,
        }
        if below_fast and below_slow and vix_rising:
            return "ORANGE", "SPY below fast/slow averages with rising VIX", metrics
        if below_fast and below_slow:
            return "YELLOW", "SPY below fast/slow averages", metrics
        return None, "", metrics

    def _realized_vol(
        self, spy_vals: Sequence[float]
    ) -> tuple[str | None, str, dict[str, float]]:
        cfg = self._cfg.get("realized_vol", {}) or {}
        fast_n = int(cfg.get("fast_days", 5))
        slow_n = int(cfg.get("slow_days", 20))
        threshold = float(cfg.get("expansion_multiple", 1.5))
        if len(spy_vals) < slow_n + 1:
            return None, "", {}

        returns = [
            math.log(spy_vals[i] / spy_vals[i - 1])
            for i in range(1, len(spy_vals))
            if spy_vals[i] > 0 and spy_vals[i - 1] > 0
        ]
        if len(returns) < slow_n:
            return None, "", {}

        rv_fast = _ann_vol(returns[-fast_n:])
        rv_slow = _ann_vol(returns[-slow_n:])
        ratio = rv_fast / rv_slow if rv_slow > 0 else 0.0
        metrics = {
            "spy_rv_fast": round(rv_fast, 4),
            "spy_rv_slow": round(rv_slow, 4),
            "spy_rv_ratio": round(ratio, 4),
        }
        if rv_slow > 0 and ratio >= threshold:
            return "YELLOW", f"SPY realized vol expansion {ratio:.1f}x", metrics
        return None, "", metrics

    def _result(
        self,
        label: str,
        reasons: list[str],
        metrics: dict[str, float | bool | None],
    ) -> RegimeResult:
        label = label if label in _SEVERITY else "GREEN"
        q_mult = 1.0
        if label == "YELLOW":
            q_mult = float(self._cfg.get("yellow_quantity_multiplier", 0.65))
        elif label == "ORANGE":
            q_mult = float(self._cfg.get("orange_quantity_multiplier", 0.30))

        metrics = dict(metrics)
        metrics["enabled"] = True
        return RegimeResult(
            label=label,
            quantity_multiplier=max(0.0, min(1.0, q_mult)),
            top_n_multiplier=1.0,
            pause_new_trades=False,
            reasons=reasons,
            metrics=metrics,
        )


def _clean_series(values: Iterable[float] | None) -> list[float]:
    out: list[float] = []
    if values is None:
        return out
    for val in values:
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0:
            out.append(f)
    return out


def _pct_change(start: float, end: float) -> float:
    return (end / start) - 1.0 if start > 0 else 0.0


def _ann_vol(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def _max_label(a: str, b: str) -> str:
    return a if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0) else b
