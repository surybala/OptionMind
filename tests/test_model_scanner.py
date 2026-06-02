import json
import threading
import time
from datetime import UTC, datetime, timedelta

from ml.models.registry import load_registry, promote_model, register_model_artifact, save_registry
from ml.providers.models import OptionChainSnapshot, PriceBar
from src.model_scanner import LivePaperInferenceProvider, ModelScanner


def test_disabled_model_scanner_returns_no_picks():
    scanner = ModelScanner({"ml_scanner": {"enabled": False}})

    assert scanner.get_top_picks(["SPY"], n=5) == []


def test_disabled_model_scanner_does_not_load_configured_provider():
    scanner = ModelScanner({"ml_scanner": {"enabled": False, "provider": "missing.module:Provider"}})

    assert scanner.model_provider is None
    assert scanner.get_top_picks(["SPY"], n=5) == []


def test_enabled_model_scanner_without_provider_returns_no_picks():
    scanner = ModelScanner({"ml_scanner": {"enabled": True, "provider": ""}})

    assert scanner.get_top_picks(["SPY"], n=5) == []


def test_live_paper_inference_provider_rejects_non_alpaca_data_provider_in_hft_mode():
    try:
        LivePaperInferenceProvider(
            {
                "hft_mode": True,
                "ml_scanner": {
                    "data_provider": "custom.module:Provider",
                },
            }
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "HFT mode requires ml_scanner.data_provider to be Alpaca-backed" in str(exc)


def test_model_scanner_normalizes_provider_candidates():
    def provider(ticker_list, n, config):
        return [
            {
                "symbol": ticker_list[0],
                "strategy": "PCS",
                "expiry": "2026-06-26",
                "premium": 0.50,
                "model_score": 0.82,
                "probability_of_profit": 0.71,
            }
        ]

    scanner = ModelScanner({"ml_scanner": {"enabled": True}})
    scanner.model_provider = provider

    picks = scanner.get_top_picks(["SPY"], n=5)

    assert picks[0]["source"] == "ml_model"
    assert picks[0]["score"] == 0.82
    assert picks[0]["prob_win"] == 0.71
    assert picks[0]["quantity"] == 1


def test_model_scanner_filters_by_min_score():
    def provider(ticker_list, n, config):
        return [
            {"symbol": "SPY", "model_score": 0.40},
            {"symbol": "QQQ", "model_score": 0.90},
        ]

    scanner = ModelScanner({"ml_scanner": {"enabled": True, "min_score": 0.5}})
    scanner.model_provider = provider

    picks = scanner.get_top_picks(["SPY", "QQQ"], n=5)

    assert [pick["symbol"] for pick in picks] == ["QQQ"]


class FakeLiveProvider:
    def __init__(self):
        self.now = datetime(2026, 5, 24, 16, 0, tzinfo=UTC)

    def get_stock_bars(self, symbols, start, end, timeframe):
        bars = {}
        for symbol in symbols:
            close = 100.0 if symbol == "SPY" else 18.0
            rows = []
            for idx in range(70):
                ts = self.now - timedelta(days=69 - idx)
                value = close + idx * 0.05
                rows.append(PriceBar(symbol, ts, value - 0.2, value + 0.2, value - 0.3, value, volume=1000 + idx))
            bars[symbol] = rows
        return bars

    def get_current_option_chain(self, underlying, expiration_gte=None, expiration_lte=None, limit=None):
        return {
            "SPY260619P00095000": OptionChainSnapshot(
                "SPY260619P00095000", underlying, self.now, bid=1.50, ask=1.60, last=1.55, source="fake"
            ),
            "SPY260619P00090000": OptionChainSnapshot(
                "SPY260619P00090000", underlying, self.now, bid=0.40, ask=0.50, last=0.45, source="fake"
            ),
            "SPY260619C00110000": OptionChainSnapshot(
                "SPY260619C00110000", underlying, self.now, bid=1.20, ask=1.30, last=1.25, source="fake"
            ),
            "SPY260619C00115000": OptionChainSnapshot(
                "SPY260619C00115000", underlying, self.now, bid=0.30, ask=0.40, last=0.35, source="fake"
            ),
        }


class WiderSpreadLiveProvider(FakeLiveProvider):
    def get_current_option_chain(self, underlying, expiration_gte=None, expiration_lte=None, limit=None):
        chain = super().get_current_option_chain(underlying, expiration_gte, expiration_lte, limit)
        chain.update(
            {
                "SPY260619P00085000": OptionChainSnapshot(
                    "SPY260619P00085000", underlying, self.now, bid=0.20, ask=0.25, last=0.22, source="fake"
                ),
                "SPY260619C00120000": OptionChainSnapshot(
                    "SPY260619C00120000", underlying, self.now, bid=0.15, ask=0.20, last=0.17, source="fake"
                ),
            }
        )
        return chain


class MixedDteLiveProvider(FakeLiveProvider):
    def get_current_option_chain(self, underlying, expiration_gte=None, expiration_lte=None, limit=None):
        return {
            "SPY260610P00095000": OptionChainSnapshot(
                "SPY260610P00095000", underlying, self.now, bid=1.40, ask=1.50, last=1.45, source="fake"
            ),
            "SPY260610P00090000": OptionChainSnapshot(
                "SPY260610P00090000", underlying, self.now, bid=0.40, ask=0.50, last=0.45, source="fake"
            ),
            "SPY260630P00095000": OptionChainSnapshot(
                "SPY260630P00095000", underlying, self.now, bid=1.60, ask=1.70, last=1.65, source="fake"
            ),
            "SPY260630P00090000": OptionChainSnapshot(
                "SPY260630P00090000", underlying, self.now, bid=0.50, ask=0.60, last=0.55, source="fake"
            ),
        }


class SlowConcurrentLiveProvider(FakeLiveProvider):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def get_current_option_chain(self, underlying, expiration_gte=None, expiration_lte=None, limit=None):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
            return super().get_current_option_chain(
                underlying,
                expiration_gte=expiration_gte,
                expiration_lte=expiration_lte,
                limit=limit,
            )
        finally:
            with self._lock:
                self._active -= 1


def _champion_registry(tmp_path):
    artifact_path = _linear_ranker_artifact(tmp_path, "linear.json", option_entry_price_coef=10.0)
    registry = register_model_artifact(load_registry(tmp_path / "registry.json"), artifact_path, model_id="champion")
    registry = promote_model(registry, "champion")
    return save_registry(registry, tmp_path / "registry.json")


def _linear_ranker_artifact(tmp_path, filename: str, *, option_entry_price_coef: float, intercept: float = 0.0):
    artifact_path = tmp_path / filename
    artifact_path.write_text(
        json.dumps(
            {
                "model_type": "linear_least_squares_v001",
                "created_at": "2026-05-24T00:00:00+00:00",
                "target_column": "expected_pnl",
                "feature_version": "features_v002",
                "label_version": "short_option_labels_v001",
                "data_range": {"start": "2026-01-01T00:00:00+00:00", "end": "2026-05-01T00:00:00+00:00"},
                "feature_columns": ["option_entry_price", "dte", "underlying_close", "strike_distance_pct"],
                "fill_values": {
                    "option_entry_price": 0.0,
                    "dte": 30.0,
                    "underlying_close": 100.0,
                    "strike_distance_pct": 0.0,
                },
                "intercept": intercept,
                "coefficients": {
                    "option_entry_price": option_entry_price_coef,
                    "dte": 0.0,
                    "underlying_close": 0.0,
                    "strike_distance_pct": 0.0,
                },
                "metrics": {"test_top_decile_actual_mean": 12.3},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def test_live_paper_inference_provider_scores_current_spreads_from_champion(tmp_path):
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert [pick["strategy"] for pick in picks] == ["PCS", "CCS"]
    assert picks[0]["symbol"] == "SPY"
    assert picks[0]["premium"] == 1.0
    assert picks[0]["short_strike"] == 95.0
    assert picks[0]["long_strike"] == 90.0
    assert picks[0]["model_id"] == "champion"
    assert {pick["ranker_strategy"] for pick in picks} == {"DEFAULT"}
    assert picks[0]["feature_version"] == "features_v002"
    assert picks[0]["ranking_context"]["dte"] == 26
    assert 0.08 < picks[0]["ranking_context"]["short_leg_otm_pct"] < 0.09
    assert "score=" in picks[0]["ranking_reason"]
    assert "otm=" in picks[0]["ranking_reason"]


def test_live_paper_inference_provider_can_emit_multiple_spread_widths(tmp_path):
    provider = WiderSpreadLiveProvider()
    registry_path = _champion_registry(tmp_path)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "spread_widths": [5, 10], "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": False},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=10)

    assert {pick["width"] for pick in picks if pick["strategy"] == "PCS"} == {5.0, 10.0}


def test_live_paper_inference_provider_respects_max_dte_cap(tmp_path):
    provider = MixedDteLiveProvider()
    registry_path = _champion_registry(tmp_path)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 21,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": False},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert len(picks) == 1
    assert picks[0]["expiry"] == "2026-06-10"


def test_live_paper_inference_provider_fetches_option_chains_in_parallel(tmp_path):
    provider = SlowConcurrentLiveProvider()
    registry_path = _champion_registry(tmp_path)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "chain_fetch_workers": 4,
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY", "QQQ", "IWM"], n=6)

    assert len(picks) > 0
    assert provider.max_active >= 2


def test_live_paper_inference_provider_ignores_strategy_specific_ranker_config(tmp_path):
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    pcs_artifact = _linear_ranker_artifact(tmp_path, "pcs.json", option_entry_price_coef=10.0)
    ccs_artifact = _linear_ranker_artifact(tmp_path, "ccs.json", option_entry_price_coef=100.0)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "strategy_rankers": {
                    "PCS": {"artifact_path": str(pcs_artifact)},
                    "CCS": {"artifact_path": str(ccs_artifact)},
                },
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert [pick["strategy"] for pick in picks] == ["PCS", "CCS"]
    assert {pick["ranker_strategy"] for pick in picks} == {"DEFAULT"}
    assert {pick["model_id"] for pick in picks} == {"champion"}
    assert str(pcs_artifact) not in {pick["model_artifact_path"] for pick in picks}
    assert str(ccs_artifact) not in {pick["model_artifact_path"] for pick in picks}


def test_live_paper_inference_provider_applies_regime_allocation_to_ranked_picks(tmp_path):
    provider = WiderSpreadLiveProvider()
    registry_path = _champion_registry(tmp_path)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "pick_selection": {
                "mode": "model_ranked",
                "regime_allocation": {
                    "enabled": True,
                    "regimes": {
                        "ORANGE": {
                            "PCS": {"max_fraction": 0.5},
                            "CCS": {"min_fraction": 0.5},
                        }
                    },
                },
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "spread_widths": [5, 10], "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "spread_widths": [5, 10], "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )
    inference.set_runtime_regime("ORANGE")

    picks = inference.get_top_picks(["SPY"], n=2)

    assert len(picks) == 2
    assert {pick["strategy"] for pick in picks} == {"PCS", "CCS"}


def _large_loss_classifier_artifact(tmp_path, always_veto: bool = False):
    """Write a minimal linear-model artifact that mimics the large-loss classifier interface."""
    score = 0.95 if always_veto else 0.10
    artifact_path = tmp_path / "large_loss_clf.json"
    artifact_path.write_text(
        json.dumps(
            {
                "model_type": "linear_least_squares_v001",
                "created_at": "2026-05-27T00:00:00+00:00",
                "target_column": "large_loss_label",
                "feature_version": "features_v005",
                "label_version": "credit_spread_labels_v002",
                "data_range": {"start": "2022-01-01T00:00:00+00:00", "end": "2026-04-25T00:00:00+00:00"},
                "feature_columns": ["option_entry_price"],
                "fill_values": {"option_entry_price": 0.0},
                "intercept": score,
                "coefficients": {"option_entry_price": 0.0},
                "metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _stop_loss_classifier_artifact(tmp_path, always_veto: bool = False):
    """Write a minimal linear-model artifact that mimics the stop-loss classifier interface."""
    score = 0.95 if always_veto else 0.10
    artifact_path = tmp_path / "stop_loss_clf.json"
    artifact_path.write_text(
        json.dumps(
            {
                "model_type": "linear_least_squares_v001",
                "created_at": "2026-05-27T00:00:00+00:00",
                "target_column": "stop_loss_hit",
                "feature_version": "features_v005",
                "label_version": "credit_spread_labels_v002",
                "data_range": {"start": "2022-01-01T00:00:00+00:00", "end": "2026-04-25T00:00:00+00:00"},
                "feature_columns": ["option_entry_price"],
                "fill_values": {"option_entry_price": 0.0},
                "intercept": score,
                "coefficients": {"option_entry_price": 0.0},
                "metrics": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def test_large_loss_classifier_veto_removes_picks(tmp_path):
    """All spreads should be vetoed when the classifier always predicts p=0.95 > threshold."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    clf_path = _large_loss_classifier_artifact(tmp_path, always_veto=True)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
                "large_loss_classifier_path": str(clf_path),
                "large_loss_veto_threshold": 0.70,
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert picks == [], "Classifier with p=0.95 should veto all spreads"


def test_large_loss_classifier_pass_includes_prob_in_pick(tmp_path):
    """Picks that survive the classifier should carry large_loss_prob in the pick dict."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    clf_path = _large_loss_classifier_artifact(tmp_path, always_veto=False)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
                "large_loss_classifier_path": str(clf_path),
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert len(picks) > 0, "Low-risk classifier should pass some spreads"
    assert all("large_loss_prob" in p for p in picks)
    assert all(p["large_loss_prob"] is not None for p in picks)


def test_stop_loss_classifier_veto_removes_picks(tmp_path):
    """Spreads should be vetoed when stop-loss risk exceeds the configured threshold."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    clf_path = _stop_loss_classifier_artifact(tmp_path, always_veto=True)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
                "stop_loss_classifier_path": str(clf_path),
                "stop_loss_veto_threshold": 0.70,
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert picks == [], "Classifier with p=0.95 should veto all spreads"


def test_stop_loss_classifier_pass_includes_prob_in_pick(tmp_path):
    """Picks that survive the stop-loss classifier should carry stop_loss_prob in the pick dict."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)
    clf_path = _stop_loss_classifier_artifact(tmp_path, always_veto=False)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
                "stop_loss_classifier_path": str(clf_path),
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert len(picks) > 0
    assert all("stop_loss_prob" in p for p in picks)
    assert all(p["stop_loss_prob"] is not None for p in picks)


def test_no_large_loss_classifier_leaves_large_loss_prob_none(tmp_path):
    """Without a classifier configured, large_loss_prob should be None in picks."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert len(picks) > 0
    assert all(p.get("large_loss_prob") is None for p in picks)
    assert all(p.get("stop_loss_prob") is None for p in picks)


def test_min_prob_profit_does_not_filter_picks(tmp_path):
    """min_prob_profit config should NOT filter picks — ML pipeline is the sole gate."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)

    # Set min_prob_profit to 0.99 (would reject everything if it were active)
    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10, "min_prob_profit": 0.99},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10, "min_prob_profit": 0.99},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    # Picks should still be returned — min_prob_profit no longer filters
    assert len(picks) > 0, "min_prob_profit should not filter picks; ML pipeline is the sole gate"


def test_prob_win_still_populated_in_picks(tmp_path):
    """prob_win is still computed and stored for display purposes, just not used as a filter."""
    provider = FakeLiveProvider()
    registry_path = _champion_registry(tmp_path)

    inference = LivePaperInferenceProvider(
        {
            "ml_scanner": {
                "registry_path": str(registry_path),
                "min_dte": 7,
                "max_dte": 45,
                "vix_symbol": "I:VIX",
            },
            "strategies": {
                "put_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
                "call_credit_spread": {"enabled": True, "strike_width": 5, "min_net_credit": 0.10},
            },
        },
        provider=provider,
        now_fn=lambda: provider.now,
    )

    picks = inference.get_top_picks(["SPY"], n=5)

    assert len(picks) > 0
    for pick in picks:
        assert "prob_win" in pick
        assert isinstance(pick["prob_win"], float)
        assert 0.0 <= pick["prob_win"] <= 1.0
        assert "ranking_context" in pick
        assert "ranking_reason" in pick
