import json
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


def _champion_registry(tmp_path):
    artifact_path = tmp_path / "linear.json"
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
                "intercept": 0.0,
                "coefficients": {
                    "option_entry_price": 10.0,
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
    registry = register_model_artifact(load_registry(tmp_path / "registry.json"), artifact_path, model_id="champion")
    registry = promote_model(registry, "champion")
    return save_registry(registry, tmp_path / "registry.json")


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
    assert picks[0]["feature_version"] == "features_v002"


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
