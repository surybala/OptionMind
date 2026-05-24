from src.model_scanner import ModelScanner


def test_disabled_model_scanner_returns_no_picks():
    scanner = ModelScanner({"ml_scanner": {"enabled": False}})

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
