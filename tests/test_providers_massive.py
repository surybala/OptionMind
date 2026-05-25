from datetime import UTC, date, datetime

import pytest
import requests

from ml.providers.massive import MassiveProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {}), timeout))
        return FakeResponse(self.responses.pop(0))


class FailingSession:
    def get(self, url, params=None, timeout=None):
        request = requests.Request("GET", url, params=params).prepare()
        error = requests.ConnectionError("network unavailable")
        error.request = request
        raise error


class NotFoundSession:
    def get(self, url, params=None, timeout=None):
        request = requests.Request("GET", url, params=params).prepare()
        response = requests.Response()
        response.status_code = 404
        response._content = b'{"error":"not found"}'
        response.request = request
        error = requests.HTTPError("not found", response=response)
        error.request = request
        raise error


def provider(responses):
    return MassiveProvider(api_key="key", base_url="https://api.massive.test", session=FakeSession(responses))


def test_from_env_accepts_massive_or_legacy_polygon_key(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("POLYGON_API_KEY", "legacy")
    assert MassiveProvider.from_env().api_key == "legacy"

    monkeypatch.setenv("MASSIVE_API_KEY", "new")
    assert MassiveProvider.from_env().api_key == "new"


def test_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MASSIVE_API_KEY or POLYGON_API_KEY"):
        MassiveProvider.from_env()


def test_get_stock_bars_normalizes_aggregate_rows():
    p = provider(
        [
            {
                "results": [
                    {"t": 1_779_984_000_000, "o": 500, "h": 505, "l": 499, "c": 504, "v": 1000, "n": 10, "vw": 502}
                ]
            }
        ]
    )

    bars = p.get_stock_bars(
        ["SPY"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
        "1Day",
    )

    assert bars["SPY"][0].close == 504
    assert bars["SPY"][0].trade_count == 10
    assert bars["SPY"][0].source == "massive"
    url, params, _ = p.session.calls[0]
    assert url == "https://api.massive.test/v2/aggs/ticker/SPY/range/1/day/2026-05-14/2026-05-15"
    assert params["adjusted"] == "false"
    assert params["apiKey"] == "key"


def test_get_option_contracts_normalizes_and_paginates_metadata():
    p = provider(
        [
            {
                "results": [
                    {
                        "ticker": "O:SPY260626P00500000",
                        "underlying_ticker": "SPY",
                        "expiration_date": "2026-06-26",
                        "strike_price": 500,
                        "contract_type": "put",
                        "exercise_style": "american",
                    }
                ],
                "next_url": "https://api.massive.test/page-2",
            },
            {
                "results": [
                    {
                        "ticker": "O:SPY260626C00550000",
                        "underlying_ticker": "SPY",
                        "expiration_date": "2026-06-26",
                        "strike_price": 550,
                        "contract_type": "call",
                    }
                ]
            },
        ]
    )

    contracts = p.get_option_contracts(
        ["SPY"],
        expiration_gte=date(2026, 6, 1),
        expiration_lte=date(2026, 6, 30),
        status="inactive",
    )

    assert [contract.symbol for contract in contracts] == ["SPY260626P00500000", "SPY260626C00550000"]
    assert contracts[0].underlying == "SPY"
    assert contracts[0].option_type == "put"
    assert contracts[0].strike == 500.0
    first_params = p.session.calls[0][1]
    assert first_params["expired"] == "true"
    assert first_params["expiration_date.gte"] == "2026-06-01"
    assert p.session.calls[1][0] == "https://api.massive.test/page-2"


def test_get_option_bars_adds_api_prefix_and_normalizes_keys():
    p = provider(
        [
            {
                "results": [
                    {"t": 1_779_984_000_000, "o": 4.2, "h": 4.8, "l": 3.9, "c": 4.4, "v": 120, "n": 18, "vw": 4.35}
                ]
            }
        ]
    )

    bars = p.get_option_bars(
        ["SPY260626P00500000"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
        "1Day",
    )

    assert list(bars) == ["SPY260626P00500000"]
    assert bars["SPY260626P00500000"][0].vwap == 4.35
    assert "/O%3ASPY260626P00500000/" in p.session.calls[0][0]


def test_get_option_trades_normalizes_trade_rows():
    p = provider(
        [
            {
                "results": [
                    {
                        "sip_timestamp": 1_779_984_060_000_000_000,
                        "price": 4.35,
                        "size": 3,
                        "exchange": 65,
                        "conditions": [209],
                    }
                ]
            }
        ]
    )

    trades = p.get_option_trades(
        ["O:SPY260626P00500000"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
    )

    trade = trades["SPY260626P00500000"][0]
    assert trade.price == 4.35
    assert trade.size == 3
    assert trade.exchange == "65"
    assert trade.conditions == ("209",)
    assert "/O%3ASPY260626P00500000" in p.session.calls[0][0]


def test_get_current_option_chain_normalizes_snapshot_rows():
    p = provider(
        [
            {
                "results": [
                    {
                        "details": {"ticker": "O:SPY260626P00500000"},
                        "last_quote": {
                            "sip_timestamp": 1_779_984_060_000_000_000,
                            "bid_price": 4.3,
                            "ask_price": 4.4,
                        },
                        "last_trade": {"price": 4.35},
                        "implied_volatility": 0.22,
                        "greeks": {"delta": -0.25, "gamma": 0.03, "theta": -0.04, "vega": 0.11},
                    }
                ]
            }
        ]
    )

    chain = p.get_current_option_chain("SPY", expiration_lte=date(2026, 6, 30))
    snapshot = chain["SPY260626P00500000"]

    assert snapshot.bid == 4.3
    assert snapshot.ask == 4.4
    assert snapshot.last == 4.35
    assert snapshot.implied_volatility == 0.22
    assert snapshot.greeks.delta == -0.25
    assert p.session.calls[0][1]["expiration_date.lte"] == "2026-06-30"


def test_massive_request_errors_redact_api_key():
    p = MassiveProvider(api_key="super-secret", base_url="https://api.massive.test", session=FailingSession())

    with pytest.raises(RuntimeError) as exc:
        p.get_stock_bars(
            ["SPY"],
            datetime(2026, 5, 14, tzinfo=UTC),
            datetime(2026, 5, 15, tzinfo=UTC),
            "1Day",
        )

    message = str(exc.value)
    assert "super-secret" not in message
    assert "apiKey=%2A%2A%2A" in message


def test_option_aggregate_404_returns_empty_bars():
    p = MassiveProvider(api_key="super-secret", base_url="https://api.massive.test", session=NotFoundSession())

    bars = p.get_option_bars(
        ["SPY260626P00500000"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
        "1Day",
    )

    assert bars == {"SPY260626P00500000": []}
