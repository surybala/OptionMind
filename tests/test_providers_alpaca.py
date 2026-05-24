from dataclasses import dataclass
from datetime import UTC, date, datetime

from ml.providers.alpaca import AlpacaProvider


@dataclass
class Obj:
    pass


def obj(**kwargs):
    value = Obj()
    for key, item in kwargs.items():
        setattr(value, key, item)
    return value


class Result:
    def __init__(self, data):
        self.data = data


class ContractsResult:
    def __init__(self, contracts):
        self.option_contracts = contracts


class FakeStockClient:
    def get_stock_bars(self, request):
        return Result(
            {
                "SPY": [
                    obj(
                        symbol="SPY",
                        timestamp=datetime(2026, 5, 14, 13, 30, tzinfo=UTC),
                        open=500,
                        high=505,
                        low=499,
                        close=504,
                        volume=1000,
                        trade_count=10,
                        vwap=502,
                    )
                ]
            }
        )


class FakeTradingClient:
    def get_option_contracts(self, request):
        return ContractsResult(
            [
                obj(
                    symbol="SPY260626P00500000",
                    underlying_symbol="SPY",
                    expiration_date=date(2026, 6, 26),
                    strike_price="500",
                    type="put",
                    status="active",
                    style="american",
                    root_symbol="SPY",
                )
            ]
        )


class FakeOptionClient:
    def get_option_bars(self, request):
        return Result(
            {
                "SPY260626P00500000": [
                    obj(
                        symbol="SPY260626P00500000",
                        timestamp=datetime(2026, 5, 14, tzinfo=UTC),
                        open=4.2,
                        high=4.8,
                        low=3.9,
                        close=4.4,
                        volume=120,
                        trade_count=18,
                        vwap=4.35,
                    )
                ]
            }
        )

    def get_option_trades(self, request):
        return Result(
            {
                "SPY260626P00500000": [
                    obj(
                        symbol="SPY260626P00500000",
                        timestamp=datetime(2026, 5, 14, 13, 31, tzinfo=UTC),
                        price=4.35,
                        size=3,
                        exchange="C",
                        conditions=["@ "],
                    )
                ]
            }
        )

    def get_option_chain(self, request):
        return Result(
            {
                "SPY260626P00500000": obj(
                    latest_quote=obj(
                        timestamp=datetime(2026, 5, 14, 13, 31, tzinfo=UTC),
                        bid_price=4.3,
                        ask_price=4.4,
                    ),
                    latest_trade=obj(price=4.35),
                    implied_volatility=0.22,
                    greeks=obj(delta=-0.25, gamma=0.03, theta=-0.04, vega=0.11, rho=-0.02),
                )
            }
        )


def provider():
    return AlpacaProvider(
        api_key="key",
        api_secret="secret",
        stock_client=FakeStockClient(),
        option_client=FakeOptionClient(),
        trading_client=FakeTradingClient(),
    )


def test_get_stock_bars_normalizes_alpaca_bars():
    bars = provider().get_stock_bars(
        ["SPY"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
        "1Day",
    )

    assert bars["SPY"][0].close == 504
    assert bars["SPY"][0].source == "alpaca"


def test_get_option_contracts_normalizes_metadata():
    contracts = provider().get_option_contracts(
        ["SPY"],
        expiration_gte=date(2026, 6, 1),
        expiration_lte=date(2026, 6, 30),
    )

    assert contracts[0].symbol == "SPY260626P00500000"
    assert contracts[0].underlying == "SPY"
    assert contracts[0].option_type == "put"
    assert contracts[0].strike == 500.0


def test_get_current_option_chain_normalizes_greeks_and_iv():
    chain = provider().get_current_option_chain("SPY")
    snapshot = chain["SPY260626P00500000"]

    assert snapshot.bid == 4.3
    assert snapshot.ask == 4.4
    assert snapshot.last == 4.35
    assert snapshot.implied_volatility == 0.22
    assert snapshot.greeks.delta == -0.25


def test_get_option_prices_normalizes_bars_and_trades():
    option_bars = provider().get_option_bars(
        ["SPY260626P00500000"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
        "1Day",
    )
    trades = provider().get_option_trades(
        ["SPY260626P00500000"],
        datetime(2026, 5, 14, tzinfo=UTC),
        datetime(2026, 5, 15, tzinfo=UTC),
    )

    assert option_bars["SPY260626P00500000"][0].vwap == 4.35
    assert trades["SPY260626P00500000"][0].price == 4.35
    assert trades["SPY260626P00500000"][0].size == 3
