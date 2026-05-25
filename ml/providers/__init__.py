"""Provider interfaces and adapters for OptionMind ML data pipelines."""

from ml.providers.alpaca import AlpacaProvider
from ml.providers.fmp import FMPProvider
from ml.providers.fred import FREDProvider
from ml.providers.massive import MassiveProvider
from ml.providers.models import (
    DividendEvent,
    EarningsEvent,
    EconomicEvent,
    Greeks,
    OptionChainSnapshot,
    OptionContract,
    OptionTrade,
    PriceBar,
)
from ml.providers.protocols import (
    DividendDataProvider,
    EconomicCalendarProvider,
    EventDataProvider,
    MarketDataProvider,
    OptionChainProvider,
    OptionContractProvider,
    OptionPriceProvider,
    VolatilityDataProvider,
)

__all__ = [
    "AlpacaProvider",
    "DividendDataProvider",
    "DividendEvent",
    "EarningsEvent",
    "EconomicCalendarProvider",
    "EconomicEvent",
    "EventDataProvider",
    "FMPProvider",
    "FREDProvider",
    "Greeks",
    "MarketDataProvider",
    "MassiveProvider",
    "OptionChainProvider",
    "OptionContract",
    "OptionContractProvider",
    "OptionPriceProvider",
    "OptionTrade",
    "OptionChainSnapshot",
    "PriceBar",
    "VolatilityDataProvider",
]
