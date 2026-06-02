"""Provider interfaces and adapters for OptionMind ML data pipelines."""

from ml.providers.alpaca import AlpacaProvider
from ml.providers.fmp import FMPProvider
from ml.providers.fred import FREDProvider
from ml.providers.massive import MassiveProvider
from ml.providers.massive_flatfiles import MassiveFlatFilesClient
from ml.providers.parquet_minute import ParquetMinuteBarProvider
from ml.providers.yfinance_provider import YFinanceProvider
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
    "MassiveFlatFilesClient",
    "MassiveProvider",
    "ParquetMinuteBarProvider",
    "YFinanceProvider",
    "OptionChainProvider",
    "OptionContract",
    "OptionContractProvider",
    "OptionPriceProvider",
    "OptionTrade",
    "OptionChainSnapshot",
    "PriceBar",
    "VolatilityDataProvider",
]
