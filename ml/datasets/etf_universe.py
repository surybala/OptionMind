"""ETF universe presets for historical ML dataset builds."""
from __future__ import annotations


# Liquid broad ETF option underlyings used for premium-selling research.
# The list intentionally favors diversified index, factor, sector, bond, and
# commodity ETFs with established listed options rather than every listed ETF.
BROAD_ETF_UNDERLYINGS = [
    # Broad US equity indexes
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "IWB",
    "IWF",
    "IWD",
    "MDY",
    "IWO",
    "IWN",
    "IJR",
    # International equity
    "EFA",
    "EEM",
    "EWJ",
    "FXI",
    # Sector ETFs
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
    # Industry / thematic ETFs with deep option markets
    "SMH",
    "SOXX",
    "KRE",
    "XRT",
    "XBI",
    # Rates, credit, and commodities
    "TLT",
    "IEF",
    "HYG",
    "LQD",
    "GLD",
    "SLV",
    "USO",
]


def broad_etf_underlyings() -> list[str]:
    """Return a copy of the broad ETF training universe."""
    return list(BROAD_ETF_UNDERLYINGS)
