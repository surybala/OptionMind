"""Curated ETF universe presets shared by live scanning and ML dataset builds."""
from __future__ import annotations


# Core liquid ETF underlyings for live premium-selling scans.
# This preset intentionally excludes leveraged/inverse products, small-cap
# heavy ETFs, and more unstable thematic/commodity names so the live ML
# pipeline starts from a steadier options-selling universe.
STABLE_ETF_UNDERLYINGS = [
    # Core US equity indexes
    "SPY",
    "QQQ",
    "DIA",
    "VTI",
    "IWB",
    "IWF",
    "IWD",
    "MDY",
    # International equity
    "EFA",
    "EWJ",
    # Sector ETFs
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XLRE",
    # Rates, credit, and defensive macro proxies
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
]


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


def stable_etf_underlyings() -> list[str]:
    """Return a copy of the live stable ETF scanning universe."""
    return list(STABLE_ETF_UNDERLYINGS)


def broad_etf_underlyings() -> list[str]:
    """Return a copy of the broad ETF training universe."""
    return list(BROAD_ETF_UNDERLYINGS)
