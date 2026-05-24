"""
Index Constituent Universe
==========================

Fetches and caches the stock tickers that make up four major US indices:

  S&P 500      -- ~500 large-cap US stocks (sourced from Wikipedia)
  NASDAQ-100   -- top 100 non-financial NASDAQ stocks (sourced from Wikipedia)
  Dow Jones 30 -- 30 blue-chip stocks (sourced from Wikipedia)
  Russell 1000 -- top ~1000 US stocks by market cap (sourced from Wikipedia)

The combined universe is the *union* of all four (~1100 unique tickers after
deduplication).  Results are cached to data/index_cache.json with a
configurable TTL (default 24 hours) so repeated agent runs do not re-scrape
Wikipedia on every invocation.

A ``CACHE_VERSION`` constant is stored in the cache file; when the code is
updated to include a new index the version number is bumped so that stale
cached files are automatically invalidated on the next run.

Usage
-----
    from src.universe_indices import get_index_tickers

    tickers = get_index_tickers()          # uses 24-hour cache
    tickers = get_index_tickers(force_refresh=True)   # bypass cache
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import requests

CACHE_FILE       = "data/index_cache.json"
CACHE_TTL_HOURS  = 24
# Bump this whenever a new index is added so stale cached files are
# automatically invalidated on the next run.
CACHE_VERSION    = 2

# Wikipedia source URLs
_SP500_URL       = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX100_URL      = "https://en.wikipedia.org/wiki/Nasdaq-100"
_DOW30_URL       = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
_RUSSELL1000_URL = "https://en.wikipedia.org/wiki/Russell_1000_Index"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Characters that indicate a non-plain-equity ticker (warrants, preferred, ADR suffixes)
_BAD_CHARS = set("^$/+")


def _clean(symbol: str) -> str:
    """Normalise a ticker string from Wikipedia (strip whitespace, map dots to hyphens for BRK.B etc.)."""
    return symbol.strip().replace(".", "-")


def _is_valid(symbol: str) -> bool:
    """Return True if the symbol looks like a tradeable equity ticker."""
    if not symbol or len(symbol) > 6:
        return False
    if any(c in symbol for c in _BAD_CHARS):
        return False
    # Allow only alphanumerics plus hyphen (e.g. BRK-B)
    return bool(re.match(r'^[A-Z0-9][A-Z0-9\-]{0,5}$', symbol))


def _parse_wikipedia_table(url: str, log=None, table_id: str = None,
                           ticker_col_names: tuple[str, ...] = ("Symbol", "Ticker")) -> list[str]:
    """
    Download a Wikipedia page and extract the ticker column from the *first*
    table whose header row contains a "Symbol" or "Ticker" column.

    Targeting only the constituents table (rather than all tables on the page)
    avoids pulling in historical additions/removals from the "Recent changes"
    section, which would otherwise include many delisted tickers.

    Parameters
    ----------
    url : str
        Wikipedia page URL.
    log : Logger | None
        Optional logger.
    table_id : str | None
        If set, restrict to the ``<table id="...">`` with this id before
        falling back to header-based detection.
    ticker_col_names : tuple[str, ...]
        Column header strings that identify the ticker column (case-insensitive).
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        if log:
            log.warning(f"Failed to fetch {url}: {exc}")
        return []

    html = resp.text

    # ── Step 1: if a table_id is specified, narrow to that table ─────────────
    if table_id:
        m = re.search(
            r'<table[^>]+id=["\']' + re.escape(table_id) + r'["\'][^>]*>(.*?)</table>',
            html, re.DOTALL | re.IGNORECASE,
        )
        search_html = m.group(0) if m else html
    else:
        search_html = html

    # ── Step 2: find all <table> blocks ──────────────────────────────────────
    tables = re.findall(r'<table[^>]*>.*?</table>', search_html, re.DOTALL | re.IGNORECASE)

    candidates: set[str] = set()

    for table_html in tables:
        # Extract all rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        if not rows:
            continue

        # Find the header row (contains <th> cells)
        ticker_col_idx: int | None = None
        for row in rows[:3]:           # header is usually the first row
            headers = re.findall(r'<th[^>]*>(.*?)</th>', row, re.DOTALL | re.IGNORECASE)
            if not headers:
                continue
            headers_text = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
            for i, h in enumerate(headers_text):
                if any(col.lower() in h.lower() for col in ticker_col_names):
                    ticker_col_idx = i
                    break
            if ticker_col_idx is not None:
                break

        if ticker_col_idx is None:
            continue   # this table has no ticker column — skip it

        # Extract the ticker column from every data row.
        # Use <t[dh]> so that rows mixing <th> (e.g. company name) and <td>
        # (exchange, symbol …) are indexed consistently with the header row.
        for row in rows[1:]:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
            if ticker_col_idx >= len(cells):
                continue
            text = re.sub(r'<[^>]+>', '', cells[ticker_col_idx]).strip()
            # Some pages wrap the ticker in a link: take the first word
            first_word = text.split()[0] if text.split() else ''
            for token in (text, first_word):
                clean = _clean(token)
                if _is_valid(clean):
                    candidates.add(clean)
                    break

        # Stop after the first table that had a ticker column — this is the
        # constituents table.  Later tables (e.g. "Recent changes") would add
        # historical / delisted tickers.
        if candidates:
            break

    return sorted(candidates)


def _fetch_sp500(log=None) -> list[str]:
    """Return S&P 500 constituent tickers from Wikipedia."""
    # The S&P 500 page has a well-known table id; passing it ensures we never
    # accidentally pull from the historical "Recent changes" table.
    raw = _parse_wikipedia_table(_SP500_URL, log,
                                 table_id="constituents",
                                 ticker_col_names=("Symbol",))
    return [s for s in raw if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s)]


def _fetch_nasdaq100(log=None) -> list[str]:
    """Return NASDAQ-100 constituent tickers from Wikipedia."""
    raw = _parse_wikipedia_table(_NDX100_URL, log,
                                 ticker_col_names=("Ticker", "Symbol"))
    return [s for s in raw if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s)]


def _fetch_dow30(log=None) -> list[str]:
    """Return Dow Jones 30 constituent tickers from Wikipedia."""
    raw = _parse_wikipedia_table(_DOW30_URL, log,
                                 ticker_col_names=("Symbol", "Ticker", "Components"))
    return [s for s in raw if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s)]


def _fetch_russell1000(log=None) -> list[str]:
    """Return Russell 1000 constituent tickers from Wikipedia."""
    raw = _parse_wikipedia_table(_RUSSELL1000_URL, log,
                                 ticker_col_names=("Ticker", "Symbol"))
    return [s for s in raw if re.match(r'^[A-Z]{1,5}(-[A-Z])?$', s)]


# ---------------------------------------------------------------------------
# Fallback hardcoded lists (used when Wikipedia scraping yields too few results)
# These match the index compositions as of early 2026.
# ---------------------------------------------------------------------------

_SP500_FALLBACK: list[str] = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
    "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
    "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP",
    "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV",
    "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B",
    "BRO", "BSX", "BWA",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
    "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COST", "CPB", "CPRT", "CPT", "CRL", "CRM",
    "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA",
    "CVS", "CVX",
    "D", "DAL", "DD", "DE", "DECK", "DEI", "DELL", "DFS", "DG", "DGX",
    "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI",
    "DTE", "DUK", "DVA", "DVN",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
    "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO",
    "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV",
    "GD", "GDDY", "GE", "GEHC", "GEN", "GEV", "GFS", "GIS", "GL", "GLW",
    "GM", "GNRC", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH",
    "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
    "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI",
    "KMX", "KO", "KR",
    "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT",
    "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT",
    "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
    "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI", "MSFT",
    "MSI", "MTB", "MTCH", "MTD", "MU",
    "NCLH", "NDAQ", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG",
    "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA",
    "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "OTIS", "OXY",
    "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR",
    "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PWR", "PXD",
    "PYPL",
    "QCOM",
    "RCL", "RE", "REG", "REGN", "RF", "RJF", "RL", "RMD", "ROK", "ROL",
    "ROP", "ROST", "RSG",
    "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS",
    "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYY",
    "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
    "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR",
    "VTRS", "VZ",
    "WAB", "WAT", "WBA", "WBD", "WDC", "WELL", "WFC", "WM", "WMB", "WMT",
    "WRB", "WRK", "WST", "WTW", "WY",
    "XEL", "XOM", "XYL",
    "YUM",
    "ZBH", "ZBRA", "ZTS",
]

_NASDAQ100_FALLBACK: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AKAM", "AMAT",
    "AMD", "AMGN", "AMZN", "ANSS", "ARM", "ASML", "AVGO", "AZN",
    "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH",
    "DASH", "DDOG", "DLTR",
    "EA", "EBAY",
    "FANG", "FAST", "FTNT",
    "GEHC", "GFS", "GOOGL",
    "HON",
    "IDXX", "ILMN", "INTC", "INTU", "ISRG",
    "KDP", "KLAC", "KHC",
    "LRCX", "LULU",
    "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRNA", "MRVL",
    "MSFT", "MU",
    "NDAQ", "NFLX", "NVDA",
    "ODFL", "ON", "ORCL",
    "PANW", "PAYX", "PCAR", "PDD", "PYPL",
    "QCOM", "QRVO",
    "REGN", "ROST",
    "SBUX", "SMCI", "SNPS",
    "TEAM", "TT", "TTWO", "TSLA", "TMUS", "TXN",
    "VRSK", "VRTX",
    "WBD", "WDC",
    "XEL",
    "ZS",
]

_DOW30_FALLBACK: list[str] = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

# Russell 1000 names not already covered by the S&P 500 / NASDAQ-100 fallbacks.
# The union is taken so duplicates are harmless; this list focuses on optionable
# mid-to-large cap names that round out the top-1000 US equity universe.
# Matches the index composition as of early 2026.
_RUSSELL1000_FALLBACK: list[str] = [
    # Technology / Software
    "BILL", "BRZE", "CDAY", "CVLT", "DOCS", "DT", "DUOL", "EXLS", "FRSH",
    "GLOB", "GWRE", "HUBS", "IBKR", "KVYO", "MANH", "NCNO", "NTNX",
    "OKTA", "PCTY", "PLTR", "QTWO", "RNG", "RPD", "S",
    "SMAR", "SNOW", "SQSP", "TOST", "TYL", "VRNS", "WDAY", "WEX", "ZI",
    # Semiconductors / Hardware
    "AEIS", "AMBA", "COHU", "CRUS", "DLB", "FORM", "IDCC", "LSCC", "MKSI",
    "POWI", "SITM", "SYNA",
    # Healthcare / Biotech
    "ACAD", "ALKS", "ARWR", "AXSM", "BHVN", "BMRN", "EXAS", "FOLD", "HALO",
    "HIMS", "IONS", "JAZZ", "KROS", "LGND", "LNTH", "MDGL", "NVCR", "NTRA",
    "PDCO", "PRGO", "RCKT", "RGEN", "RXRX", "SAGE", "SRPT", "SUPN", "TGTX",
    # Financial Services
    "AMG", "APO", "AXS", "CBSH", "CSWI", "EQH", "FCNCA", "FNB", "HLNE",
    "HLI", "JLL", "LPLA", "MCY", "MORN", "OZK", "PIPR", "SF", "SLM", "SSNC",
    "STEP", "VIRT",
    # Consumer Discretionary
    "BJ", "BOOT", "CAKE", "DKS", "ELF", "FIVE", "HBI", "MODG",
    "PLCE", "RH", "SFM", "SHAK", "WING",
    # Consumer Staples
    "COTY", "FRPT", "INGR", "LANC", "PFGC", "POST", "SPTN", "USFD",
    # Industrial
    "AIT", "AWI", "BWXT", "CACI", "ESAB", "FIX", "GXO", "HXL", "KTOS",
    "MMS", "MOOG", "OSK", "RRX", "SAIA", "TDW", "WERN", "XPO",
    # Energy
    "AM", "CHRD", "DINO", "HPK", "KNTK", "MTDR", "NOG", "OVV", "PR", "RRC",
    "SM", "SWN",
    # Real Estate
    "AIV", "BXP", "EGP", "EPRT", "IIPR", "IRT", "KRC", "LXP", "NHI",
    "NXST", "PSTL", "REXR", "RLJ", "SAFE", "SBRA", "SKT", "SUI", "WD",
    # Utilities / Infrastructure
    "CWEN", "NTGR", "OGS", "SWX",
    # Media / Telecom
    "ATUS", "GTN", "NYT", "PARA", "SIRI",
    # Materials
    "CBT", "CLF", "CMC", "HCC", "KOP", "MTRN", "WOR",
    # Miscellaneous mid-caps commonly screened for options
    "ACA", "ACLS", "ACM", "AGIO", "ALIT", "AMN", "APPF", "ASTE",
    "AVAV", "AX", "AZZ", "BCPC", "BHC", "BLD", "BLMN", "BMI", "BRC",
    "BRKR", "BRX", "BSY", "CBZ", "CIEN", "CLH", "CLVT", "CNX",
    "COLM", "CPF", "CRI", "CSL", "CUBI", "DNB", "DSGX", "EAT",
    "EHC", "EME", "ENS", "EPAC", "FAF", "FELE", "FLO", "FNF", "GBX",
    "GKOS", "GMED", "GTLS", "HAE", "HELE", "HGV", "HNI", "HUBG",
    "IBTX", "ITT", "JHG", "KAR", "KBH", "KNSL", "LB",
    "LCII", "LNC", "LPX", "LSTR", "MATX", "MEDP", "MIDD", "MLI", "MMSI",
    "MUSA", "NFG", "NLY", "NPK", "NVT", "OGN", "OMF", "ORCC",
    "PATK", "PBF", "PBI", "PCVX", "PFBC", "PNFP", "PPC",
    "PRG", "PSN", "PVH", "QLYS", "RES", "ROCK", "RPM", "RRR",
    "RXO", "SANM", "SCL", "SEM", "SFNC", "SNDR", "SNX",
    "SPNT", "SPSC", "SRCL", "SUM", "SXI", "TENB", "TKR", "TWI", "USPH",
    "VCEL", "VITL", "WKC", "WSBC", "ZWS",
]

# Minimum number of tickers we expect from Wikipedia for the scrape to be trusted
_SP500_MIN      = 400
_NASDAQ100_MIN  = 80
_DOW30_MIN      = 25
_RUSSELL1000_MIN = 200   # Wikipedia page may not list all 1000; accept partial scrape


def get_index_tickers(
    cache_ttl_hours: int = CACHE_TTL_HOURS,
    force_refresh: bool = False,
    log=None,
) -> list[str]:
    """
    Return the union of S&P 500, NASDAQ-100, Dow Jones 30, and Russell 1000
    tickers.

    Tickers are sourced from Wikipedia and cached to data/index_cache.json.
    On cache hit (same ``CACHE_VERSION``, within TTL) the file is returned
    immediately.  On miss, version mismatch, or ``force_refresh``, all four
    index pages are scraped and the result is written to cache.

    If Wikipedia scraping returns fewer tickers than expected for a given index
    (network error, page restructure, etc.) the built-in fallback list is used
    for that index so the scanner always gets a reasonable universe.

    Parameters
    ----------
    cache_ttl_hours : int
        How long the cached list is valid (default: 24 hours).
    force_refresh : bool
        Bypass cache and re-scrape (default: False).
    log : logging.Logger | None
        Optional logger; if None informational messages are printed.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of equity ticker symbols.
    """
    def _log(msg: str) -> None:
        if log:
            log.info(msg)
        else:
            print(msg)

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, encoding='utf-8') as fh:
                cache = json.load(fh)
            cache_ver = cache.get('cache_version', 1)
            if 'index_tickers' in cache and cache_ver == CACHE_VERSION:
                expires_at = (
                    datetime.fromisoformat(cache['cached_at'])
                    + timedelta(hours=cache_ttl_hours)
                )
                if datetime.now() < expires_at:
                    tickers = cache['index_tickers']
                    _log(
                        f"Index universe loaded from cache: {len(tickers)} tickers "
                        f"(expires {expires_at.strftime('%Y-%m-%d %H:%M')})"
                    )
                    return tickers
        except Exception:
            pass  # corrupt / missing cache — fall through to refresh

    # ── Scrape Wikipedia ─────────────────────────────────────────────────────
    _log("Fetching S&P 500 constituents from Wikipedia...")
    sp500 = _fetch_sp500(log)
    if len(sp500) < _SP500_MIN:
        _log(
            f"  S&P 500 scrape returned only {len(sp500)} tickers "
            f"(expected >= {_SP500_MIN}) — using fallback list."
        )
        sp500 = _SP500_FALLBACK
    else:
        _log(f"  S&P 500: {len(sp500)} tickers")

    time.sleep(0.5)   # be polite between Wikipedia requests

    _log("Fetching NASDAQ-100 constituents from Wikipedia...")
    ndx100 = _fetch_nasdaq100(log)
    if len(ndx100) < _NASDAQ100_MIN:
        _log(
            f"  NASDAQ-100 scrape returned only {len(ndx100)} tickers "
            f"(expected >= {_NASDAQ100_MIN}) — using fallback list."
        )
        ndx100 = _NASDAQ100_FALLBACK
    else:
        _log(f"  NASDAQ-100: {len(ndx100)} tickers")

    time.sleep(0.5)

    _log("Fetching Dow Jones 30 constituents from Wikipedia...")
    dow30 = _fetch_dow30(log)
    if len(dow30) < _DOW30_MIN:
        _log(
            f"  Dow 30 scrape returned only {len(dow30)} tickers "
            f"(expected >= {_DOW30_MIN}) — using fallback list."
        )
        dow30 = _DOW30_FALLBACK
    else:
        _log(f"  Dow 30: {len(dow30)} tickers")

    time.sleep(0.5)

    _log("Fetching Russell 1000 constituents from Wikipedia...")
    russell1000 = _fetch_russell1000(log)
    if len(russell1000) < _RUSSELL1000_MIN:
        _log(
            f"  Russell 1000 scrape returned only {len(russell1000)} tickers "
            f"(expected >= {_RUSSELL1000_MIN}) — using fallback list."
        )
        russell1000 = _RUSSELL1000_FALLBACK
    else:
        _log(f"  Russell 1000: {len(russell1000)} tickers")

    # ── Deduplicate and sort ─────────────────────────────────────────────────
    combined = sorted(set(sp500 + ndx100 + dow30 + russell1000))
    _log(
        f"Index universe: {len(combined)} unique tickers "
        f"(SP500={len(sp500)}, NDX100={len(ndx100)}, DOW30={len(dow30)}, "
        f"Russell1000={len(russell1000)})"
    )

    # ── Cache result ─────────────────────────────────────────────────────────
    os.makedirs('data', exist_ok=True)
    try:
        # Preserve any existing keys (e.g. 'tickers' from universe.py cache)
        existing: dict = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, encoding='utf-8') as fh:
                    existing = json.load(fh)
            except Exception:
                pass
        existing['cache_version'] = CACHE_VERSION
        existing['cached_at']     = datetime.now().isoformat()
        existing['index_tickers'] = combined
        with open(CACHE_FILE, 'w', encoding='utf-8') as fh:
            json.dump(existing, fh, indent=2)
        _log(f"Index universe cached to {CACHE_FILE}")
    except Exception as exc:
        _log(f"Warning: could not write cache — {exc}")

    return combined
