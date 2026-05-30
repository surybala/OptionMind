"""Shared OSI (Option Symbology Initiative) symbol parsing.

An OSI option symbol has the form: ROOT YYMMDD C|P STRIKE(8 digits, /1000).

    Example: AAPL240119C00200000
             ^^^^------^--------
             root  date C/P strike ($200.00)

This module provides a single canonical parser used by:
    - src/alpaca_data.py   (snapshot → row conversion)
    - src/model_scanner.py (contract construction from chain snapshots)
    - ml/providers/alpaca.py (contract normalisation)
"""
from __future__ import annotations

import re
from datetime import date
from typing import NamedTuple

_OSI_RE = re.compile(r"^([A-Z./]+)(\d{6})([CP])(\d{8})$")


class OsiFields(NamedTuple):
    """Parsed fields from an OSI option symbol."""

    underlying: str
    expiration: date
    option_type: str   # "call" or "put"
    strike: float


def parse_osi(symbol: str) -> OsiFields | None:
    """Parse an OSI option symbol into its constituent fields.

    Returns ``None`` if *symbol* does not match the expected format.

    >>> parse_osi("AAPL240119C00200000")
    OsiFields(underlying='AAPL', expiration=datetime.date(2024, 1, 19), option_type='call', strike=200.0)
    """
    match = _OSI_RE.match(symbol)
    if not match:
        return None

    date_str = match.group(2)  # e.g. '240119'
    yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
    year = 2000 + yy if yy < 70 else 1900 + yy

    return OsiFields(
        underlying=match.group(1),
        expiration=date(year, mm, dd),
        option_type="call" if match.group(3) == "C" else "put",
        strike=int(match.group(4)) / 1000.0,
    )
