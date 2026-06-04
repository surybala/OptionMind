import pandas as pd

from ml.datasets.ingest_massive_flatfiles import (
    SymbolFilters,
    _apply_symbol_filters,
    _is_missing_flatfile_error,
    _option_underlying_from_ticker,
)


class NoSuchKey(Exception):
    pass


def test_is_missing_flatfile_error_handles_nosuchkey_exception_name():
    assert _is_missing_flatfile_error(NoSuchKey())


def test_is_missing_flatfile_error_handles_botocore_style_response():
    exc = Exception("missing")
    exc.response = {"Error": {"Code": "NoSuchKey"}}

    assert _is_missing_flatfile_error(exc)


def test_is_missing_flatfile_error_ignores_other_errors():
    exc = Exception("forbidden")
    exc.response = {"Error": {"Code": "403"}}

    assert not _is_missing_flatfile_error(exc)


def test_option_underlying_from_ticker_extracts_variable_length_root():
    assert _option_underlying_from_ticker("O:SPY250117P00400000") == "SPY"
    assert _option_underlying_from_ticker("SPY250117P00400000") == "SPY"
    assert _option_underlying_from_ticker("TLT260620C00095000") == "TLT"
    assert _option_underlying_from_ticker("bad") is None


def test_apply_symbol_filters_matches_option_underlyings():
    frame = pd.DataFrame(
        [
            {"ticker": "SPY250117P00400000", "value": 1},
            {"ticker": "QQQ250117C00500000", "value": 2},
            {"ticker": "SPYG250117C00065000", "value": 3},
        ]
    )

    filtered = _apply_symbol_filters(
        frame,
        "options",
        SymbolFilters(underlyings={"SPY", "QQQ"}),
    )

    assert list(filtered["ticker"]) == [
        "SPY250117P00400000",
        "QQQ250117C00500000",
    ]


def test_apply_symbol_filters_unions_exact_tickers_and_underlyings():
    frame = pd.DataFrame(
        [
            {"ticker": "SPY250117P00400000", "value": 1},
            {"ticker": "QQQ250117C00500000", "value": 2},
            {"ticker": "TLT260620C00095000", "value": 3},
        ]
    )

    filtered = _apply_symbol_filters(
        frame,
        "options",
        SymbolFilters(
            exact_tickers={"TLT260620C00095000"},
            underlyings={"SPY"},
        ),
    )

    assert list(filtered["ticker"]) == [
        "SPY250117P00400000",
        "TLT260620C00095000",
    ]
