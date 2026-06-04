from argparse import Namespace
from pathlib import Path

import pytest

from ml.datasets.build_intraday_risk_dataset import _providers_from_args
from ml.providers.massive import MassiveProvider
from ml.providers.parquet_minute import ParquetMinuteBarProvider


def test_providers_from_args_uses_massive(monkeypatch):
    provider = MassiveProvider(api_key="test")
    monkeypatch.setattr(MassiveProvider, "from_env", classmethod(lambda cls: provider))

    market_provider, option_provider = _providers_from_args(
        Namespace(
            provider="massive",
            stock_dataset_root=None,
            option_dataset_root=None,
        )
    )

    assert market_provider is provider
    assert option_provider is provider


def test_providers_from_args_uses_parquet_roots(tmp_path):
    stock_root = tmp_path / "stocks"
    option_root = tmp_path / "options"

    market_provider, option_provider = _providers_from_args(
        Namespace(
            provider="parquet",
            stock_dataset_root=str(stock_root),
            option_dataset_root=str(option_root),
        )
    )

    assert isinstance(market_provider, ParquetMinuteBarProvider)
    assert isinstance(option_provider, ParquetMinuteBarProvider)
    assert Path(market_provider.dataset_root) == stock_root
    assert Path(option_provider.dataset_root) == option_root


def test_providers_from_args_requires_roots_for_parquet():
    with pytest.raises(ValueError, match="stock-dataset-root and --option-dataset-root"):
        _providers_from_args(
            Namespace(
                provider="parquet",
                stock_dataset_root=None,
                option_dataset_root=None,
            )
        )
