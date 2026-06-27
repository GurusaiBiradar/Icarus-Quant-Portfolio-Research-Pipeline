"""
Shared pytest fixtures available to all test modules automatically.
"""

import pandas as pd
import pytest


@pytest.fixture
def sample_price_long() -> pd.DataFrame:
    """
    Minimal synthetic long-format price DataFrame with two tickers (AAA, BBB),
    5 business days, clean data. Used as a base for pipeline and factor tests.
    """
    dates = pd.date_range("2021-01-01", periods=5, freq="B")
    rows = []
    for ticker in ("AAA", "BBB"):
        for date in dates:
            rows.append({"date": date, "ticker": ticker, "close": 100.0, "volume": 1_000_000})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_price_long_large() -> pd.DataFrame:
    """
    Longer synthetic DataFrame — 3 tickers, 300 business days.
    Used by factor tests that need enough history for momentum windows.
    """
    dates = pd.date_range("2019-01-01", periods=300, freq="B")
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    for ticker in ("AAA", "BBB", "CCC"):
        prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
        for date, price in zip(dates, prices):
            rows.append({
                "date": date,
                "ticker": ticker,
                "close": float(price),
                "volume": 1_000_000,
            })
    return pd.DataFrame(rows)
