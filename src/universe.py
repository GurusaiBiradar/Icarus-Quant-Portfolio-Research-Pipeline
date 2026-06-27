"""
Stage 1 — Universe construction.

Scrapes ~50 S&P 500 tickers from Wikipedia, downloads price history via
yfinance, and applies three liquidity/history filters to produce a clean
investment universe.

Known limitation: uses *current* S&P 500 membership applied to historical
prices, not true point-in-time membership → survivorship bias. Document in README.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Fallback used when Wikipedia is unreachable.
_FALLBACK_TICKERS: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY", "JPM",
    "V", "XOM", "UNH", "MA", "JNJ", "PG", "HD", "MRK", "AVGO", "CVX", "PEP",
    "ABBV", "KO", "COST", "ADBE", "WMT", "MCD", "CRM", "BAC", "TMO", "ACN",
]


@dataclass
class UniverseConfig:
    min_avg_dollar_volume: float = 5_000_000   # $5M/day ADV
    min_price: float = 5.0                     # $5 minimum average close
    min_history_days: int = 756                # ~3 years of trading days
    lookback_window: int = 1260                # ~5 years downloaded, filters on 3


def get_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 tickers from Wikipedia. Falls back to hardcoded list."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        # yfinance uses hyphens; Wikipedia uses dots (e.g. BRK.B → BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception:
        return list(_FALLBACK_TICKERS)


def fetch_price_history(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """
    Download OHLCV data for all tickers via yfinance.

    Returns a long-format DataFrame with columns:
        date, ticker, close, volume
    Only rows where both close and volume are present are kept.
    """
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns a MultiIndex (field, ticker) when len(tickers) > 1
    close = raw["Close"]
    volume = raw["Volume"]

    # Melt each wide panel to long format then merge
    close_long = close.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    volume_long = volume.reset_index().melt(id_vars="Date", var_name="ticker", value_name="volume")

    price_long = close_long.merge(volume_long, on=["Date", "ticker"])
    price_long = price_long.rename(columns={"Date": "date"})
    price_long = price_long.dropna(subset=["close", "volume"])
    price_long["date"] = pd.to_datetime(price_long["date"])

    return price_long.reset_index(drop=True)


def apply_universe_filters(
    price_long: pd.DataFrame,
    config: UniverseConfig,
) -> tuple[list[str], pd.DataFrame]:
    """
    Apply three filters to the long-format price DataFrame.

    Filters:
        1. Minimum average closing price ≥ config.min_price
        2. Minimum average dollar volume ≥ config.min_avg_dollar_volume
        3. Minimum trading history ≥ config.min_history_days

    Returns:
        passing_tickers: list of tickers that pass all three filters
        filter_report: DataFrame summarising each ticker's stats and pass/fail
    """
    df = price_long.copy()
    df["dollar_volume"] = df["close"] * df["volume"]

    stats = (
        df.groupby("ticker")
        .agg(
            avg_price=("close", "mean"),
            avg_dollar_volume=("dollar_volume", "mean"),
            n_days=("date", "count"),
        )
        .reset_index()
    )

    stats["pass_price"] = stats["avg_price"] >= config.min_price
    stats["pass_adv"] = stats["avg_dollar_volume"] >= config.min_avg_dollar_volume
    stats["pass_history"] = stats["n_days"] >= config.min_history_days
    stats["pass_all"] = stats["pass_price"] & stats["pass_adv"] & stats["pass_history"]

    passing_tickers = stats.loc[stats["pass_all"], "ticker"].tolist()
    return passing_tickers, stats


def build_universe(
    n_sample: int = 50,
    config: UniverseConfig | None = None,
    period: str = "5y",
    seed: int = 42,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """
    End-to-end orchestration: scrape → download → filter → sample.

    Returns:
        universe_tickers: final sampled list of passing tickers
        price_long: long-format price/volume DataFrame for universe tickers
        filter_report: per-ticker filter stats
    """
    if config is None:
        config = UniverseConfig()

    all_tickers = get_sp500_tickers()

    # Sample a larger pool to account for filter drop-off
    random.seed(seed)
    pool = random.sample(all_tickers, min(len(all_tickers), n_sample * 3))

    price_long = fetch_price_history(pool, period=period)
    passing_tickers, filter_report = apply_universe_filters(price_long, config)

    # Sample down to n_sample from passing tickers
    random.seed(seed)
    universe_tickers = random.sample(passing_tickers, min(len(passing_tickers), n_sample))

    # Keep only universe tickers in price_long
    price_long = price_long[price_long["ticker"].isin(universe_tickers)].reset_index(drop=True)

    return universe_tickers, price_long, filter_report
