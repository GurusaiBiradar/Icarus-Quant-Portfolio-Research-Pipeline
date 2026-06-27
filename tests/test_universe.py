"""
Tests for src/universe.py

All tests use synthetic data — no network calls, no yfinance dependency.
"""

import pandas as pd

from src.universe import UniverseConfig, apply_universe_filters


def _make_price_long(
    ticker: str,
    n_days: int,
    avg_close: float,
    avg_volume: float,
) -> pd.DataFrame:
    """Build a synthetic long-format DataFrame for a single ticker."""
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame({
        "date": dates,
        "ticker": ticker,
        "close": avg_close,
        "volume": avg_volume,
    })


CONFIG = UniverseConfig(
    min_avg_dollar_volume=5_000_000,
    min_price=5.0,
    min_history_days=100,
)


class TestApplyUniverseFilters:
    def _run(self, frames: list[pd.DataFrame]):
        price_long = pd.concat(frames, ignore_index=True)
        return apply_universe_filters(price_long, CONFIG)

    def test_clean_pass(self):
        """Stock meeting all three criteria should pass."""
        frames = [_make_price_long("PASS", n_days=200, avg_close=50.0, avg_volume=200_000)]
        passing, report = self._run(frames)
        assert "PASS" in passing

    def test_too_cheap(self):
        """Stock with avg price below $5 should fail."""
        frames = [_make_price_long("CHEAP", n_days=200, avg_close=4.0, avg_volume=200_000)]
        passing, report = self._run(frames)
        assert "CHEAP" not in passing
        assert not report.loc[report["ticker"] == "CHEAP", "pass_price"].values[0]

    def test_too_illiquid(self):
        """Stock with ADV below $5M should fail."""
        # close=50, volume=1 → dollar_volume = $50/day
        frames = [_make_price_long("ILLIQ", n_days=200, avg_close=50.0, avg_volume=1)]
        passing, report = self._run(frames)
        assert "ILLIQ" not in passing
        assert not report.loc[report["ticker"] == "ILLIQ", "pass_adv"].values[0]

    def test_too_short_history(self):
        """Stock with fewer days than min_history_days should fail."""
        frames = [_make_price_long("SHORT", n_days=50, avg_close=50.0, avg_volume=200_000)]
        passing, report = self._run(frames)
        assert "SHORT" not in passing
        assert not report.loc[report["ticker"] == "SHORT", "pass_history"].values[0]

    def test_borderline_pass(self):
        """Stock exactly at each threshold should pass (≥, not >)."""
        # dollar_volume = 5.0 * 1_000_000 = $5M exactly
        frames = [_make_price_long("BORDER", n_days=100, avg_close=5.0, avg_volume=1_000_000)]
        passing, report = self._run(frames)
        assert "BORDER" in passing

    def test_multiple_mixed(self):
        """Only the clean stock should pass when mixed with failures."""
        frames = [
            _make_price_long("GOOD",   n_days=200, avg_close=50.0, avg_volume=200_000),
            _make_price_long("CHEAP",  n_days=200, avg_close=2.0,  avg_volume=200_000),
            _make_price_long("ILLIQ",  n_days=200, avg_close=50.0, avg_volume=10),
            _make_price_long("SHORT",  n_days=10,  avg_close=50.0, avg_volume=200_000),
        ]
        passing, report = self._run(frames)
        assert passing == ["GOOD"]
        assert len(report) == 4

    def test_filter_report_columns(self):
        """Report must contain the expected columns."""
        frames = [_make_price_long("X", n_days=200, avg_close=50.0, avg_volume=200_000)]
        _, report = self._run(frames)
        for col in ("ticker", "avg_price", "avg_dollar_volume", "n_days",
                    "pass_price", "pass_adv", "pass_history", "pass_all"):
            assert col in report.columns
