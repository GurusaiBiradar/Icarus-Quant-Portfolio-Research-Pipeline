"""
Tests for src/backtest_loop.py

Strategy:
  - generate_rebalance_dates: last trading day per month, sorted, all in index.
  - run_backtest structural guarantees (checked on both model types):
      * output has "weights" and "portfolio_returns" keys
      * weights columns == close_wide columns
      * every weight row sums to ~1 (solver tolerance)
      * all weights in [0, max_weight + tol]
      * len(portfolio_returns) == len(weights) - 1
      * portfolio_returns index ⊂ weights index
      * no NaN in outputs
  - Edge case: too little data after burn-in raises ValueError.

We do not test for exact weight values — the backtest integrates multiple
already-tested modules; the structural guarantees are the right thing to check.

The synthetic close panel uses 20 tickers (20 × 0.10 = 2.0 ≥ 1.0 feasibility)
over 4 years so there are enough rebalance periods after the 13-month burn-in.
"""

import numpy as np
import pandas as pd
import pytest

from src.alpha_model import RankAverageModel, XGBoostAlphaModel
from src.backtest_loop import generate_rebalance_dates, run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close(n_years: int = 4, n_tickers: int = 20, seed: int = 0) -> pd.DataFrame:
    n_days  = int(n_years * 252)
    dates   = pd.date_range("2018-01-02", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rng     = np.random.default_rng(seed)
    prices  = 100 * np.cumprod(
        1 + rng.normal(0.0005, 0.01, (n_days, n_tickers)), axis=0
    )
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# generate_rebalance_dates
# ---------------------------------------------------------------------------

class TestGenerateRebalanceDates:
    def test_returns_list_of_timestamps(self):
        close = _make_close()
        dates = generate_rebalance_dates(close)
        assert isinstance(dates, list)
        assert all(isinstance(d, pd.Timestamp) for d in dates)

    def test_all_dates_are_in_close_index(self):
        close = _make_close()
        for d in generate_rebalance_dates(close):
            assert d in close.index

    def test_dates_are_sorted(self):
        close = _make_close()
        dates = generate_rebalance_dates(close)
        assert dates == sorted(dates)

    def test_one_date_per_calendar_month(self):
        """Consecutive dates must fall in different (year, month) pairs."""
        close = _make_close()
        dates = generate_rebalance_dates(close)
        periods = [d.to_period("M") for d in dates]
        assert len(periods) == len(set(periods))

    def test_each_date_is_last_trading_day_of_its_month(self):
        """No date in close.index has the same month but a later day."""
        close = _make_close()
        for rebal in generate_rebalance_dates(close):
            same_month = close.index[
                (close.index.year == rebal.year)
                & (close.index.month == rebal.month)
            ]
            assert rebal == same_month.max()

    def test_count_approximately_matches_months(self):
        """4 years ≈ 48 months of data → ~48 rebalance dates."""
        close = _make_close(n_years=4)
        dates = generate_rebalance_dates(close)
        assert 45 <= len(dates) <= 50


# ---------------------------------------------------------------------------
# run_backtest — shared structural checks
# ---------------------------------------------------------------------------

def _run(model, n_tickers=20, max_weight=0.10):
    close = _make_close(n_tickers=n_tickers)
    return run_backtest(close, model, burn_in_months=13,
                        max_weight=max_weight), close


class TestRunBacktestStructural:
    """Structural guarantees that must hold for any valid AlphaModel."""

    @pytest.fixture(params=["rank", "xgb"])
    def result_close(self, request):
        model = RankAverageModel() if request.param == "rank" else XGBoostAlphaModel()
        result, close = _run(model)
        return result, close

    def test_output_keys(self, result_close):
        result, _ = result_close
        assert set(result.keys()) == {"weights", "portfolio_returns"}

    def test_weights_columns_match_universe(self, result_close):
        result, close = result_close
        assert list(result["weights"].columns) == list(close.columns)

    def test_weights_sum_to_one_per_row(self, result_close):
        result, _ = result_close
        row_sums = result["weights"].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-3

    def test_weights_non_negative(self, result_close):
        result, _ = result_close
        assert (result["weights"] >= -1e-4).all().all()

    def test_weights_within_max_weight(self, result_close):
        result, _ = result_close
        assert (result["weights"] <= 0.10 + 1e-4).all().all()

    def test_portfolio_returns_length(self, result_close):
        result, _ = result_close
        assert len(result["portfolio_returns"]) == len(result["weights"]) - 1

    def test_portfolio_returns_index_subset_of_weights_index(self, result_close):
        result, _ = result_close
        assert result["portfolio_returns"].index.isin(result["weights"].index).all()

    def test_no_nan_in_weights(self, result_close):
        result, _ = result_close
        assert not result["weights"].isna().any().any()

    def test_no_nan_in_portfolio_returns(self, result_close):
        result, _ = result_close
        assert not result["portfolio_returns"].isna().any()

    def test_weights_index_name(self, result_close):
        result, _ = result_close
        assert result["weights"].index.name == "date"

    def test_portfolio_returns_name(self, result_close):
        result, _ = result_close
        assert result["portfolio_returns"].name == "portfolio_return"


# ---------------------------------------------------------------------------
# run_backtest — date ordering and lookahead sanity
# ---------------------------------------------------------------------------

class TestRunBacktestDates:
    def test_active_dates_are_after_burn_in(self):
        """All weight-record dates must be ≥ data_start + burn_in_months."""
        close  = _make_close()
        result = run_backtest(close, RankAverageModel(), burn_in_months=13)
        cutoff = close.index[0] + pd.DateOffset(months=13)
        assert (result["weights"].index >= cutoff).all()

    def test_weights_dates_are_sorted(self):
        close  = _make_close()
        result = run_backtest(close, RankAverageModel())
        idx    = result["weights"].index
        assert list(idx) == sorted(idx)

    def test_portfolio_returns_dates_are_sorted(self):
        close  = _make_close()
        result = run_backtest(close, RankAverageModel())
        idx    = result["portfolio_returns"].index
        assert list(idx) == sorted(idx)


# ---------------------------------------------------------------------------
# run_backtest — edge cases
# ---------------------------------------------------------------------------

class TestRunBacktestEdgeCases:
    def test_raises_if_too_little_data_after_burn_in(self):
        """Only 14 months of data with 13-month burn-in → 1 active date → ValueError."""
        close = _make_close(n_years=1)   # ~12 months — not enough after burn-in
        with pytest.raises(ValueError, match="burn-in"):
            run_backtest(close, RankAverageModel(), burn_in_months=13)

    def test_custom_factor_registry_is_used(self):
        """Passing a single-factor registry should run without error."""
        from src.factors import FactorSpec, momentum
        custom = {"momentum": FactorSpec(fn=momentum, ascending=True)}
        close  = _make_close()
        result = run_backtest(
            close, RankAverageModel(), factor_registry=custom
        )
        assert len(result["weights"]) > 0
