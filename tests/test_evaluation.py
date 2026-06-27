"""
Tests for src/evaluation.py

Strategy:
  - All formula tests verify against a manually derived expected value so the
    test is independent of the implementation (not testing the function against
    itself).
  - sharpe_ratio: formula correctness, direction, NaN for zero-std input.
  - max_drawdown: known-series spot checks; verified by tracing through
    the cumulative-returns / peak / drawdown algebra by hand.
  - ic_series / _spearman_ic: perfect correlation cases give ±1.0 exactly;
    random series verified against scipy.stats.spearmanr as ground truth.
  - build_realized_returns: values match close[T+1]/close[T]-1; last date excluded.
  - summarize: key presence, IC keys only when inputs are provided.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.evaluation import (
    RISK_FREE_ANNUAL,
    _spearman_ic,
    annualized_return,
    annualized_volatility,
    build_realized_returns,
    cumulative_returns,
    ic_ir,
    ic_series,
    max_drawdown,
    mean_ic,
    sharpe_ratio,
    summarize,
)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_formula_matches_manual(self):
        """Verify formula: (mean_excess / std_excess) × √12."""
        returns    = pd.Series([0.02, 0.01, 0.03, 0.00, -0.01, 0.02] * 4)
        rf_monthly = RISK_FREE_ANNUAL / 12
        excess     = returns - rf_monthly
        expected   = excess.mean() / excess.std() * np.sqrt(12)
        assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-9)

    def test_zero_std_returns_nan(self):
        """Constant returns → std = 0 → Sharpe undefined → NaN."""
        assert np.isnan(sharpe_ratio(pd.Series([0.01] * 24)))

    def test_positive_for_high_return(self):
        """Returns well above risk-free rate → Sharpe > 0."""
        returns = pd.Series([0.05, 0.04, 0.06, 0.05] * 6)
        assert sharpe_ratio(returns) > 0

    def test_negative_for_low_return(self):
        """Returns below risk-free rate → Sharpe < 0."""
        returns = pd.Series([0.001] * 24)   # 0.1% monthly << rf
        assert sharpe_ratio(returns) < 0

    def test_higher_mean_gives_higher_sharpe(self):
        """Same std, higher mean → higher Sharpe."""
        base = pd.Series([0.01, 0.03] * 12)
        high = base + 0.01
        assert sharpe_ratio(high) > sharpe_ratio(base)

    def test_custom_risk_free_rate(self):
        returns = pd.Series([0.02, 0.01, 0.03, 0.00] * 6)
        s_default = sharpe_ratio(returns, risk_free_annual=0.02)
        s_zero    = sharpe_ratio(returns, risk_free_annual=0.00)
        # Lower risk-free rate → higher excess returns → higher Sharpe
        assert s_zero > s_default


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_monotone_positive_returns_zero_drawdown(self):
        """Strictly increasing portfolio → no drawdown."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.015])
        assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown_two_periods(self):
        """
        returns = [+10%, −50%]
        cum     = [1.10, 0.55]
        peak    = [1.10, 1.10]
        dd      = [0, (0.55-1.10)/1.10] = [0, -0.5]
        max_dd  = -0.5
        """
        assert max_drawdown(pd.Series([0.10, -0.50])) == pytest.approx(-0.5, rel=1e-9)

    def test_recovers_to_peak_dd_is_worst_trough(self):
        """
        returns = [+20%, −10%, +25%, −20%]
        cum     = [1.20, 1.08, 1.35, 1.08]
        peak    = [1.20, 1.20, 1.35, 1.35]
        dd      = [0, -0.10, 0, -0.20]
        max_dd  = -0.20
        """
        returns = pd.Series([0.20, -0.10, 0.25, -0.20])
        assert max_drawdown(returns) == pytest.approx(-0.20, rel=1e-9)

    def test_returns_negative_or_zero(self):
        returns = pd.Series([0.1, -0.3, 0.1])
        assert max_drawdown(returns) < 0

    def test_single_positive_return(self):
        assert max_drawdown(pd.Series([0.05])) == pytest.approx(0.0, abs=1e-10)

    def test_single_negative_return(self):
        """Single negative return: cum < peak from the start → drawdown."""
        returns = pd.Series([-0.10])
        # cum = [0.90], peak = [0.90] (peak = cummax of cum = cum itself)
        # Actually for first period: cum[0] = 0.90, peak[0] = cummax = 0.90 → dd = 0
        # This is correct: you can't have a drawdown relative to a prior peak
        # if there is no prior peak.
        assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# cumulative_returns / annualized_return / annualized_volatility
# ---------------------------------------------------------------------------

class TestDerivedPortfolioMetrics:
    def test_cumulative_product(self):
        returns = pd.Series([0.10, -0.10])
        cum     = cumulative_returns(returns)
        assert cum.iloc[0] == pytest.approx(1.10, rel=1e-9)
        assert cum.iloc[1] == pytest.approx(1.10 * 0.90, rel=1e-9)

    def test_annualized_return_flat(self):
        """1% monthly for 12 months → CAGR = (1.01)^12 - 1 ≈ 12.68%."""
        returns = pd.Series([0.01] * 12)
        expected = 1.01 ** 12 - 1
        assert annualized_return(returns) == pytest.approx(expected, rel=1e-9)

    def test_annualized_volatility_formula(self):
        returns = pd.Series([0.01, 0.03, -0.01, 0.02] * 6)
        expected = returns.std() * np.sqrt(12)
        assert annualized_volatility(returns) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# _spearman_ic (private helper — tested directly for the core math)
# ---------------------------------------------------------------------------

class TestSpearmanIc:
    def test_perfect_positive_correlation(self):
        """Identical ordering of scores and returns → IC = 1.0."""
        scores  = pd.Series({"A": 0.8, "B": 0.5, "C": 0.2})
        returns = pd.Series({"A": 0.10, "B": 0.05, "C": 0.01})
        assert _spearman_ic(scores, returns) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative_correlation(self):
        """Exactly reversed ordering → IC = −1.0."""
        scores  = pd.Series({"A": 0.8, "B": 0.5, "C": 0.2})
        returns = pd.Series({"A": 0.01, "B": 0.05, "C": 0.10})
        assert _spearman_ic(scores, returns) == pytest.approx(-1.0, abs=1e-9)

    def test_matches_scipy_spearmanr(self):
        """Hand-rolled result must match scipy.stats.spearmanr."""
        rng     = np.random.default_rng(42)
        scores  = pd.Series(rng.normal(0, 1, 20))
        returns = pd.Series(rng.normal(0, 1, 20))
        expected = spearmanr(scores.values, returns.values).statistic
        assert _spearman_ic(scores, returns) == pytest.approx(expected, abs=1e-9)

    def test_nan_in_scores_excluded(self):
        """NaN tickers drop out; remaining must still give a valid IC."""
        scores  = pd.Series({"A": 0.8, "B": float("nan"), "C": 0.2})
        returns = pd.Series({"A": 0.10, "B": 0.05, "C": 0.01})
        # Only A and C remain — perfect positive correlation
        assert _spearman_ic(scores, returns) == pytest.approx(1.0, abs=1e-9)

    def test_fewer_than_two_valid_returns_nan(self):
        scores  = pd.Series({"A": float("nan"), "B": float("nan")})
        returns = pd.Series({"A": 0.10, "B": 0.05})
        assert np.isnan(_spearman_ic(scores, returns))


# ---------------------------------------------------------------------------
# ic_series
# ---------------------------------------------------------------------------

class TestIcSeries:
    def _make_inputs(self):
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        scores = pd.DataFrame(
            {"A": [0.8, 0.7, 0.9], "B": [0.5, 0.5, 0.5], "C": [0.2, 0.3, 0.1]},
            index=dates,
        )
        # Same ordering → IC = 1.0 at every date
        returns = pd.DataFrame(
            {"A": [0.10, 0.08, 0.12], "B": [0.05, 0.05, 0.05], "C": [0.01, 0.02, 0.00]},
            index=dates,
        )
        return scores, returns

    def test_output_is_series(self):
        scores, returns = self._make_inputs()
        assert isinstance(ic_series(scores, returns), pd.Series)

    def test_index_name(self):
        scores, returns = self._make_inputs()
        assert ic_series(scores, returns).index.name == "date"

    def test_series_name(self):
        scores, returns = self._make_inputs()
        assert ic_series(scores, returns).name == "ic"

    def test_perfect_correlation_gives_one(self):
        scores, returns = self._make_inputs()
        ic = ic_series(scores, returns)
        np.testing.assert_allclose(ic.values, 1.0, atol=1e-9)

    def test_only_common_dates_included(self):
        """Extra dates in either DataFrame are ignored."""
        dates_s = pd.date_range("2020-01-31", periods=3, freq="ME")
        dates_r = pd.date_range("2020-01-31", periods=2, freq="ME")
        scores  = pd.DataFrame({"A": [0.8, 0.5, 0.2]}, index=dates_s)
        returns = pd.DataFrame({"A": [0.1, 0.05]}, index=dates_r)   # 2 values, 2 dates
        ic = ic_series(scores, returns)
        assert len(ic) == 2


# ---------------------------------------------------------------------------
# mean_ic / ic_ir
# ---------------------------------------------------------------------------

class TestIcAggregates:
    def test_mean_ic_matches_pandas_mean(self):
        ic = pd.Series([0.1, 0.3, 0.2, 0.4])
        assert mean_ic(ic) == pytest.approx(ic.mean(), rel=1e-9)

    def test_mean_ic_ignores_nan(self):
        ic = pd.Series([0.1, float("nan"), 0.3])
        assert mean_ic(ic) == pytest.approx(0.2, rel=1e-9)

    def test_ic_ir_formula(self):
        """IC IR = mean / std × √12."""
        ic = pd.Series([0.10, 0.20, 0.15, 0.05] * 6)
        expected = ic.mean() / ic.std() * np.sqrt(12)
        assert ic_ir(ic) == pytest.approx(expected, rel=1e-9)

    def test_ic_ir_zero_std_returns_nan(self):
        # Use 0.0 (exactly representable) so fp noise doesn't mask the zero std
        ic = pd.Series([0.0] * 12)
        assert np.isnan(ic_ir(ic))


# ---------------------------------------------------------------------------
# build_realized_returns
# ---------------------------------------------------------------------------

class TestBuildRealizedReturns:
    def _make_close(self, n_days=100, n_tickers=4):
        dates   = pd.date_range("2020-01-02", periods=n_days, freq="B")
        tickers = [f"T{i}" for i in range(n_tickers)]
        rng     = np.random.default_rng(0)
        prices  = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_tickers)), axis=0)
        return pd.DataFrame(prices, index=dates, columns=tickers)

    def test_shape(self):
        """n rebalance dates → n-1 rows (last date has no forward return)."""
        close = self._make_close()
        rebal = close.index[[10, 30, 50, 70]].tolist()
        df = build_realized_returns(close, rebal)
        assert df.shape == (3, 4)

    def test_last_date_excluded(self):
        close = self._make_close()
        rebal = close.index[[10, 30, 50]].tolist()
        df = build_realized_returns(close, rebal)
        assert rebal[-1] not in df.index

    def test_values_match_manual_calculation(self):
        """Spot check: df.loc[t0] == close.loc[t1] / close.loc[t0] - 1."""
        close = self._make_close()
        t0, t1, t2 = close.index[10], close.index[30], close.index[50]
        df = build_realized_returns(close, [t0, t1, t2])
        expected = close.loc[t1] / close.loc[t0] - 1
        pd.testing.assert_series_equal(df.loc[t0], expected, check_names=False)

    def test_index_name(self):
        close = self._make_close()
        rebal = close.index[[10, 30, 50]].tolist()
        assert build_realized_returns(close, rebal).index.name == "date"

    def test_columns_match_tickers(self):
        close = self._make_close()
        rebal = close.index[[10, 30]].tolist()
        df = build_realized_returns(close, rebal)
        assert list(df.columns) == list(close.columns)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

class TestSummarize:
    def _returns(self):
        return pd.Series([0.01, 0.02, -0.01, 0.03, 0.01, -0.02] * 4)

    def test_required_keys_present(self):
        result = summarize(self._returns())
        for key in ("sharpe_ratio", "max_drawdown", "annualized_return",
                    "annualized_volatility", "n_periods"):
            assert key in result

    def test_ic_keys_absent_without_inputs(self):
        result = summarize(self._returns())
        assert "mean_ic" not in result
        assert "ic_ir"   not in result

    def test_ic_keys_present_with_inputs(self):
        dates   = pd.date_range("2020-01-31", periods=6, freq="ME")
        scores  = pd.DataFrame({"A": [0.8]*6, "B": [0.5]*6, "C": [0.2]*6}, index=dates)
        returns = pd.DataFrame({"A": [0.1]*6, "B": [0.05]*6, "C": [0.01]*6}, index=dates)
        result  = summarize(self._returns(), scores, returns)
        assert "mean_ic" in result
        assert "ic_ir"   in result

    def test_n_periods_matches_input_length(self):
        r = self._returns()
        assert summarize(r)["n_periods"] == float(len(r))

    def test_sharpe_matches_standalone(self):
        r = self._returns()
        assert summarize(r)["sharpe_ratio"] == pytest.approx(sharpe_ratio(r), rel=1e-9)

    def test_max_drawdown_matches_standalone(self):
        r = self._returns()
        assert summarize(r)["max_drawdown"] == pytest.approx(max_drawdown(r), rel=1e-9)
