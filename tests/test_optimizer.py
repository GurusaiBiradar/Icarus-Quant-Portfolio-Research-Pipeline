"""
Tests for src/optimizer.py

Strategy:
  - Structural guarantees (sum-to-one, non-negative, max-weight) verified
    on random valid inputs — these must hold regardless of mu/cov values.
  - Known-solution tests use identity covariance + very low risk_aversion
    so the objective reduces to a simple LP whose solution is predictable.
  - NaN exclusion: verify excluded tickers get exactly 0 and the rest
    sum to 1.
  - Edge cases: all-NaN mu raises ValueError; infeasible constraint set
    (1 ticker, max_weight < 1) raises ValueError.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import optimize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eye_cov(tickers: list[str], var: float = 0.001) -> pd.DataFrame:
    """Diagonal covariance (no correlation) with equal variance."""
    n   = len(tickers)
    mat = np.eye(n) * var
    return pd.DataFrame(mat, index=tickers, columns=tickers)


def _mu(tickers: list[str], scores: list[float]) -> pd.Series:
    return pd.Series(dict(zip(tickers, scores)))


# ---------------------------------------------------------------------------
# Structural guarantees — must hold for any valid input
# ---------------------------------------------------------------------------

class TestStructuralGuarantees:
    def _random_valid_inputs(self, n: int = 20, seed: int = 0):
        rng     = np.random.default_rng(seed)
        tickers = [f"T{i}" for i in range(n)]
        mu      = pd.Series(rng.uniform(0, 1, n), index=tickers)
        # Build a valid PSD covariance via A @ A.T + diagonal
        A       = rng.normal(0, 0.01, (n, n))
        cov_arr = A @ A.T + np.eye(n) * 0.001
        cov     = pd.DataFrame(cov_arr, index=tickers, columns=tickers)
        return mu, cov

    def test_weights_sum_to_one(self):
        mu, cov = self._random_valid_inputs()
        w = optimize(mu, cov)
        assert sum(w) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        mu, cov = self._random_valid_inputs()
        w = optimize(mu, cov)
        assert (w >= -1e-4).all()   # solver (CLARABEL/OSQP) noise up to ~1e-5

    def test_no_weight_exceeds_max(self):
        mu, cov = self._random_valid_inputs()
        w = optimize(mu, cov)
        assert (w <= 0.10 + 1e-4).all()   # solver noise up to ~1e-5

    def test_output_index_matches_mu_index(self):
        mu, cov = self._random_valid_inputs()
        w = optimize(mu, cov)
        assert list(w.index) == list(mu.index)

    def test_output_series_name(self):
        mu, cov = self._random_valid_inputs()
        assert optimize(mu, cov).name == "weight"

    def test_custom_max_weight_respected(self):
        mu, cov = self._random_valid_inputs(n=10)
        w = optimize(mu, cov, max_weight=0.20)
        assert (w <= 0.20 + 1e-6).all()


# ---------------------------------------------------------------------------
# Known-solution tests
# ---------------------------------------------------------------------------

class TestKnownSolutions:
    def test_top_scorer_gets_max_weight_at_low_risk_aversion(self):
        """
        With near-zero risk aversion and uncorrelated assets, the optimizer
        reduces to a simple LP: maximize mu^T w.  The top scorer must be
        capped at max_weight.
        """
        tickers = ["A", "B", "C", "D", "E"]
        mu  = _mu(tickers, [1.0, 0.5, 0.3, 0.2, 0.1])
        cov = _eye_cov(tickers)
        w   = optimize(mu, cov, risk_aversion=1e-6, max_weight=0.30)
        assert w["A"] == pytest.approx(0.30, abs=1e-4)

    def test_lowest_scorer_gets_zero_at_low_risk_aversion(self):
        """The lowest-ranked ticker should receive near-zero weight."""
        tickers = ["A", "B", "C", "D", "E"]
        mu  = _mu(tickers, [1.0, 0.8, 0.6, 0.4, 0.0])
        cov = _eye_cov(tickers)
        w   = optimize(mu, cov, risk_aversion=1e-6, max_weight=0.30)
        assert w["E"] == pytest.approx(0.0, abs=1e-4)

    def test_equal_mu_equal_variance_gives_equal_weights(self):
        """
        Equal scores + identity covariance → optimizer minimizes variance only.
        Minimum variance of uncorrelated equal-variance assets = equal weights.
        """
        n       = 10
        tickers = [f"T{i}" for i in range(n)]
        mu      = _mu(tickers, [0.5] * n)
        cov     = _eye_cov(tickers)
        w       = optimize(mu, cov, max_weight=1.0)
        expected = 1.0 / n
        for weight in w:
            assert weight == pytest.approx(expected, abs=1e-4)

    def test_two_tickers_no_cap_low_risk_aversion(self):
        """2 tickers, no cap: all weight goes to higher scorer."""
        mu  = _mu(["A", "B"], [1.0, 0.0])
        cov = _eye_cov(["A", "B"])
        w   = optimize(mu, cov, risk_aversion=1e-6, max_weight=1.0)
        assert w["A"] == pytest.approx(1.0, abs=1e-4)
        assert w["B"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# NaN exclusion
# ---------------------------------------------------------------------------

class TestNaNExclusion:
    # With default max_weight=0.10 we need ≥10 valid tickers for feasibility
    # (10 × 0.10 = 1.0).  Use max_weight=0.6 so 2 valid tickers suffice.
    _MW = 0.6

    def _nan_inputs(self):
        mu  = pd.Series({"A": 0.8, "B": float("nan"), "C": 0.3})
        cov = pd.DataFrame(
            np.eye(3) * 0.001,
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        return mu, cov

    def test_nan_ticker_receives_zero_weight(self):
        mu, cov = self._nan_inputs()
        w = optimize(mu, cov, max_weight=self._MW)
        assert w["B"] == pytest.approx(0.0, abs=1e-8)

    def test_valid_tickers_sum_to_one_when_some_are_nan(self):
        mu, cov = self._nan_inputs()
        w = optimize(mu, cov, max_weight=self._MW)
        assert w["A"] + w["C"] == pytest.approx(1.0, abs=1e-4)

    def test_output_index_includes_nan_tickers(self):
        """NaN tickers must still appear in the output with weight 0."""
        mu, cov = self._nan_inputs()
        w = optimize(mu, cov, max_weight=self._MW)
        assert set(w.index) == {"A", "B", "C"}

    def test_all_nan_raises(self):
        mu  = pd.Series({"A": float("nan"), "B": float("nan")})
        cov = _eye_cov(["A", "B"])
        with pytest.raises(ValueError, match="NaN"):
            optimize(mu, cov)


# ---------------------------------------------------------------------------
# Infeasible / error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_infeasible_raises_value_error(self):
        """
        1 ticker requires w[0] = 1, but max_weight = 0.5 forbids it.
        The solver should report infeasible and we should raise ValueError.
        """
        mu  = pd.Series({"A": 1.0})
        cov = pd.DataFrame({"A": [0.001]}, index=["A"])
        with pytest.raises(ValueError):
            optimize(mu, cov, max_weight=0.5)

    def test_error_message_contains_status(self):
        mu  = pd.Series({"A": 1.0})
        cov = pd.DataFrame({"A": [0.001]}, index=["A"])
        try:
            optimize(mu, cov, max_weight=0.5)
        except ValueError as exc:
            assert "status" in str(exc).lower() or "infeasible" in str(exc).lower()
