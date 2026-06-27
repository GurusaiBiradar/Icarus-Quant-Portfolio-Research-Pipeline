"""
Tests for src/risk_model.py

Strategy:
  - compute_log_returns: shape (n-1), spot-check value, NaN propagation
    (a NaN price day drops that row AND the following row from returns).
  - is_psd / check_psd: known PSD and non-PSD matrices.
  - ledoit_wolf_cov: shape, symmetry, PSD, index/column labels,
    scale_factor arithmetic, and a direct comparison against sklearn to
    ensure we're not accidentally wrapping it differently.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from src.risk_model import (
    check_psd,
    compute_log_returns,
    is_psd,
    ledoit_wolf_cov,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_close(n_days: int = 300, n_tickers: int = 4) -> pd.DataFrame:
    dates   = pd.date_range("2020-01-02", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rng     = np.random.default_rng(0)
    prices  = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_tickers)), axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_shape_one_row_shorter(self):
        close = _make_close(100)
        assert compute_log_returns(close).shape == (99, 4)

    def test_no_nan_in_output(self):
        close = _make_close(100)
        assert not compute_log_returns(close).isna().any().any()

    def test_columns_match_input(self):
        close = _make_close(50)
        assert list(compute_log_returns(close).columns) == list(close.columns)

    def test_index_starts_at_second_close_date(self):
        """First return date should be close.index[1], not close.index[0]."""
        close = _make_close(50)
        assert compute_log_returns(close).index[0] == close.index[1]

    def test_spot_check_value(self):
        """log(p1 / p0) — verify one cell independently."""
        close  = _make_close(50)
        ret    = compute_log_returns(close)
        ticker = close.columns[0]
        expected = np.log(close[ticker].iloc[1] / close[ticker].iloc[0])
        assert ret[ticker].iloc[0] == pytest.approx(expected, rel=1e-12)

    def test_nan_price_drops_two_rows(self):
        """
        A NaN price at day d makes two return rows uncomputable:
          - log(NaN / p[d-1]) = NaN  → return at close.index[d] dropped
          - log(p[d+1] / NaN) = NaN  → return at close.index[d+1] dropped
        """
        close = _make_close(50)
        nan_date      = close.index[10]
        next_date     = close.index[11]
        close.iloc[10] = np.nan

        ret = compute_log_returns(close)
        assert nan_date  not in ret.index
        assert next_date not in ret.index

    def test_values_are_log_returns_not_simple(self):
        """Confirm log-return formula, not pct_change."""
        close = _make_close(10)
        ret   = compute_log_returns(close)
        # For small returns, log(1+r) ≈ r, but they differ for large swings
        simple = close.pct_change().dropna()
        # log returns should be slightly smaller in magnitude for positive returns
        assert not ret.equals(simple)


# ---------------------------------------------------------------------------
# is_psd
# ---------------------------------------------------------------------------

class TestIsPsd:
    def test_identity_is_psd(self):
        assert is_psd(np.eye(4))

    def test_positive_diagonal_is_psd(self):
        assert is_psd(np.diag([0.5, 1.0, 2.0]))

    def test_negative_definite_is_not_psd(self):
        assert not is_psd(-np.eye(3))

    def test_near_zero_negative_eigenvalue_within_tol(self):
        """Eigenvalue of -1e-12 passes with default tol=1e-8."""
        M = np.eye(3).copy()
        M[0, 0] = -1e-12
        assert is_psd(M)

    def test_negative_eigenvalue_outside_tol_fails(self):
        M = np.eye(3).copy()
        M[0, 0] = -1e-4
        assert not is_psd(M)

    def test_zero_eigenvalue_is_psd(self):
        """Rank-deficient (but PSD) matrix — one zero eigenvalue."""
        M = np.array([[1.0, 1.0], [1.0, 1.0]])   # eigenvalues: 2, 0
        assert is_psd(M)


# ---------------------------------------------------------------------------
# check_psd
# ---------------------------------------------------------------------------

class TestCheckPsd:
    def test_passes_silently_for_identity(self):
        check_psd(pd.DataFrame(np.eye(3)))   # must not raise

    def test_raises_for_negative_definite(self):
        with pytest.raises(ValueError, match="PSD"):
            check_psd(pd.DataFrame(-np.eye(3)))

    def test_error_contains_eigenvalue_info(self):
        try:
            check_psd(pd.DataFrame(-np.eye(3)))
        except ValueError as exc:
            assert "eigenvalue" in str(exc).lower()


# ---------------------------------------------------------------------------
# ledoit_wolf_cov
# ---------------------------------------------------------------------------

class TestLedoitWolfCov:
    def test_output_shape(self):
        close = _make_close(300, n_tickers=4)
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        assert cov.shape == (4, 4)

    def test_index_and_columns_match_tickers(self):
        close = _make_close(300, n_tickers=4)
        ret   = compute_log_returns(close)
        cov   = ledoit_wolf_cov(ret)
        assert list(cov.index)   == list(close.columns)
        assert list(cov.columns) == list(close.columns)

    def test_is_symmetric(self):
        close = _make_close(300)
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        pd.testing.assert_frame_equal(cov, cov.T)

    def test_is_psd(self):
        close = _make_close(300)
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        assert is_psd(cov.values)

    def test_check_psd_passes(self):
        close = _make_close(300)
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        check_psd(cov)   # must not raise

    def test_diagonal_entries_are_positive(self):
        """Variances (diagonal elements) must be strictly positive."""
        close = _make_close(300)
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        assert (np.diag(cov.values) > 0).all()

    def test_scale_factor_scales_linearly(self):
        """scale_factor=2 should give exactly 2× the scale_factor=1 result."""
        close = _make_close(300)
        ret   = compute_log_returns(close)
        pd.testing.assert_frame_equal(
            ledoit_wolf_cov(ret, scale_factor=2.0),
            ledoit_wolf_cov(ret, scale_factor=1.0) * 2.0,
        )

    def test_default_scale_factor_is_21(self):
        close = _make_close(300)
        ret   = compute_log_returns(close)
        pd.testing.assert_frame_equal(
            ledoit_wolf_cov(ret),
            ledoit_wolf_cov(ret, scale_factor=21.0),
        )

    def test_matches_sklearn_directly(self):
        """Numbers must match sklearn LedoitWolf called with identical inputs."""
        close = _make_close(300)
        ret   = compute_log_returns(close)

        lw = LedoitWolf(assume_centered=False)
        lw.fit(ret.values)
        expected = pd.DataFrame(
            lw.covariance_ * 21.0,
            index=ret.columns,
            columns=ret.columns,
        )
        pd.testing.assert_frame_equal(ledoit_wolf_cov(ret), expected)

    def test_fewer_observations_still_psd(self):
        """Works and stays PSD even with a tight observation window."""
        close = _make_close(60, n_tickers=4)   # ~3 months of data
        cov   = ledoit_wolf_cov(compute_log_returns(close))
        assert is_psd(cov.values)
