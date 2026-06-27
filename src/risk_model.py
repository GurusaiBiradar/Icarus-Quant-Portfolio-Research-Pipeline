"""
Stage 4 — Risk model: Ledoit-Wolf shrinkage covariance estimator.

Why Ledoit-Wolf (not sample covariance):
  The sample covariance matrix is notoriously noisy when n_assets is close
  to n_observations — with ~50 stocks and ~750 daily observations the
  condition number can be in the thousands, making the optimizer unstable.
  Ledoit-Wolf shrinks the sample covariance toward a scaled identity matrix
  using an analytically optimal shrinkage coefficient, reducing estimation
  error while guaranteeing a PSD result.

Frequency convention:
  We estimate on daily log returns (more observations → lower estimation
  error) and scale by scale_factor (default 21 ≈ trading days per month)
  to produce a monthly covariance consistent with the monthly rebalancing
  loop and the alpha model's monthly-horizon scores.

PSD guarantee:
  The cvxpy optimizer requires a PSD covariance matrix. Ledoit-Wolf always
  produces PSD matrices analytically, but floating-point edge cases can
  produce tiny negative eigenvalues. check_psd() is called explicitly after
  every estimation as a defensive guard.

Lookahead rule:
  At rebalance date T, pass only returns up to T. This is enforced by the
  backtest loop slicing close_wide before calling compute_log_returns().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def compute_log_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a close price panel.

    log_return[t] = log(close[t] / close[t-1])

    The first row (NaN from the one-day lag) is always dropped. Any day
    where at least one ticker has a NaN price is also dropped — after the
    universe filters these gaps are rare and typically indicate a genuine
    trading halt.

    Returns:
        DataFrame (date × ticker) with one fewer row than close_wide.
    """
    log_ret = np.log(close_wide / close_wide.shift(1))
    return log_ret.dropna(how="any")


def is_psd(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Return True if *matrix* is positive semi-definite within *tol*.

    Uses eigvalsh (symmetric eigenvalue solver) — faster and more stable
    than eig for real symmetric matrices.
    """
    return bool(np.linalg.eigvalsh(matrix).min() >= -tol)


def check_psd(cov: pd.DataFrame, tol: float = 1e-8) -> None:
    """
    Assert that *cov* is PSD within *tol*. Raises ValueError if not.

    Call this after every ledoit_wolf_cov() in the backtest loop.
    """
    min_eig = np.linalg.eigvalsh(cov.values).min()
    if min_eig < -tol:
        raise ValueError(
            f"Covariance matrix is not PSD: min eigenvalue = {min_eig:.6e} "
            f"(tolerance {tol:.0e})."
        )


def ledoit_wolf_cov(
    returns: pd.DataFrame,
    scale_factor: float = 21.0,
) -> pd.DataFrame:
    """
    Estimate the Ledoit-Wolf shrinkage covariance matrix.

    Args:
        returns:      Daily log-return panel (date × ticker), no NaN.
                      Typically the output of compute_log_returns() sliced
                      to the current rebalance window.
        scale_factor: Scale the raw daily covariance by this factor before
                      returning. Default 21 ≈ trading days per month,
                      converting to monthly units for the optimizer.
                      Pass 252 to annualize.

    Returns:
        DataFrame (ticker × ticker) of the scaled shrinkage covariance.
        Index and columns are the ticker labels from *returns*.
    """
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns.values)
    cov_matrix = lw.covariance_ * scale_factor
    return pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
