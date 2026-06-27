"""
Stage 5 — Portfolio optimizer: mean-variance with cvxpy.

Solves a quadratic program to find the maximum mean-variance portfolio:

    maximize  w^T μ − risk_aversion × w^T Σ w
    subject to
      sum(w) = 1          (fully invested)
      w ≥ 0               (long-only, no shorting)
      w ≤ max_weight      (concentration cap, default 10%)

μ is the cross-sectional alpha score vector from the alpha model.
Σ is the Ledoit-Wolf monthly covariance matrix from the risk model.

Why cvxpy:
  Separates problem specification from solver selection. The QP is small
  (≤ 50 variables) so the default solver (CLARABEL / OSQP) handles it in
  milliseconds per rebalance period.

NaN handling:
  Tickers with NaN in μ (insufficient history for the full lookback) are
  excluded from the QP and receive weight 0. The remaining weights still
  sum to 1.0.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def optimize(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    max_weight: float = 0.10,
) -> pd.Series:
    """
    Mean-variance portfolio optimization.

    Args:
        mu:            Alpha score vector, Series(ticker → score).
                       Tickers with NaN are excluded and receive weight 0.
        cov:           Monthly covariance matrix (ticker × ticker), must be PSD.
                       Must contain all non-NaN tickers from mu.index.
        risk_aversion: Return / risk trade-off parameter.  Higher values tilt
                       weights more aggressively toward high-score stocks;
                       lower values produce more diversified weights.
        max_weight:    Upper bound on any single stock's weight (default 0.10).
                       Constraint is feasible when n_valid × max_weight ≥ 1.

    Returns:
        Series(ticker → weight) covering ALL tickers in mu.index.
        Valid tickers' weights sum to 1.0; NaN-excluded tickers get 0.

    Raises:
        ValueError: if all mu values are NaN, or if the solver cannot find
                    an optimal solution (infeasible or unbounded problem).
    """
    valid = mu.dropna()
    if valid.empty:
        raise ValueError("All tickers in mu have NaN scores — cannot optimize.")

    mu_vec  = valid.values.astype(float)
    cov_mat = cov.loc[valid.index, valid.index].values.astype(float)

    n = len(valid)
    w = cp.Variable(n)

    objective = cp.Maximize(
        mu_vec @ w - risk_aversion * cp.quad_form(w, cov_mat)
    )
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= max_weight,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise ValueError(
            f"Optimizer returned status '{prob.status}'. "
            "Check that the covariance matrix is PSD and that "
            f"n_valid ({n}) × max_weight ({max_weight}) ≥ 1."
        )

    weights = pd.Series(w.value, index=valid.index, name="weight")
    return weights.reindex(mu.index, fill_value=0.0)
