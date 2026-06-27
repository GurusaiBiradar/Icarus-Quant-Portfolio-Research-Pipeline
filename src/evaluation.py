"""
Stage 7 — Evaluation: Sharpe ratio, max drawdown, IC.

All metrics are hand-rolled from first principles to demonstrate understanding
of the underlying math — no pyfolio, no empyrical.

Conventions (stated explicitly for interview clarity):
  - All metrics assume MONTHLY returns.
  - Risk-free rate: constant 2% per annum → rf_monthly = 0.02 / 12.
    Rationale: simple, defensible approximation for a multi-year backtest.
  - Sharpe annualization: excess_monthly_sharpe × √12.
    Using √12 (not √252) because our return series is monthly.
  - IC: Spearman rank correlation between predicted alpha scores and
    realized forward returns, computed cross-sectionally at each rebalance
    date.  Spearman is used because scores are ranks, not cardinal forecasts.
    Hand-rolled as Pearson correlation of ranks (equivalent formulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RISK_FREE_ANNUAL: float = 0.02   # stated explicitly; see module docstring


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> float:
    """
    Annualized Sharpe ratio for a monthly return series.

    Sharpe = (mean(r − rf_m) / std(r − rf_m)) × √12

    where rf_m = risk_free_annual / 12.  Returns NaN if the excess-return
    standard deviation is zero (constant returns).
    """
    rf_monthly = risk_free_annual / 12
    excess = returns - rf_monthly
    std = excess.std()
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(12))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of a monthly return series.

    At each date t:
        drawdown(t) = (cumulative_peak(t) − cumulative_value(t))
                      / cumulative_peak(t)

    Returns the most negative drawdown value (e.g. −0.25 for a 25% drawdown).
    Returns 0.0 if the series never falls below its running peak.
    """
    cum   = (1 + returns).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    return float(dd.min())


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Cumulative wealth factor from a monthly return series.

    cumulative_returns[t] = product of (1 + r_i) for i ≤ t.
    Starting value is (1 + r_0); the series does not prepend a 1.0 base.
    """
    return (1 + returns).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """
    Compound annual growth rate from a monthly return series.

    CAGR = (total_wealth_factor) ^ (12 / n_periods) − 1
    """
    n = len(returns)
    total = float((1 + returns).prod())
    return total ** (12 / n) - 1


def annualized_volatility(returns: pd.Series) -> float:
    """Monthly return std scaled to annual: std × √12."""
    return float(returns.std() * np.sqrt(12))


# ---------------------------------------------------------------------------
# IC (Information Coefficient) — alpha model quality
# ---------------------------------------------------------------------------

def _spearman_ic(scores: pd.Series, returns: pd.Series) -> float:
    """
    Spearman rank IC between predicted scores and realized returns.

    Computed as Pearson correlation of ranks (exactly equivalent to the
    standard Spearman formula).  Tickers with NaN in either series are
    excluded.  Returns NaN if fewer than 2 valid observations remain.
    """
    valid = scores.dropna().index.intersection(returns.dropna().index)
    if len(valid) < 2:
        return float("nan")
    s_rank = scores.loc[valid].rank()
    r_rank = returns.loc[valid].rank()
    return float(np.corrcoef(s_rank.values, r_rank.values)[0, 1])


def ic_series(
    predicted_scores: pd.DataFrame,
    realized_returns: pd.DataFrame,
) -> pd.Series:
    """
    Spearman rank IC at each rebalance date.

    Args:
        predicted_scores: DataFrame (date × ticker) of alpha scores predicted
                          at each rebalance date.
        realized_returns: DataFrame (date × ticker) of forward simple returns
                          for each period.  Typically built with
                          build_realized_returns().

    Returns:
        Series(date → IC), where IC ∈ [−1, 1].
        Dates present in predicted_scores but not in realized_returns (or
        vice versa) are omitted.
    """
    common = predicted_scores.index.intersection(realized_returns.index)
    ics = {
        date: _spearman_ic(predicted_scores.loc[date], realized_returns.loc[date])
        for date in common
    }
    result = pd.Series(ics, name="ic")
    result.index.name = "date"
    return result


def mean_ic(ic: pd.Series) -> float:
    """Mean IC over all periods, ignoring NaN."""
    return float(ic.mean())


def ic_ir(ic: pd.Series) -> float:
    """
    IC Information Ratio: mean_IC / std_IC × √12.

    Measures signal consistency: how reliably the model ranks stocks
    correctly, not just on average but period by period.  An IC IR > 1
    is generally considered indicative of a robust alpha signal.
    Returns NaN if IC standard deviation is zero.
    """
    std = ic.std()
    if std < 1e-12:   # guard against floating-point near-zero std
        return float("nan")
    return float(ic.mean() / std * np.sqrt(12))


# ---------------------------------------------------------------------------
# Build IC inputs from close prices
# ---------------------------------------------------------------------------

def build_realized_returns(
    close_wide: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Build forward simple returns for IC computation.

    At rebalance date T_i:
        realized_return[ticker] = close[T_{i+1}] / close[T_i] − 1

    The last rebalance date has no entry (T+1 is unobserved).

    Args:
        close_wide:       Date × ticker daily close panel.
        rebalance_dates:  List of rebalance dates (sorted).

    Returns:
        DataFrame (date × ticker), indexed by T_i.
    """
    records = []
    for i in range(len(rebalance_dates) - 1):
        t_now  = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]
        price_now  = close_wide.loc[close_wide.index <= t_now].iloc[-1]
        price_next = close_wide.loc[close_wide.index <= t_next].iloc[-1]
        fwd = (price_next / price_now - 1).rename(t_now)
        records.append(fwd)

    df = pd.DataFrame(records)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(
    portfolio_returns: pd.Series,
    predicted_scores: pd.DataFrame | None = None,
    realized_returns: pd.DataFrame | None = None,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> dict[str, float]:
    """
    Compute all evaluation metrics and return as a flat dict.

    IC metrics (mean_ic, ic_ir) are included only when both
    predicted_scores and realized_returns are provided.

    Args:
        portfolio_returns: Monthly portfolio return series from backtest_loop.
        predicted_scores:  (date × ticker) alpha scores — for IC computation.
        realized_returns:  (date × ticker) forward returns — for IC computation.
        risk_free_annual:  Annual risk-free rate (default 2%).

    Returns:
        Dict of metric name → float value.
    """
    result: dict[str, float] = {
        "sharpe_ratio":         sharpe_ratio(portfolio_returns, risk_free_annual),
        "max_drawdown":         max_drawdown(portfolio_returns),
        "annualized_return":    annualized_return(portfolio_returns),
        "annualized_volatility": annualized_volatility(portfolio_returns),
        "n_periods":            float(len(portfolio_returns)),
    }

    if predicted_scores is not None and realized_returns is not None:
        ic = ic_series(predicted_scores, realized_returns)
        result["mean_ic"] = mean_ic(ic)
        result["ic_ir"]   = ic_ir(ic)

    return result
