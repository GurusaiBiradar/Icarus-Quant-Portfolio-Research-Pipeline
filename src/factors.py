"""
Stage 2 — Alpha signals: momentum and short-term reversal.

Both factors operate on a daily close panel (date × ticker) and use
pd.DateOffset + Series.asof() for all lookbacks — never positional shift.

Why DateOffset + asof() instead of shift(N):
  - shift(N) is position-based: if any rows are missing, it silently lands
    on the wrong date (e.g. shift(21) on a series with 3 missing days gives
    you a price from 24 actual trading days ago, not 21).
  - DateOffset(months=N) computes an exact calendar date (e.g. 2021-03-15
    → 2021-02-15), handling month-length differences automatically.
  - asof(target) resolves that target to the last valid trading day on or
    before it — so if the target falls on a weekend or holiday, you still
    get a real price. If target is before the series starts, you get NaN,
    which propagates correctly.

The factors return daily panels so the backtest loop can slice up to date T
and take the last row, giving each period's cross-sectional scores with zero
future leakage.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    """Metadata bundle for a single alpha factor.

    fn        — callable (close_wide → score panel); use functools.partial
                to bake in non-default parameters (e.g. different lookbacks).
    ascending — True  = higher raw score → stronger buy signal (momentum)
                False = lower  raw score → stronger buy signal (reversal)
    """
    fn: Callable[[pd.DataFrame], pd.DataFrame]
    ascending: bool


def _prices_as_of(close_wide: pd.DataFrame, n_months: int) -> pd.DataFrame:
    """
    For every date T in close_wide, return the last known close price
    on or before (T − n_months calendar months).

    Steps:
      1. target_dates = index − DateOffset(months=n_months)
         Each element is an exact calendar date, not a row offset.
      2. Series.asof(target_dates) resolves each target to the last row
         in the series whose index is ≤ that target. Returns NaN if the
         target precedes the start of the series.
      3. Reassign the original index so the result aligns with close_wide.
    """
    target_dates = close_wide.index - pd.DateOffset(months=n_months)
    return pd.DataFrame(
        {ticker: close_wide[ticker].asof(target_dates).values
         for ticker in close_wide.columns},
        index=close_wide.index,
    )


def momentum(
    close_wide: pd.DataFrame,
    short_months: int = 1,
    long_months: int = 12,
) -> pd.DataFrame:
    """
    12-1 month momentum factor.

    At date T:
        score = price(T − short_months) / price(T − long_months) − 1

    short_months=1 skips the most recent month (the standard "skip-month"
    convention — the last month's return is dominated by reversal noise
    that would cancel the trend signal).

    NaN: any date where T − long_months precedes the start of the series.
    """
    short_prices = _prices_as_of(close_wide, short_months)
    long_prices  = _prices_as_of(close_wide, long_months)
    return short_prices / long_prices - 1


def reversal(
    close_wide: pd.DataFrame,
    lag_months: int = 1,
) -> pd.DataFrame:
    """
    1-month short-term reversal factor.

    At date T:
        score = close[T] / price(T − lag_months) − 1

    Uses today's close (observable at end-of-day T) divided by the price
    from lag_months ago. No future data used.

    NaN: any date where T − lag_months precedes the start of the series.
    """
    past_prices = _prices_as_of(close_wide, lag_months)
    return close_wide / past_prices - 1


def compute_factors(
    close_wide: pd.DataFrame,
    registry: dict[str, FactorSpec] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all factors in *registry* on the full daily close panel.

    The backtest loop slices close_wide up to date T, calls this, and
    takes the last row of each panel — so factor scores at T only use
    data available at T.

    Args:
        close_wide: date × ticker close price panel.
        registry:   factor registry to use; defaults to FACTOR_REGISTRY.
                    Pass a custom dict to compute only a subset of factors
                    or to add new ones without touching this module.

    Returns:
        {factor_name: daily panel (date × ticker) of scores}
    """
    if registry is None:
        registry = FACTOR_REGISTRY
    return {name: spec.fn(close_wide) for name, spec in registry.items()}


def validate_no_lookahead(
    factor_wide: pd.DataFrame,
    close_wide: pd.DataFrame,
    n_months: int,
) -> None:
    """
    Guard: assert that factor scores are NaN for all dates within the first
    n_months of the close history.

    With DateOffset lookbacks, any date T where T − n_months precedes the
    start of close_wide will produce NaN via asof(). This check confirms
    that behaviour is intact — if someone breaks it (e.g. fills NaN with 0),
    this guard will catch it. Raises AssertionError if violated.

    Args:
        factor_wide: output of momentum() or reversal()
        close_wide:  the close panel used to compute the factor
        n_months:    lookback length in months (long_months for momentum,
                     lag_months for reversal)
    """
    start   = close_wide.index[0]
    cutoff  = start + pd.DateOffset(months=n_months)
    early   = factor_wide.loc[factor_wide.index < cutoff]

    assert early.isna().all().all(), (
        f"Factor has non-NaN values before {cutoff.date()} "
        f"({n_months} months from data start {start.date()}). "
        "Possible lookahead bias or broken NaN propagation."
    )


# ---------------------------------------------------------------------------
# Default factor registry
# ---------------------------------------------------------------------------
# Add new factors here — no other file needs to change.
# Use functools.partial to register variants with non-default parameters:
#   "momentum_6_1": FactorSpec(fn=partial(momentum, long_months=6), ascending=True)

FACTOR_REGISTRY: dict[str, FactorSpec] = {
    "momentum": FactorSpec(fn=momentum, ascending=True),
    "reversal": FactorSpec(fn=reversal, ascending=False),
}
