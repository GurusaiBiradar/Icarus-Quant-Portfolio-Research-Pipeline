"""
Stage 6 — Backtest loop: monthly expanding-window rebalancing.

Ties together factors (Stage 2), alpha model (Stage 3), risk model (Stage 4),
and optimizer (Stage 5) into a chronologically valid backtest:

  For each rebalance date T after the burn-in period:
    1. Slice all data to [start, T] — expanding window, no future leakage.
    2. Compute factor panels on the slice.
    3. Fit the alpha model on all observable (factor, return) training pairs
       up to T — no-op for RankAverageModel, retrains XGBoost on growing data.
    4. Predict alpha scores for the cross-section at T.
    5. Estimate the Ledoit-Wolf covariance from log returns up to T.
    6. Solve the mean-variance QP for portfolio weights.
    7. Compute the simple portfolio return for the period [T, T_next].

Expanding vs rolling window:
  This implementation uses a growing training set. Rolling window (fixed
  lookback) is a noted future extension.

Lookahead-bias guarantees:
  - close_slice is always capped at T before any computation.
  - Factor panels are computed on close_slice — factor scores at T use only
    prices available at or before T (enforced by DateOffset lookbacks).
  - Training pairs passed to build_training_dataset use only prior rebalance
    dates; the last pair's forward return is close[T]/close[T_prev]-1, which
    is observable at T.
  - The covariance is estimated from returns up to T.
"""

from __future__ import annotations

import pandas as pd

from src.alpha_model import AlphaModel, build_training_dataset, get_factor_cross_section
from src.factors import FACTOR_REGISTRY, FactorSpec, compute_factors
from src.optimizer import optimize
from src.risk_model import check_psd, compute_log_returns, ledoit_wolf_cov


def generate_rebalance_dates(close_wide: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Return the last actual trading day of each calendar month in close_wide.

    Groups the date index by (year, month) period and takes the last date
    in each group — avoids synthetic month-end dates that may fall on
    weekends or holidays.
    """
    dates = close_wide.index.to_series()
    return dates.groupby(dates.dt.to_period("M")).last().tolist()


def run_backtest(
    close_wide: pd.DataFrame,
    model: AlphaModel,
    factor_registry: dict[str, FactorSpec] | None = None,
    burn_in_months: int = 13,
    risk_aversion: float = 1.0,
    max_weight: float = 0.10,
    capture_scores: bool = False,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    Monthly expanding-window backtest loop.

    Args:
        close_wide:      Date × ticker daily close panel.
        model:           Any AlphaModel instance.  fit() is called before each
                         predict_scores() — it is a no-op for RankAverageModel.
        factor_registry: Factor registry; defaults to FACTOR_REGISTRY.
        burn_in_months:  Months of data before the first live rebalance.
                         Must cover the longest factor lookback: ≥ 13 for
                         12-month momentum + 1 safety month.
        risk_aversion:   Passed to optimizer.optimize().
        max_weight:      Per-stock weight cap; passed to optimizer.optimize().
                         Feasibility requires n_valid_tickers × max_weight ≥ 1.
        capture_scores:  If True, also return a "scores" DataFrame
                         (date × ticker) of the raw alpha scores used at each
                         rebalance date.  Used by run_pipeline.py for IC
                         computation and model comparison.

    Returns:
        {
          "weights":           DataFrame (date × ticker) — optimized weights at
                               each rebalance date after burn-in.
          "portfolio_returns": Series (date) — simple portfolio return for each
                               period [T, T_next].  The last rebalance date has
                               no entry because its forward return is unknown.
          "scores":            DataFrame (date × ticker) — present only when
                               capture_scores=True.
        }

    Raises:
        ValueError: if fewer than 2 rebalance periods follow the burn-in.
    """
    if factor_registry is None:
        factor_registry = FACTOR_REGISTRY

    all_rebal   = generate_rebalance_dates(close_wide)
    cutoff      = close_wide.index[0] + pd.DateOffset(months=burn_in_months)
    active      = [d for d in all_rebal if d >= cutoff]

    if len(active) < 2:
        raise ValueError(
            f"Fewer than 2 rebalance periods after the {burn_in_months}-month "
            "burn-in.  Extend the data window or reduce burn_in_months."
        )

    weights_records: list[pd.Series]               = []
    return_records:  list[tuple[pd.Timestamp, float]] = []
    scores_records:  list[pd.Series]               = []

    for i, t_now in enumerate(active):
        # ── Step 1: expanding window ─────────────────────────────────────────
        close_slice = close_wide.loc[close_wide.index <= t_now]

        # ── Step 2: factor panels ────────────────────────────────────────────
        factor_panels = compute_factors(close_slice, registry=factor_registry)

        # ── Step 3: fit alpha model ──────────────────────────────────────────
        # Use ALL rebalance dates before t_now (even from burn-in; rows with
        # NaN factor scores are silently dropped inside build_training_dataset).
        prior = [d for d in all_rebal if d < t_now]
        if prior:
            X_tr, y_tr = build_training_dataset(
                factor_panels, close_slice, prior + [t_now]
            )
            if len(X_tr) > 0:
                model.fit(X_tr, y_tr)

        # ── Step 4: alpha scores ─────────────────────────────────────────────
        cross = get_factor_cross_section(factor_panels, t_now)
        mu    = model.predict_scores(cross)
        if capture_scores:
            scores_records.append(mu.rename(t_now))

        # ── Step 5: risk model ───────────────────────────────────────────────
        log_rets = compute_log_returns(close_slice)
        cov      = ledoit_wolf_cov(log_rets)
        check_psd(cov)

        # Align mu and cov to common tickers
        common  = mu.index.intersection(cov.index)
        mu_aln  = mu.reindex(common)
        cov_aln = cov.loc[common, common]

        # ── Step 6: optimize ─────────────────────────────────────────────────
        try:
            w = optimize(mu_aln, cov_aln,
                         risk_aversion=risk_aversion, max_weight=max_weight)
        except ValueError:
            # Equal-weight fallback when QP is infeasible (too few valid tickers)
            valid = mu_aln.dropna()
            w = pd.Series(1.0 / len(valid), index=valid.index, name="weight")
            w = w.reindex(mu_aln.index, fill_value=0.0)

        # Restore full-universe index (tickers absent from cov get weight 0)
        w = w.reindex(mu.index, fill_value=0.0)
        weights_records.append(w.rename(t_now))

        # ── Step 7: portfolio return for [t_now, t_next] ────────────────────
        if i < len(active) - 1:
            t_next     = active[i + 1]
            price_now  = close_wide.loc[close_wide.index <= t_now].iloc[-1]
            price_next = close_wide.loc[close_wide.index <= t_next].iloc[-1]
            port_ret   = (w * (price_next / price_now - 1)).sum()
            return_records.append((t_now, port_ret))

    weights_df = pd.DataFrame(weights_records).fillna(0.0)
    weights_df.index.name = "date"

    port_returns = pd.Series(
        {t: r for t, r in return_records},
        name="portfolio_return",
    )
    port_returns.index.name = "date"

    result: dict[str, pd.DataFrame | pd.Series] = {
        "weights": weights_df,
        "portfolio_returns": port_returns,
    }
    if capture_scores and scores_records:
        scores_df = pd.DataFrame(scores_records)
        scores_df.index.name = "date"
        result["scores"] = scores_df
    return result
