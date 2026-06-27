"""
Stage 3 — Alpha model: rank-average baseline + XGBoost.

AlphaModel protocol
-------------------
Both models expose the same two-method interface so the backtest loop
never needs to branch on model type:

    model.fit(X, y)                        # no-op for RankAverageModel
    scores = model.predict_scores(cross)   # dict[str, Series] → Series

To add a new model (e.g. LightGBM, ridge regression), implement these two
methods and pass the instance into the backtest loop — no other file changes.

Factor direction metadata
-------------------------
RankAverageModel reads ascending flags from factors.FACTOR_REGISTRY by
default.  Pass a custom ascending dict to override (useful for testing or
for running with a non-standard factor set).

Lookahead-bias rules:
  - build_training_dataset() uses only rebalance_dates[:-1]; the last
    period's forward return is not yet observed at rebalance time.
  - All train/test splits are chronological — never random.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# AlphaModel protocol — the shared interface every model must implement
# ---------------------------------------------------------------------------

@runtime_checkable
class AlphaModel(Protocol):
    """
    Structural protocol for alpha models.

    Any class with fit() and predict_scores() satisfies this protocol —
    no explicit subclassing required.  Add new models without touching
    the backtest loop.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train on (factor scores → forward return) pairs."""
        ...

    def predict_scores(
        self,
        factor_cross_section: dict[str, pd.Series],
    ) -> pd.Series:
        """
        Predict alpha scores for a cross-section of tickers.

        Args:
            factor_cross_section: {factor_name: Series(ticker → score at T)}

        Returns:
            Series(ticker → alpha score), higher = stronger buy signal.
        """
        ...


# ---------------------------------------------------------------------------
# Rank-average baseline
# ---------------------------------------------------------------------------

class RankAverageModel:
    """
    Cross-sectional rank-average baseline.  Requires no training.

    At each date T:
      1. Rank each factor cross-sectionally as percentiles in (0, 1].
         Direction follows the ascending flag per factor (from FACTOR_REGISTRY
         by default): momentum ascending, reversal descending.
      2. Average percentile ranks into a single combined score.
         Any ticker with NaN in any factor receives NaN in the combined score.

    fit() is a no-op — kept so RankAverageModel satisfies AlphaModel and
    the backtest loop can call fit() unconditionally on any model type.
    """

    def __init__(self, ascending: dict[str, bool] | None = None) -> None:
        """
        Args:
            ascending: {factor_name: bool} mapping — True means higher raw
                       score is a stronger buy signal.  Defaults to reading
                       from factors.FACTOR_REGISTRY; pass explicitly to use
                       a custom or partial factor set.
        """
        if ascending is None:
            from src.factors import FACTOR_REGISTRY
            ascending = {name: spec.ascending for name, spec in FACTOR_REGISTRY.items()}
        self._ascending = ascending

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: ARG002
        pass

    def predict_scores(
        self,
        factor_cross_section: dict[str, pd.Series],
    ) -> pd.Series:
        ranked: dict[str, pd.Series] = {}
        for name, scores in factor_cross_section.items():
            asc = self._ascending.get(name, True)
            ranked[name] = scores.rank(ascending=asc, pct=True, na_option="keep")

        combined = pd.DataFrame(ranked).mean(axis=1, skipna=False)
        combined.name = "rank_avg_score"
        return combined


# ---------------------------------------------------------------------------
# Training data builder (shared by all ML models)
# ---------------------------------------------------------------------------

def get_factor_cross_section(
    factor_panels: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
) -> dict[str, pd.Series]:
    """
    Extract factor scores for all tickers as of a given date.

    Slices each panel to rows ≤ as_of and takes the last row, preserving
    NaN values (unlike DataFrame.asof(), which skips NaN rows).

    Returns:
        {factor_name: Series(ticker → score)}
    """
    return {
        name: panel.loc[panel.index <= as_of].iloc[-1]
        for name, panel in factor_panels.items()
    }


def build_training_dataset(
    factor_panels: dict[str, pd.DataFrame],
    close_wide: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) training pairs for any ML alpha model.

    For each consecutive pair (t_i, t_{i+1}) in rebalance_dates:
      X: factor scores for every ticker at t_i
      y: forward 1-period return = close[t_{i+1}] / close[t_i] − 1

    The last rebalance date is excluded — its forward return is unobserved.
    Rows with NaN factor scores or NaN returns are dropped after stacking.

    Returns:
        X: DataFrame with MultiIndex (date, ticker), columns = factor names
        y: Series with same MultiIndex, values = forward returns
    """
    period_frames: list[pd.DataFrame] = []

    for i in range(len(rebalance_dates) - 1):
        t_now  = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]

        cross_section = get_factor_cross_section(factor_panels, t_now)
        frame = pd.DataFrame(cross_section)
        frame.index.name = "ticker"

        price_now  = close_wide.loc[close_wide.index <= t_now].iloc[-1]
        price_next = close_wide.loc[close_wide.index <= t_next].iloc[-1]
        frame["forward_return"] = price_next / price_now - 1

        frame["date"] = t_now
        period_frames.append(frame.reset_index().set_index(["date", "ticker"]))

    if not period_frames:
        cols      = list(factor_panels.keys())
        empty_idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        return (
            pd.DataFrame(columns=cols, index=empty_idx),
            pd.Series(name="forward_return", index=empty_idx, dtype=float),
        )

    all_data = pd.concat(period_frames).dropna()
    y = all_data.pop("forward_return")
    return all_data, y


# ---------------------------------------------------------------------------
# XGBoost alpha model
# ---------------------------------------------------------------------------

class XGBoostAlphaModel:
    """
    XGBoost regressor wrapped for cross-sectional alpha prediction.

    Satisfies AlphaModel: fit() trains on historical (X, y) from
    build_training_dataset(); predict_scores() accepts a factor cross-section
    dict with the same interface as RankAverageModel.

    Only kept in the final pipeline if it out-performs RankAverageModel on
    IC across multiple out-of-sample periods (evaluated in evaluation.py).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        import xgboost as xgb
        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            verbosity=0,
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostAlphaModel":
        """Train on (factor scores → forward return) pairs from build_training_dataset()."""
        self._model.fit(X.values, y.values)
        self._is_fitted = True
        return self

    def predict_scores(
        self,
        factor_cross_section: dict[str, pd.Series],
    ) -> pd.Series:
        """
        Predict alpha scores for a cross-section of tickers.

        Args:
            factor_cross_section: {factor_name: Series(ticker → score at T)}

        Returns:
            Series(ticker → predicted return), higher = stronger buy signal.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_scores().")
        X = pd.DataFrame(factor_cross_section)   # index = ticker
        preds = self._model.predict(X.values)
        return pd.Series(preds, index=X.index, name="xgb_score")
