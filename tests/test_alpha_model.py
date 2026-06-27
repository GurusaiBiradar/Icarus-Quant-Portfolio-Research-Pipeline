"""
Tests for src/alpha_model.py

Strategy:
  - AlphaModel protocol: verify both models satisfy it via isinstance check.
  - RankAverageModel: known cross-sections with manually derived expected
    ranks; direction convention; NaN propagation; fit() no-op.
  - get_factor_cross_section: spot-check values; NaN-row preservation.
  - build_training_dataset: forward returns vs. manual computation; last
    date excluded; X/y share index; edge cases.
  - XGBoostAlphaModel: fit/predict cycle; unified predict_scores interface
    (accepts dict[str, Series] same as RankAverageModel); error before fit.
"""

import numpy as np
import pandas as pd
import pytest

from src.alpha_model import (
    AlphaModel,
    RankAverageModel,
    XGBoostAlphaModel,
    build_training_dataset,
    get_factor_cross_section,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factor_panels(n_days: int = 400, n_tickers: int = 3) -> dict[str, pd.DataFrame]:
    dates   = pd.date_range("2020-01-02", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rng     = np.random.default_rng(0)
    return {
        "momentum": pd.DataFrame(
            rng.normal(0, 0.1, (n_days, n_tickers)), index=dates, columns=tickers
        ),
        "reversal": pd.DataFrame(
            rng.normal(0, 0.05, (n_days, n_tickers)), index=dates, columns=tickers
        ),
    }


def _make_close_wide(n_days: int = 400, n_tickers: int = 3) -> pd.DataFrame:
    dates   = pd.date_range("2020-01-02", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rng     = np.random.default_rng(1)
    prices  = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, (n_days, n_tickers)), axis=0)
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _cross(momentum_vals: dict, reversal_vals: dict) -> dict[str, pd.Series]:
    return {
        "momentum": pd.Series(momentum_vals),
        "reversal": pd.Series(reversal_vals),
    }


# ---------------------------------------------------------------------------
# AlphaModel protocol
# ---------------------------------------------------------------------------

class TestAlphaModelProtocol:
    def test_rank_average_satisfies_protocol(self):
        assert isinstance(RankAverageModel(), AlphaModel)

    def test_xgboost_satisfies_protocol(self):
        assert isinstance(XGBoostAlphaModel(), AlphaModel)


# ---------------------------------------------------------------------------
# RankAverageModel
# ---------------------------------------------------------------------------

class TestRankAverageModel:
    def test_output_index_matches_tickers(self):
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.1, "B": 0.3, "C": 0.2}, {"A": -0.05, "B": 0.02, "C": -0.03})
        )
        assert set(result.index) == {"A", "B", "C"}

    def test_scores_in_unit_interval(self):
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.1, "B": 0.3, "C": 0.2}, {"A": -0.05, "B": 0.02, "C": -0.03})
        )
        assert result.between(0, 1).all()

    def test_momentum_direction(self):
        """Highest momentum wins when reversal is tied: A > C > B."""
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.5, "B": 0.1, "C": 0.2}, {"A": 0.0, "B": 0.0, "C": 0.0})
        )
        assert result["A"] > result["C"] > result["B"]

    def test_reversal_direction(self):
        """Most negative reversal wins when momentum is tied: A > C > B."""
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.0, "B": 0.0, "C": 0.0}, {"A": -0.1, "B": 0.05, "C": -0.02})
        )
        assert result["A"] > result["C"] > result["B"]

    def test_nan_propagates_to_combined(self):
        """NaN in any single factor → NaN combined score."""
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.1, "B": float("nan"), "C": 0.2}, {"A": -0.05, "B": -0.03, "C": 0.02})
        )
        assert pd.isna(result["B"])
        assert not pd.isna(result["A"])
        assert not pd.isna(result["C"])

    def test_uniform_scores_give_equal_rank(self):
        """All tickers with identical scores → identical combined score."""
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.1, "B": 0.1, "C": 0.1}, {"A": -0.05, "B": -0.05, "C": -0.05})
        )
        assert result.nunique() == 1

    def test_result_name(self):
        result = RankAverageModel().predict_scores(
            _cross({"A": 0.1, "B": 0.2}, {"A": -0.01, "B": 0.01})
        )
        assert result.name == "rank_avg_score"

    def test_custom_ascending_overrides_registry(self):
        """Passing ascending dict explicitly overrides FACTOR_REGISTRY."""
        # Flip reversal direction: ascending=True means positive reversal wins
        model  = RankAverageModel(ascending={"momentum": True, "reversal": True})
        result = model.predict_scores(
            _cross({"A": 0.0, "B": 0.0, "C": 0.0}, {"A": -0.1, "B": 0.05, "C": -0.02})
        )
        # With ascending=True, highest reversal (B at 0.05) should win
        assert result["B"] > result["C"] > result["A"]

    def test_fit_is_noop(self):
        """fit() must not raise and must not alter predict_scores output."""
        model  = RankAverageModel()
        cross  = _cross({"A": 0.1, "B": 0.3}, {"A": -0.05, "B": 0.02})
        before = model.predict_scores(cross).copy()
        model.fit(pd.DataFrame(), pd.Series(dtype=float))
        after  = model.predict_scores(cross)
        pd.testing.assert_series_equal(before, after)


# ---------------------------------------------------------------------------
# get_factor_cross_section
# ---------------------------------------------------------------------------

class TestGetFactorCrossSection:
    def test_returns_correct_keys(self):
        panels = _make_factor_panels()
        result = get_factor_cross_section(panels, panels["momentum"].index[100])
        assert set(result.keys()) == {"momentum", "reversal"}

    def test_output_index_is_tickers(self):
        panels = _make_factor_panels()
        result = get_factor_cross_section(panels, panels["momentum"].index[100])
        assert list(result["momentum"].index) == list(panels["momentum"].columns)

    def test_values_match_panel_row(self):
        panels = _make_factor_panels()
        date   = panels["momentum"].index[100]
        result = get_factor_cross_section(panels, date)
        pd.testing.assert_series_equal(result["momentum"], panels["momentum"].loc[date])

    def test_preserves_nan_row(self):
        """A fully-NaN row must not be skipped — asof()-style skipping would hide bugs."""
        panels = _make_factor_panels(50)
        panels["momentum"].iloc[10] = float("nan")
        result = get_factor_cross_section(panels, panels["momentum"].index[10])
        assert result["momentum"].isna().all()


# ---------------------------------------------------------------------------
# build_training_dataset
# ---------------------------------------------------------------------------

class TestBuildTrainingDataset:
    def _setup(self):
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        idx    = panels["momentum"].index
        rebal  = [idx[i] for i in [100, 120, 140, 160, 180]]
        return panels, close, rebal

    def test_last_rebalance_date_not_in_X(self):
        panels, close, rebal = self._setup()
        X, _ = build_training_dataset(panels, close, rebal)
        assert rebal[-1] not in X.index.get_level_values("date").unique()

    def test_all_but_last_date_in_X(self):
        panels, close, rebal = self._setup()
        X, _  = build_training_dataset(panels, close, rebal)
        dates = set(X.index.get_level_values("date").unique())
        for d in rebal[:-1]:
            assert d in dates

    def test_y_values_match_forward_returns(self):
        """Spot-check y against manually computed forward return at t0 → t1."""
        panels, close, rebal = self._setup()
        X, y = build_training_dataset(panels, close, rebal)
        t0, t1 = rebal[0], rebal[1]
        for ticker in close.columns:
            p0       = close.loc[close.index <= t0].iloc[-1][ticker]
            p1       = close.loc[close.index <= t1].iloc[-1][ticker]
            assert y.loc[(t0, ticker)] == pytest.approx(p1 / p0 - 1, rel=1e-9)

    def test_X_and_y_share_index(self):
        panels, close, rebal = self._setup()
        X, y = build_training_dataset(panels, close, rebal)
        assert X.index.equals(y.index)

    def test_X_columns_are_factor_names(self):
        panels, close, rebal = self._setup()
        X, _ = build_training_dataset(panels, close, rebal)
        assert set(X.columns) == {"momentum", "reversal"}

    def test_index_names(self):
        panels, close, rebal = self._setup()
        X, _ = build_training_dataset(panels, close, rebal)
        assert X.index.names == ["date", "ticker"]

    def test_two_dates_produces_data(self):
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        rebal  = [panels["momentum"].index[100], panels["momentum"].index[120]]
        X, y   = build_training_dataset(panels, close, rebal)
        assert len(X) > 0

    def test_one_date_returns_empty(self):
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        X, y   = build_training_dataset(panels, close, [panels["momentum"].index[100]])
        assert len(X) == 0 and len(y) == 0


# ---------------------------------------------------------------------------
# XGBoostAlphaModel
# ---------------------------------------------------------------------------

class TestXGBoostAlphaModel:
    def _fitted_model(self):
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        idx    = panels["momentum"].index
        rebal  = [idx[i] for i in range(50, 400, 20)]
        X, y   = build_training_dataset(panels, close, rebal)
        return XGBoostAlphaModel().fit(X, y), panels, idx

    def _cross_section_at(self, panels, idx, i):
        return get_factor_cross_section(panels, idx[i])

    def test_predict_scores_accepts_cross_section_dict(self):
        """predict_scores takes dict[str, Series] — same interface as RankAverageModel."""
        model, panels, idx = self._fitted_model()
        cross = self._cross_section_at(panels, idx, 300)
        preds = model.predict_scores(cross)
        assert isinstance(preds, pd.Series)

    def test_predict_index_is_tickers(self):
        model, panels, idx = self._fitted_model()
        cross = self._cross_section_at(panels, idx, 300)
        preds = model.predict_scores(cross)
        assert list(preds.index) == list(panels["momentum"].columns)

    def test_predict_length_matches_universe(self):
        model, panels, idx = self._fitted_model()
        cross = self._cross_section_at(panels, idx, 300)
        preds = model.predict_scores(cross)
        assert len(preds) == len(panels["momentum"].columns)

    def test_predict_series_name(self):
        model, panels, idx = self._fitted_model()
        cross = self._cross_section_at(panels, idx, 300)
        assert model.predict_scores(cross).name == "xgb_score"

    def test_raises_before_fit(self):
        model  = XGBoostAlphaModel()
        panels = _make_factor_panels(400)
        cross  = get_factor_cross_section(panels, panels["momentum"].index[300])
        with pytest.raises(RuntimeError, match="fit"):
            model.predict_scores(cross)

    def test_fit_returns_self(self):
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        idx    = panels["momentum"].index
        rebal  = [idx[i] for i in range(50, 400, 20)]
        X, y   = build_training_dataset(panels, close, rebal)
        model  = XGBoostAlphaModel()
        assert model.fit(X, y) is model

    def test_swappable_with_rank_average(self):
        """Both models accept the same cross-section dict — swap without changing caller."""
        panels = _make_factor_panels(400)
        close  = _make_close_wide(400)
        idx    = panels["momentum"].index
        rebal  = [idx[i] for i in range(50, 400, 20)]
        X, y   = build_training_dataset(panels, close, rebal)
        cross  = get_factor_cross_section(panels, idx[300])

        rank_model = RankAverageModel()
        xgb_model  = XGBoostAlphaModel().fit(X, y)

        for model in (rank_model, xgb_model):
            scores = model.predict_scores(cross)
            assert isinstance(scores, pd.Series)
            assert len(scores) == len(panels["momentum"].columns)
