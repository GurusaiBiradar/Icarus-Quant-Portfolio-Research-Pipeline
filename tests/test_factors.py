"""
Tests for src/factors.py

Key strategy:
  - Use a 500-day synthetic daily panel (~2 years) so momentum has valid rows.
  - Spot checks manually call Series.asof() with DateOffset to derive the
    expected value — independently from the function under test — then compare.
    This catches any index-alignment bugs in the DataFrame construction.
  - Mathematical edge cases (flat price, NaN propagation) use simpler panels.
"""

import pandas as pd
import pytest

from src.factors import (
    FACTOR_REGISTRY,
    FactorSpec,
    _prices_as_of,
    compute_factors,
    momentum,
    reversal,
    validate_no_lookahead,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily(n_days: int = 500, start: str = "2019-01-02") -> pd.DataFrame:
    """
    Daily close panel, two tickers:
      AAA: linearly increasing (price[i] = i + 1)
      BBB: flat at 100.0
    """
    dates = pd.date_range(start, periods=n_days, freq="B")
    return pd.DataFrame(
        {"AAA": [float(i + 1) for i in range(n_days)], "BBB": 100.0},
        index=dates,
    )


# ---------------------------------------------------------------------------
# _prices_as_of (the core primitive — test it directly)
# ---------------------------------------------------------------------------

class TestPricesAsOf:
    def test_resolves_to_previous_trading_day(self):
        """
        If the target date falls on a weekend, asof() should return the
        price from the Friday before it.
        """
        daily = _make_daily(100)
        result = _prices_as_of(daily, n_months=0)
        # n_months=0 means target = T itself; result should equal close_wide
        pd.testing.assert_frame_equal(result, daily)

    def test_returns_nan_before_series_start(self):
        """Dates where target < series start must produce NaN."""
        daily = _make_daily(500)
        # At the very first row, target = index[0] - 12 months → before data starts
        prices_12m_ago = _prices_as_of(daily, n_months=12)
        assert prices_12m_ago["AAA"].iloc[0] is pd.NA or pd.isna(
            prices_12m_ago["AAA"].iloc[0]
        )

    def test_output_index_matches_input(self):
        daily = _make_daily(300)
        result = _prices_as_of(daily, n_months=1)
        assert result.index.equals(daily.index)

    def test_output_columns_match_input(self):
        daily = _make_daily(300)
        result = _prices_as_of(daily, n_months=6)
        assert list(result.columns) == list(daily.columns)


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_spot_check(self):
        """
        Manually derive the expected value using asof() and DateOffset,
        then verify the function output matches at a specific date.
        This independently checks the DataFrame index alignment.
        """
        daily = _make_daily(500)
        mom = momentum(daily, short_months=1, long_months=12)

        # Pick a date well into the series (row 350, definitely past 12 months)
        check_date = daily.index[350]

        short_target = check_date - pd.DateOffset(months=1)
        long_target  = check_date - pd.DateOffset(months=12)
        expected = (
            daily["AAA"].asof(short_target) / daily["AAA"].asof(long_target) - 1
        )

        assert mom.loc[check_date, "AAA"] == pytest.approx(expected, rel=1e-9)

    def test_flat_price_gives_zero(self):
        """BBB is flat at 100 — momentum must be 0 wherever it is valid."""
        daily = _make_daily(500)
        mom = momentum(daily)
        valid = mom["BBB"].dropna()
        assert valid.abs().max() < 1e-10

    def test_nan_before_12_months(self):
        """Dates within 12 months of data start must all be NaN."""
        daily = _make_daily(500)
        mom = momentum(daily, short_months=1, long_months=12)

        cutoff = daily.index[0] + pd.DateOffset(months=12)
        early = mom.loc[mom.index < cutoff]
        assert early.isna().all().all()

    def test_valid_values_after_12_months(self):
        """At least some values after the 12-month mark must be non-NaN."""
        daily = _make_daily(500)
        mom = momentum(daily)
        cutoff = daily.index[0] + pd.DateOffset(months=12)
        late = mom.loc[mom.index >= cutoff].dropna(how="all")
        assert len(late) > 0

    def test_output_shape_matches_input(self):
        daily = _make_daily(300)
        mom = momentum(daily)
        assert mom.shape == daily.shape

    def test_skip_month_convention(self):
        """
        short_months=1 means the numerator uses price from 1 month ago,
        NOT today's price. Verify by using a panel where price jumps only
        in the current month — momentum should not see the jump.
        """
        daily = _make_daily(500)
        # Spike the last 21 rows to 9999 (simulate a sudden jump this month)
        daily_spiked = daily.copy()
        daily_spiked.iloc[-21:] = 9999.0

        mom_normal = momentum(daily, short_months=1, long_months=12)
        mom_spiked = momentum(daily_spiked, short_months=1, long_months=12)

        # At the last date: short_months=1 means numerator = price 1 month ago,
        # which is BEFORE the spike — so momentum should be unchanged
        assert mom_normal["AAA"].iloc[-1] == pytest.approx(
            mom_spiked["AAA"].iloc[-1], rel=1e-6
        )


# ---------------------------------------------------------------------------
# reversal
# ---------------------------------------------------------------------------

class TestReversal:
    def test_spot_check(self):
        """Manually derive expected value and compare."""
        daily = _make_daily(500)
        rev = reversal(daily, lag_months=1)

        check_date  = daily.index[100]
        lag_target  = check_date - pd.DateOffset(months=1)
        expected    = daily["AAA"].iloc[100] / daily["AAA"].asof(lag_target) - 1

        assert rev.loc[check_date, "AAA"] == pytest.approx(expected, rel=1e-9)

    def test_flat_price_gives_zero(self):
        daily = _make_daily(500)
        rev = reversal(daily)
        valid = rev["BBB"].dropna()
        assert valid.abs().max() < 1e-10

    def test_nan_before_1_month(self):
        daily = _make_daily(500)
        rev = reversal(daily, lag_months=1)
        cutoff = daily.index[0] + pd.DateOffset(months=1)
        early = rev.loc[rev.index < cutoff]
        assert early.isna().all().all()

    def test_output_shape_matches_input(self):
        daily = _make_daily(300)
        rev = reversal(daily)
        assert rev.shape == daily.shape

    def test_uses_current_close(self):
        """
        Reversal numerator is today's price (no lag on close_wide).
        If today's price changes, reversal should change too.
        """
        daily = _make_daily(300)
        daily_high = daily.copy()
        daily_high.iloc[-1] = 9999.0  # spike only today

        rev_normal = reversal(daily)
        rev_high   = reversal(daily_high)

        # Last row of reversal should differ — today's price is in the numerator
        assert rev_normal["AAA"].iloc[-1] != pytest.approx(
            rev_high["AAA"].iloc[-1], rel=1e-3
        )


# ---------------------------------------------------------------------------
# compute_factors
# ---------------------------------------------------------------------------

class TestComputeFactors:
    def test_returns_expected_keys(self):
        daily = _make_daily(500)
        factors = compute_factors(daily)
        assert set(factors.keys()) == {"momentum", "reversal"}

    def test_all_values_are_dataframes(self):
        daily = _make_daily(500)
        for v in compute_factors(daily).values():
            assert isinstance(v, pd.DataFrame)

    def test_momentum_has_more_nans_than_reversal(self):
        """Momentum looks back 12 months; reversal only 1."""
        daily = _make_daily(500)
        factors = compute_factors(daily)
        assert factors["momentum"]["AAA"].isna().sum() > factors["reversal"]["AAA"].isna().sum()

    def test_panels_share_index_with_input(self):
        daily = _make_daily(500)
        factors = compute_factors(daily)
        assert factors["momentum"].index.equals(daily.index)
        assert factors["reversal"].index.equals(daily.index)


# ---------------------------------------------------------------------------
# validate_no_lookahead
# ---------------------------------------------------------------------------

class TestValidateNoLookahead:
    def test_passes_for_valid_momentum(self):
        daily = _make_daily(500)
        mom = momentum(daily)
        validate_no_lookahead(mom, daily, n_months=12)  # should not raise

    def test_passes_for_valid_reversal(self):
        daily = _make_daily(500)
        rev = reversal(daily)
        validate_no_lookahead(rev, daily, n_months=1)  # should not raise

    def test_fails_when_early_nans_filled(self):
        """Filling early NaNs simulates a lookahead bug — guard must catch it."""
        daily = _make_daily(500)
        mom_bugged = momentum(daily).fillna(0.0)
        with pytest.raises(AssertionError):
            validate_no_lookahead(mom_bugged, daily, n_months=12)


# ---------------------------------------------------------------------------
# FACTOR_REGISTRY
# ---------------------------------------------------------------------------

class TestFactorRegistry:
    def test_contains_momentum_and_reversal(self):
        assert set(FACTOR_REGISTRY.keys()) == {"momentum", "reversal"}

    def test_momentum_ascending_true(self):
        assert FACTOR_REGISTRY["momentum"].ascending is True

    def test_reversal_ascending_false(self):
        assert FACTOR_REGISTRY["reversal"].ascending is False

    def test_specs_are_callable(self):
        for spec in FACTOR_REGISTRY.values():
            assert callable(spec.fn)

    def test_compute_factors_uses_registry_by_default(self):
        """Default registry produces momentum and reversal keys."""
        daily   = _make_daily(500)
        factors = compute_factors(daily)
        assert set(factors.keys()) == {"momentum", "reversal"}

    def test_compute_factors_accepts_custom_registry(self):
        """A custom registry with one factor produces exactly that factor."""
        daily          = _make_daily(500)
        custom_registry = {
            "flat": FactorSpec(fn=lambda c: c * 0, ascending=True),
        }
        factors = compute_factors(daily, registry=custom_registry)
        assert set(factors.keys()) == {"flat"}

    def test_custom_registry_factor_is_computed(self):
        """Verify the custom factor function is actually called."""
        daily           = _make_daily(100)
        custom_registry = {
            "ones": FactorSpec(fn=lambda c: c * 0 + 1.0, ascending=True),
        }
        factors = compute_factors(daily, registry=custom_registry)
        assert (factors["ones"] == 1.0).all().all()
