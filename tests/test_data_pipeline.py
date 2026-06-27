"""
Tests for src/data_pipeline.py

All tests use synthetic data and a tmp_path fixture for Parquet I/O.
No network calls.
"""

import pandas as pd

from src.data_pipeline import (
    clean_price_long,
    load_processed_panel,
    load_raw,
    save_processed_panel,
    save_raw,
    to_wide_panel,
)


class TestCleanPriceLong:
    def test_removes_duplicate_date_ticker(self, sample_price_long):
        df = pd.concat([sample_price_long, sample_price_long.iloc[[0]]], ignore_index=True)
        cleaned = clean_price_long(df)
        assert cleaned.duplicated(subset=["date", "ticker"]).sum() == 0

    def test_removes_non_positive_close(self, sample_price_long):
        sample_price_long.loc[0, "close"] = 0.0
        sample_price_long.loc[1, "close"] = -5.0
        cleaned = clean_price_long(sample_price_long)
        assert (cleaned["close"] > 0).all()

    def test_removes_non_positive_volume(self, sample_price_long):
        sample_price_long.loc[0, "volume"] = 0
        cleaned = clean_price_long(sample_price_long)
        assert (cleaned["volume"] > 0).all()

    def test_clean_data_unchanged_length(self, sample_price_long):
        cleaned = clean_price_long(sample_price_long)
        assert len(cleaned) == len(sample_price_long)


class TestToWidePanel:
    def test_shape(self, sample_price_long):
        wide = to_wide_panel(sample_price_long, field="close")
        assert wide.shape == (5, 2)  # 5 dates × 2 tickers

    def test_columns_are_tickers(self, sample_price_long):
        wide = to_wide_panel(sample_price_long, field="close")
        assert set(wide.columns) == {"AAA", "BBB"}

    def test_index_is_sorted_by_date(self, sample_price_long):
        wide = to_wide_panel(sample_price_long, field="close")
        assert wide.index.is_monotonic_increasing

    def test_volume_pivot(self, sample_price_long):
        wide = to_wide_panel(sample_price_long, field="volume")
        assert (wide == 1_000_000).all().all()


class TestRoundTrip:
    """Verify save → load preserves shape, dtypes, and values."""

    def test_raw_round_trip(self, sample_price_long, tmp_path):
        path = tmp_path / "raw.parquet"
        save_raw(sample_price_long, path=path)
        loaded = load_raw(path=path)
        assert loaded.shape == sample_price_long.shape
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
        assert set(loaded.columns) == set(sample_price_long.columns)

    def test_panel_round_trip(self, sample_price_long, tmp_path):
        wide = to_wide_panel(sample_price_long, field="close")
        path = tmp_path / "close.parquet"
        save_processed_panel(wide, name="close", path=path)
        loaded = load_processed_panel(name="close", path=path)
        assert loaded.shape == wide.shape
        assert pd.api.types.is_datetime64_any_dtype(loaded.index)
        pd.testing.assert_frame_equal(loaded, wide)
