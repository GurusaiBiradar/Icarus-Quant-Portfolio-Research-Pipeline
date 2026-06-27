"""
Stage 1 — Data pipeline.

Cleans the raw long-format price DataFrame produced by universe.py, persists
it to Parquet, and provides wide-panel pivots (date × ticker) used by the
factor and risk modules downstream.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_price_long(price_long: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obviously bad rows:
      - Duplicate (date, ticker) pairs — keep first occurrence
      - Non-positive close prices
      - Non-positive volumes

    Does not forward-fill gaps; gaps are handled per-module where appropriate.
    """
    df = price_long.copy()
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first")
    df = df[df["close"] > 0]
    df = df[df["volume"] > 0]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence — raw long format
# ---------------------------------------------------------------------------

def save_raw(price_long: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist long-format DataFrame to Parquet. Returns the file path."""
    path = path or DATA_DIR / "raw_price_long.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    price_long.to_parquet(path, index=False)
    return path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load long-format DataFrame from Parquet."""
    path = path or DATA_DIR / "raw_price_long.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Wide-panel pivot
# ---------------------------------------------------------------------------

def to_wide_panel(price_long: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """
    Pivot long-format DataFrame to a wide date × ticker matrix.

    Args:
        price_long: DataFrame with columns [date, ticker, close, volume, ...]
        field: which column to use as values ("close" or "volume")

    Returns:
        DataFrame indexed by date, columns are tickers, sorted by date.
    """
    wide = price_long.pivot(index="date", columns="ticker", values=field)
    wide = wide.sort_index()
    wide.columns.name = None
    return wide


# ---------------------------------------------------------------------------
# Persistence — processed wide panels
# ---------------------------------------------------------------------------

def save_processed_panel(wide: pd.DataFrame, name: str, path: Path | None = None) -> Path:
    """
    Persist a wide date × ticker panel to Parquet.

    Args:
        name: logical name, e.g. "close" or "volume" — used in filename
    """
    path = path or DATA_DIR / f"processed_{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(path)
    return path


def load_processed_panel(name: str, path: Path | None = None) -> pd.DataFrame:
    """Load a wide date × ticker panel from Parquet."""
    path = path or DATA_DIR / f"processed_{name}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_pipeline(price_long: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Orchestrate the full pipeline: clean → save raw → pivot → save panels.

    Returns a dict with keys "close" and "volume" holding the wide panels.
    """
    clean = clean_price_long(price_long)
    save_raw(clean)

    close_wide = to_wide_panel(clean, field="close")
    volume_wide = to_wide_panel(clean, field="volume")

    save_processed_panel(close_wide, "close")
    save_processed_panel(volume_wide, "volume")

    return {"close": close_wide, "volume": volume_wide}
