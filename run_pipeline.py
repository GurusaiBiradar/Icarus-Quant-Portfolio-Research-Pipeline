"""
Full pipeline runner — downloads price data, runs both backtests, and saves
all precomputed Parquet outputs to data/ for the Streamlit dashboard.

Usage:
    conda activate icarus
    python run_pipeline.py

Close data is cached after the first run.  Set FORCE_REFRESH = True below to
re-download.  Re-running without FORCE_REFRESH skips the ~1-minute download.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.alpha_model import RankAverageModel, XGBoostAlphaModel
from src.backtest_loop import generate_rebalance_dates, run_backtest
from src.data_pipeline import build_pipeline, load_processed_panel
from src.evaluation import build_realized_returns, ic_series, summarize
from src.universe import UniverseConfig, build_universe

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FORCE_REFRESH = False   # set True to re-download price data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_or_build_close() -> pd.DataFrame:
    """Return the close panel, downloading via yfinance if not cached."""
    cache = DATA_DIR / "processed_close.parquet"
    if cache.exists() and not FORCE_REFRESH:
        print("  Loading cached close panel from data/processed_close.parquet")
        return load_processed_panel("close")

    print("  Downloading price data (~1 min) …")
    cfg = UniverseConfig(
        min_avg_dollar_volume=5_000_000,
        min_price=5.0,
        min_history_days=756,
        lookback_window=1260,
    )
    _, price_long, filter_report = build_universe(n_sample=50, config=cfg, period="5y")
    filter_report.to_parquet(DATA_DIR / "filter_report.parquet", index=False)

    panels = build_pipeline(price_long)   # also writes raw + processed Parquet
    print(f"  Universe: {panels['close'].shape[1]} tickers × "
          f"{panels['close'].shape[0]} trading days")
    return panels["close"]


def _run_model(close_wide: pd.DataFrame, model, label: str) -> dict:
    print(f"  Running backtest [{label}] …")
    t0 = time.time()
    result = run_backtest(
        close_wide,
        model,
        burn_in_months=13,
        max_weight=0.10,
        capture_scores=True,
    )
    elapsed = time.time() - t0
    n = len(result["portfolio_returns"])
    sr = result["portfolio_returns"].mean() / result["portfolio_returns"].std() * 12 ** 0.5
    print(f"    {n} periods  |  annualised Sharpe ≈ {sr:.2f}  |  {elapsed:.0f}s")

    result["weights"].to_parquet(DATA_DIR / f"backtest_weights_{label}.parquet")
    result["portfolio_returns"].to_frame().to_parquet(
        DATA_DIR / f"portfolio_returns_{label}.parquet"
    )
    if "scores" in result:
        result["scores"].to_parquet(DATA_DIR / f"backtest_scores_{label}.parquet")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== Icarus pipeline ===\n")

    # ── Stage 1: universe + data ─────────────────────────────────────────────
    print("[1/4] Universe & data")
    close_wide = _load_or_build_close()

    # ── Stage 2: realized returns (for IC) ───────────────────────────────────
    print("[2/4] Realized returns")
    all_rebal  = generate_rebalance_dates(close_wide)
    realized   = build_realized_returns(close_wide, all_rebal)
    realized.to_parquet(DATA_DIR / "realized_returns.parquet")
    print(f"  {len(realized)} rebalance periods")

    # ── Stage 3: backtests ───────────────────────────────────────────────────
    print("[3/4] Backtests")
    results = {
        "rank": _run_model(close_wide, RankAverageModel(),   "rank"),
        "xgb":  _run_model(close_wide, XGBoostAlphaModel(),  "xgb"),
    }

    # ── Stage 4: IC + summary metrics ────────────────────────────────────────
    print("[4/4] IC & metrics")
    metrics_rows: dict[str, dict] = {}

    for label, result in results.items():
        port_ret = result["portfolio_returns"]
        scores   = result.get("scores")

        # IC — align scores and realized returns on common dates
        ic: pd.Series | None = None
        if scores is not None:
            common = scores.index.intersection(realized.index)
            if len(common) > 0:
                ic = ic_series(scores.loc[common], realized.loc[common])
                ic.to_frame().to_parquet(DATA_DIR / f"ic_{label}.parquet")
                print(f"  [{label}] mean IC = {ic.mean():.4f}  |  "
                      f"IC IR = {ic.mean() / ic.std() * 12 ** 0.5:.2f}")

        metrics_rows[label] = summarize(
            port_ret,
            scores.loc[common] if (scores is not None and ic is not None) else None,
            realized.loc[common] if ic is not None else None,
        )

    metrics_df = pd.DataFrame(metrics_rows).T
    metrics_df.index.name = "model"
    metrics_df.to_parquet(DATA_DIR / "metrics.parquet")

    # ── XGBoost decision ─────────────────────────────────────────────────────
    if "mean_ic" in metrics_df.columns:
        rank_ic = metrics_df.loc["rank", "mean_ic"]
        xgb_ic  = metrics_df.loc["xgb",  "mean_ic"]
        winner  = "XGBoost" if xgb_ic > rank_ic else "Rank-Average (baseline)"
        print(f"\n  Alpha model decision: {winner} wins on IC "
              f"({rank_ic:.4f} vs {xgb_ic:.4f})")

    print("\n=== Done.  Launch dashboard with: streamlit run app.py ===\n")


if __name__ == "__main__":
    main()
