"""
Streamlit dashboard for the Icarus quant portfolio research pipeline.

Reads precomputed Parquet files from data/ — no live API calls, no model
training.  Run `python run_pipeline.py` first to generate the data files.

Launch:
    conda activate icarus
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluation import (
    RISK_FREE_ANNUAL,
    annualized_return,
    annualized_volatility,
    cumulative_returns,
    ic_ir,
    max_drawdown,
    mean_ic,
    sharpe_ratio,
)

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Icarus — Quant Research Pipeline",
    page_icon="📈",
    layout="wide",
)

st.title("Icarus — Quant Portfolio Research Pipeline")
st.caption(
    f"Monthly rebalance  ·  Ledoit-Wolf risk model  ·  "
    f"12-1 momentum + 1-month reversal  ·  "
    f"Risk-free rate {RISK_FREE_ANNUAL:.0%} p.a.  ·  Sharpe annualised ×√12"
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(filename: str) -> pd.DataFrame | None:
    path = DATA_DIR / filename
    return pd.read_parquet(path) if path.exists() else None


def _load_series(filename: str, col: str) -> pd.Series | None:
    df = _load(filename)
    return df[col] if df is not None and col in df.columns else None


# Guard: require at least the rank-average results
if not (DATA_DIR / "portfolio_returns_rank.parquet").exists():
    st.warning(
        "⚠️  No precomputed data found in `data/`.  "
        "Run `python run_pipeline.py` first, then reload this page."
    )
    st.stop()

port_rank   = _load_series("portfolio_returns_rank.parquet", "portfolio_return")
port_xgb    = _load_series("portfolio_returns_xgb.parquet",  "portfolio_return")
w_rank      = _load("backtest_weights_rank.parquet")
w_xgb       = _load("backtest_weights_xgb.parquet")
ic_rank_df  = _load("ic_rank.parquet")
ic_xgb_df   = _load("ic_xgb.parquet")
ic_rank     = ic_rank_df["ic"] if ic_rank_df is not None else None
ic_xgb      = ic_xgb_df["ic"]  if ic_xgb_df  is not None else None

models = {"Rank-Average": port_rank}
if port_xgb is not None:
    models["XGBoost"] = port_xgb

# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

st.subheader("Performance Summary")

cols = st.columns(len(models) * 4)
for m_i, (label, ret) in enumerate(models.items()):
    b = m_i * 4
    sr   = sharpe_ratio(ret)
    mdd  = max_drawdown(ret)
    ar   = annualized_return(ret)
    avol = annualized_volatility(ret)

    with cols[b]:
        st.metric(f"**{label}** — Sharpe", f"{sr:.2f}")
    with cols[b + 1]:
        st.metric("Max Drawdown", f"{mdd:.1%}")
    with cols[b + 2]:
        st.metric("Ann. Return", f"{ar:.1%}")
    with cols[b + 3]:
        st.metric("Ann. Volatility", f"{avol:.1%}")

# ---------------------------------------------------------------------------
# Cumulative returns
# ---------------------------------------------------------------------------

st.subheader("Cumulative Returns")

cum_df = pd.DataFrame({"Rank-Average": cumulative_returns(port_rank) - 1})
if port_xgb is not None:
    cum_df["XGBoost"] = cumulative_returns(port_xgb) - 1

st.line_chart(cum_df)
st.caption("Cumulative simple return (start = 0%)")

# ---------------------------------------------------------------------------
# Portfolio weights over time
# ---------------------------------------------------------------------------

st.subheader("Portfolio Weights Over Time")

tab_rank, tab_xgb = st.tabs(["Rank-Average", "XGBoost"])

def _weights_chart(w: pd.DataFrame | None, label: str) -> None:
    if w is None:
        st.info(f"No {label} weights data found.")
        return
    # Show only tickers that ever received > 1% weight
    active = w.columns[(w > 0.01).any()]
    st.area_chart(w[active])
    st.caption(f"{len(active)} tickers shown (ever > 1% weight)")

with tab_rank:
    _weights_chart(w_rank, "Rank-Average")
with tab_xgb:
    _weights_chart(w_xgb, "XGBoost")

# ---------------------------------------------------------------------------
# IC analysis
# ---------------------------------------------------------------------------

st.subheader("IC Analysis — Alpha Signal Quality")

ic_data   = {k: v for k, v in [("Rank-Average", ic_rank), ("XGBoost", ic_xgb)] if v is not None}

if ic_data:
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        st.write("IC per Rebalance Period")
        ic_chart = pd.DataFrame(ic_data)
        st.bar_chart(ic_chart)

    with table_col:
        st.write("Summary")
        rows = {
            lbl: {
                "Mean IC": f"{mean_ic(ic):.4f}",
                "IC IR":   f"{ic_ir(ic):.2f}",
                "Periods": str(int(ic.notna().sum())),
            }
            for lbl, ic in ic_data.items()
        }
        st.table(pd.DataFrame(rows).T)
        st.caption(
            "IC = Spearman rank correlation between predicted scores and "
            "realised forward returns.  IC IR = Mean IC / Std IC × √12."
        )

    # XGBoost decision
    if ic_rank is not None and ic_xgb is not None:
        xgb_wins = mean_ic(ic_xgb) > mean_ic(ic_rank)
        msg = (
            "✅ XGBoost **outperforms** the rank-average baseline on IC — retained."
            if xgb_wins else
            "📊 XGBoost does **not** consistently outperform the rank-average baseline — "
            "baseline retained as the primary model."
        )
        st.info(msg)
else:
    st.info("No IC data found.  Ensure `run_pipeline.py` completed successfully.")

# ---------------------------------------------------------------------------
# Methodology & known limitations
# ---------------------------------------------------------------------------

with st.expander("Methodology & Known Limitations"):
    st.markdown(f"""
**Factors**
- 12-1 month momentum with skip-month convention (avoids short-term reversal noise).
- 1-month short-term reversal.
- P/E replaced with reversal: `yfinance` provides only *current* P/E, which would
  introduce future information into a historical backtest.

**Alpha model**
- Rank-average baseline: averages cross-sectional percentile ranks of both factors.
- XGBoost: trained on expanding historical data to predict cross-sectional forward
  returns. Retained only if it consistently out-performs the baseline on IC across
  multiple out-of-sample periods.

**Risk model**
- Ledoit-Wolf shrinkage covariance estimated from daily log returns, scaled ×21 to
  monthly units.  Ledoit-Wolf is used because the sample covariance is unstable with
  ~50 assets and ~750 observations; Ledoit-Wolf shrinks toward the identity matrix
  with an analytically optimal coefficient and guarantees a PSD result.

**Optimiser**
- Mean-variance QP (long-only, max 10% per stock, weights sum to 1) solved via cvxpy.

**Backtest design**
- Monthly rebalance, expanding window (all data up to each rebalance date used for
  training and estimation).
- 13-month burn-in covers the 12-month momentum lookback + 1 safety month.

**Known limitations**
- **Survivorship bias**: current S&P 500 membership applied to historical prices —
  stocks delisted or removed from the index during the sample period are absent.
- Limited out-of-sample periods (~35 over 5 years) → substantial statistical
  uncertainty around all reported Sharpe ratios and IC estimates.
- Risk-free rate approximation: constant {RISK_FREE_ANNUAL:.0%} p.a. ({RISK_FREE_ANNUAL/12:.4f}/month).
""")
