# Project Brief: Quant Portfolio Research Pipeline (v1)

## Context / who I am
- CS/Data Science master's student targeting quantitative portfolio management
  and asset management internships (research-track roles, e.g. equity factor
  research, multi-asset/tactical allocation).
- This is one of several portfolio projects. This document covers **only**
  the simplified end-to-end quant pipeline — proving I understand the full
  lifecycle: data → signal → risk → optimized portfolio → monitoring →
  evaluation.
- Nothing has been built yet. Starting from scratch.

## Goal
Build the **simplest credible version** of a full quantitative portfolio
research pipeline — real data, real math, every stage explainable in an
interview, nothing overengineered.

Deploy: GitHub repo (README + code + screenshots). Streamlit Cloud deployment
is a cheap, near-free add-on once `app.py` works locally, since the dashboard
only ever reads precomputed files (no live API calls, so nothing can break
mid-demo).

## Guiding principle
**One example of each idea, not every variation.** Two factors, not five.
One optimizer, not two. Hand-rolled simple metrics, not heavy external
libraries. Every stage should be ~30-100 lines of clear, well-tested code.

Flag real limitations honestly in the README rather than hiding them. This
is also a learning exercise, not just a CV deliverable — explain the *why*
behind each piece of code, not just the *what*.

---

## The pipeline, end to end

Universe → Alpha signals → Risk model → Portfolio optimization
→ **[rebalancing loop wraps stages 2-4]** → Execution (skipped)
→ Monitoring → Attribution/evaluation

The rebalancing loop is what turns one-shot stages into an actual backtest.
Without it, stage 6's "cumulative return over time" chart has nothing to
plot.

---

## Stage 1 — Universe
- ~50 stocks sampled from the S&P 500 (Wikipedia scrape, with a hardcoded
  `_FALLBACK_TICKERS` list of ~30 large caps if Wikipedia is unreachable)
- 3 filters:
  - min average price: $5
  - min average dollar volume (ADV): $5M/day
  - min history: **3–5 years of daily data** (raised from 2 years — see
    Stage 2/loop reasoning below)
- Tools: `yfinance`, `pandas`
- Known limitation (document, don't hide): uses **current** S&P 500
  membership applied to historical prices, not true point-in-time
  membership → **survivorship bias**

### Module: `src/universe.py`
- `UniverseConfig` dataclass: `min_avg_dollar_volume`, `min_price`,
  `min_history_days`, `lookback_window`
- `get_sp500_tickers()`: scrapes Wikipedia, falls back to hardcoded list
- `fetch_price_history(tickers, period)`: downloads via yfinance, returns
  long-format DataFrame (date, ticker, close, volume)
- `apply_universe_filters(price_long, config)`: applies the 3 filters,
  returns `(passing_tickers, filter_report)`
- `build_universe(n_sample, config)`: end-to-end orchestration

### Module: `src/data_pipeline.py`
- `clean_price_long(price_long)`: drops duplicates, drops non-positive
  prices/volumes
- `save_raw()` / `load_raw()`: Parquet persistence for long-format data
- `to_wide_panel(price_long, field)`: pivots to date × ticker matrix
- `save_processed_panel()` / `load_processed_panel()`: Parquet persistence
  for wide panels
- `build_pipeline(price_long)`: orchestrates clean → save → pivot → save

### Test/validate
- Filter logic against synthetic edge cases: too cheap, too illiquid, too
  short history, borderline pass, clean pass
- Round-trip save/load preserves shape, dtypes, and values

---

## Stage 2 — Alpha signals
- **Factor 1 — momentum**: 12-month trailing return, excluding the most
  recent month
- **Factor 2 — short-term reversal** (past 1-month return) — **this
  replaces a P/E-based "value" factor**. `yfinance .info` only returns a
  current snapshot, not a historical time series, so it can't give a
  point-in-time P/E for a past date — using it would leak future
  information into the backtest. Reversal uses the exact same price data
  as momentum, so it's automatically point-in-time safe.
- **Combination method — build and compare both:**
  1. A simple rank-average baseline (no ML)
  2. An XGBoost model trained to predict next-period return from the two
     factor scores — shallow trees (`max_depth=2` or `3`), strong
     regularization (`reg_lambda`, `min_child_weight` set high) to limit
     its capacity, since 2 factors on ~50 stocks gives a complex model
     very little room to find real signal before it starts fitting noise
  - **Decision rule**: only keep XGBoost in the final pipeline if it beats
    the rank-average baseline by a real, consistent margin across multiple
    out-of-sample periods. If it doesn't, the baseline wins — that's the
    honest, reportable finding, not a failure.
- **Lookahead-bias guard**: factor at time T must only use data available
  at time T, predicting the return from T to T+1
- Tools: `pandas`, `numpy`, `xgboost`, `scikit-learn`

### Module: `src/factors.py`
- Momentum and reversal calculation functions, operating on wide panels
- Explicit lookahead-bias guards (assert no future dates used)

### Module: `src/alpha_model.py`
- Rank-average baseline combiner
- XGBoost training/prediction with **chronological** train/test split
  (never random — random splits leak future information into training)

### Test/validate
- Spot-check momentum calculation against a manual hand calculation for
  one stock/date
- Verify train/test split is chronological, not random
- Verify no future data leaks into features
- Compare baseline vs. XGBoost out-of-sample IC across multiple periods,
  not just one split

---

## Stage 3 — Risk model
- One covariance matrix using **Ledoit-Wolf shrinkage**
  (`sklearn.covariance.LedoitWolf`)
- Recomputed at each rebalance, using only data available up to that point
- Skip VaR/CVaR for v1 — the covariance matrix alone is enough to feed the
  optimizer
- Tools: `scikit-learn`, `numpy`

### Module: `src/risk_model.py`
- Ledoit-Wolf covariance matrix calculation

### Test/validate
- Verify the covariance matrix is **positive semi-definite (PSD)** —
  required for the optimizer to run at all

---

## Stage 4 — Portfolio optimization
- **Mean-variance optimization** only (skip risk parity/HRP/Black-Litterman
  for v1 — note these as a "future extension" in the README)
- Constraints: **long-only** (no shorting), **max 10%** in any single stock
- Tools: `cvxpy`

### Module: `src/optimizer.py`
- cvxpy mean-variance optimization with long-only + max-position
  constraints

### Test/validate
- Verify output weights sum to 1.0
- Verify the max 10% constraint is respected

---

## The connecting piece — rebalancing loop (NEW, ties stages 2–4 together)

This is not a separate pipeline stage — it's the orchestration logic that
makes stages 2, 3, and 4 repeat through time instead of running once.
Without this, there's no multi-period performance to chart in stage 6, and
no multiple out-of-sample test points for stage 7.

**Monthly, expanding-window loop:**
1. At month *t*: train the alpha model using only data available up to *t*
2. Compute that month's factor scores and Ledoit-Wolf covariance matrix
   from data up to *t* only
3. Run the optimizer → get target weights for month *t*
4. Hold those weights for one month, record the realized return
5. Move to *t+1*, repeat

Use an **expanding window** (all data up to *t*) for simplicity — a rolling
window is a fine future extension, not needed for v1.

**Knock-on effects of adding this loop (already reflected above):**
- Data window raised to 3–5 years, since the budget has to cover: the
  12-month momentum lookback + an initial training window for the model +
  enough leftover monthly rebalances to call it a real backtest
- This loop is also what fixes the "single train/test split is fragile"
  concern — monthly rebalancing across several years naturally produces
  many sequential out-of-sample test points instead of just one

### Module: `src/backtest_loop.py`
- Orchestrates the monthly expanding-window loop across stages 2–4
- Persists per-period weights and realized returns for stages 6 and 7

### Test/validate
- Confirm each period's training data cutoff is respected (no leakage
  across the loop boundary)
- Confirm the number of realized rebalance periods is tracked and saved

---

## Stage 5 — Execution
- **Skipped entirely for v1.** One sentence in the README explaining this
  is out of scope (real execution/TCA is a different skill set, less
  relevant to the research-track roles this project targets)

---

## Stage 6 — Monitoring
- One Streamlit page:
  - A table of current portfolio weights
  - One performance chart: cumulative return of the optimized portfolio
    vs. an equal-weight benchmark
- Use Streamlit's built-in charting (`st.line_chart`) — skip Plotly for v1
- **Important**: dashboard reads from saved Parquet/CSV files (the
  precomputed backtest loop output), **not** live API calls — for
  reliability when shown to recruiters or in interviews
- Tools: `streamlit`

### Module: `app.py`
- Streamlit dashboard reading from pre-computed saved results

---

## Stage 7 — Attribution / evaluation
- Three hand-rolled metrics (write the formulas myself, don't import
  `pyfolio`):
  - **Sharpe ratio** — needs an explicit risk-free rate assumption (a
    constant approximation, e.g. 2%, is fine for v1 — just state it) and
    the correct annualization factor matched to return frequency: `×√12`
    for monthly rebalance returns (the likely case given the loop above),
    `×√252` only if working with daily portfolio returns. Pick one,
    document it.
  - **Max drawdown**
  - **Information Coefficient (IC)** — correlation between predicted
    signal and actual subsequent return
- **Report the number of independent rebalance periods explicitly** next
  to the Sharpe/IC numbers. With monthly rebalancing over a few years,
  that number is small, and performance metrics carry wide uncertainty as
  a result — say so honestly in the README rather than letting the numbers
  speak alone.
- Tools: `numpy`, `scipy.stats` (for the IC correlation calculation)

### Module: `src/evaluation.py`
- Sharpe ratio, max drawdown, Information Coefficient — hand-rolled

### Test/validate
- Verify Sharpe/drawdown formulas against a known textbook example or a
  simple synthetic series with a known answer

---

## What NOT to add (deliberately out of scope for v1)
- Sector/country constraints
- Multiple risk models or multiple optimizers
- Walk-forward hyperparameter re-tuning each period (fixed model,
  rolling/expanding data is enough)
- Risk parity, Black-Litterman, HRP (note as future extensions only)
- Live API calls in the dashboard

These would contradict the "one example of each idea" principle, which is
the strongest part of this project's scope discipline.

---

## Repo structure
```
.
├── src/
│   ├── universe.py
│   ├── data_pipeline.py
│   ├── factors.py
│   ├── alpha_model.py
│   ├── risk_model.py
│   ├── optimizer.py
│   ├── backtest_loop.py
│   └── evaluation.py
├── app.py
├── data/            (gitignored — Parquet outputs)
├── tests/
└── README.md
```

## Build order
1. `src/universe.py` + `src/data_pipeline.py`
2. `src/factors.py` — momentum + reversal, with lookahead-bias guards
3. `src/alpha_model.py` — rank-average baseline + XGBoost, chronological
   split
4. `src/risk_model.py` — Ledoit-Wolf covariance
5. `src/optimizer.py` — cvxpy mean-variance, long-only + max-position
6. `src/backtest_loop.py` — monthly expanding-window orchestration across
   stages 2–4
7. `src/evaluation.py` — Sharpe, max drawdown, IC
8. `app.py` — Streamlit dashboard reading precomputed results
9. `README.md` — known limitations, architecture rationale, how-to-run
10. Push to GitHub; optionally deploy `app.py` to Streamlit Cloud

## Tools needed
- Already familiar: `pandas`, `numpy`, `yfinance`, `pyarrow`
- New for this project: `xgboost`, `scikit-learn`, `cvxpy`, `scipy`,
  `streamlit`

## Working style for implementation
- Build incrementally — test each module before moving to the next
- Explain concepts in plain language alongside the code; this is a
  learning exercise as much as a deliverable
- Flag honest limitations rather than overselling, consistent with how the
  README is written throughout
- Validate logic with small/synthetic data first if real data access is
  uncertain in any given environment; real runs happen wherever live
  internet access to Yahoo Finance/Wikipedia is available

## README must include (don't skip)
- Survivorship bias limitation (Stage 1)
- Why P/E/value was dropped in favor of reversal (Stage 2)
- Why XGBoost was kept or dropped, with the comparison numbers (Stage 2)
- Why execution is out of scope (Stage 5)
- Number of independent rebalance periods and the resulting uncertainty in
  performance metrics (Stage 7)
- Risk-free rate assumption and annualization convention used for Sharpe
  (Stage 7)
- Future extensions: risk parity/Black-Litterman/HRP, sector constraints,
  rolling window, point-in-time universe membership, TCA/execution
