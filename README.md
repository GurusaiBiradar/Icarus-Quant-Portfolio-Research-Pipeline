# Icarus — Quantitative Portfolio Research Pipeline

A complete, end-to-end equity factor research pipeline: universe construction → alpha signals → risk model → portfolio optimization → walk-forward backtest → evaluation. Every component is written from scratch — no pyfolio, no QuantStats wrappers.

---

## Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│  yfinance / Wikipedia                                                   │
│        │                                                                │
│        ▼                                                                │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Universe   │───▶│   Factors    │───▶│ Alpha Model  │               │
│  │  ~50 S&P    │    │  Momentum    │    │ Rank-Average │               │
│  │  stocks     │    │  Reversal    │    │ vs XGBoost   │               │
│  └─────────────┘    └──────────────┘    └──────┬───────┘               │
│                                                │                       │
│  ┌─────────────┐    ┌──────────────┐           │                       │
│  │  Optimizer  │◀───│  Risk Model  │◀──────────┘                       │
│  │  Mean-Var   │    │ Ledoit-Wolf  │                                    │
│  │  QP (cvxpy) │    │  Covariance  │                                    │
│  └──────┬──────┘    └──────────────┘                                    │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────┐                       │
│  │  Expanding-Window Monthly Backtest Loop     │                       │
│  │  (47 out-of-sample rebalance periods)       │                       │
│  └──────────────────────┬──────────────────────┘                       │
│                         │                                               │
│                         ▼                                               │
│         Sharpe · Max Drawdown · IC · IC IR                             │
│                    Streamlit Dashboard                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Results

**Backtest period**: 2019–2024 · 29 tickers · 47 independent monthly rebalance periods

| Model | Sharpe (ann.) | Mean IC | IC IR |
|---|---|---|---|
| Rank-Average (baseline) | **1.13** | −0.0105 | −0.13 |
| XGBoost | 0.94 | −0.0366 | −0.52 |

**Finding: the rank-average baseline retained.**

XGBoost shows lower Sharpe and worse IC across all 47 periods. With only two factors over a ~29-stock cross-section, the model has insufficient variation to learn real signal before it overfits. Adding complexity here is not justified — that is the honest finding.

> *Positive Sharpe with near-zero IC is not a contradiction.* The portfolio is long-only and fully invested, so it earns a positive market beta. The 2019–2024 sample included a strong equity bull run. Sharpe measures total risk-adjusted return; IC measures cross-sectional ranking accuracy — these are orthogonal.

> *Near-zero IC is expected given the setup:* the standard error of mean IC is `std_IC / √47 ≈ 0.015`, so a mean IC of −0.0105 is statistically indistinguishable from zero. Signal is either genuinely absent on this small universe or buried beneath survivorship-bias-inflated returns.

---

## Stack

| Library | Role |
|---|---|
| `pandas` `numpy` `yfinance` `pyarrow` | Data ingestion and Parquet storage |
| `scikit-learn` | Ledoit-Wolf covariance estimation |
| `xgboost` | ML alpha model (conditional on IC test) |
| `cvxpy` | Mean-variance QP optimizer |
| `scipy.stats` | IC Spearman correlation (ground-truth check) |
| `streamlit` | Dashboard (reads precomputed Parquet files) |

Python 3.11 · conda environment `icarus`

---

## Quick Start

```bash
conda create -n icarus python=3.11 -y
conda activate icarus
pip install pandas numpy yfinance pyarrow scikit-learn xgboost cvxpy scipy streamlit pytest

# Download data + run both backtests (~15 min on first run; cached after that)
python run_pipeline.py

# Launch dashboard
streamlit run app.py

# Tests (fast)
pytest tests/ --ignore=tests/test_backtest_loop.py

# Full test suite including backtest integration (~7 min)
pytest tests/
```

---

## Source Layout

```
src/
  universe.py        Universe construction + liquidity filters
  data_pipeline.py   Clean, save, pivot to wide panels; Parquet I/O
  factors.py         Momentum (12-1 month) + reversal (1 month) signals
  alpha_model.py     Rank-average baseline + XGBoost alpha model
  risk_model.py      Ledoit-Wolf shrinkage covariance
  optimizer.py       Mean-variance QP (long-only, max 10% per stock)
  backtest_loop.py   Monthly expanding-window rebalancing loop
  evaluation.py      Sharpe, max drawdown, IC — hand-rolled
data/                Parquet outputs (gitignored)
tests/
app.py               Streamlit dashboard
run_pipeline.py      End-to-end orchestration
```

---

## Stage Details

### Universe

Approximately 50 stocks drawn from the current S&P 500 constituent list (Wikipedia scrape, with a hardcoded large-cap fallback). Three filters applied:

| Filter | Threshold |
|---|---|
| Min average closing price | $5 |
| Min average dollar volume (ADV) | $5M / day |
| Min trading history | 756 days (~3 years) |

Five years of daily OHLCV data are downloaded via `yfinance`.

### Factors

**Momentum** — 12-1 month return (skip-month convention):

```
score(T) = price(T − 1 month) / price(T − 12 months) − 1
```

The skip-month convention excludes the most recent month because short-term reversal noise in the last few weeks partially cancels the trend signal.

**Reversal** — 1-month return:

```
score(T) = price(T) / price(T − 1 month) − 1
```

Both factors use calendar-date lookbacks (`pd.DateOffset`) resolved to the last available trading day via `Series.asof()` — not positional `shift(N)`. This matters: `shift(21)` lands on the wrong date whenever any rows are missing; `asof()` always returns a real price or `NaN`, which propagates cleanly through the pipeline.

### Alpha Model

Two models run in parallel. The backtest is executed independently for each, then compared on IC across all out-of-sample periods:

**Rank-average baseline** — no training. Each factor is ranked cross-sectionally as a percentile (momentum ascending, reversal descending) and the two percentile ranks are averaged into a combined alpha score.

**XGBoost** — trained at each rebalance date on all historical `(factor_scores_at_T, forward_return_T→T+1)` pairs using an expanding window. Shallow trees (`max_depth=3`), strong regularization.

Both expose the same `AlphaModel` interface (`fit()` / `predict_scores()`), so the backtest loop substitutes them without any branching.

### Risk Model

Ledoit-Wolf shrinkage covariance estimated from daily log returns, scaled ×21 to monthly units:

With ~50 assets and ~750 observations, the sample covariance condition number reaches into the thousands, making the QP numerically unstable. Ledoit-Wolf shrinks toward a scaled identity matrix with an analytically optimal coefficient and guarantees a positive semi-definite result. A `check_psd` assertion runs after every estimation.

### Optimizer

Mean-variance QP solved via cvxpy:

```
maximise  w^T μ − λ · w^T Σ w
subject to
  Σ w_i = 1      (fully invested)
  w_i ≥ 0        (long-only)
  w_i ≤ 0.10     (max 10% per stock)
```

`μ` is the alpha score vector; `Σ` is the Ledoit-Wolf monthly covariance. Tickers with NaN alpha scores are excluded from the QP and assigned weight zero; remaining weights still sum to 1.

### Backtest Loop

Monthly rebalancing, expanding window:

1. At rebalance date T (after a 13-month burn-in), slice all data to `[start, T]`
2. Compute factor panels on the slice — no future prices used
3. Fit the alpha model on all `(factor_scores_at_t, forward_return)` pairs before T
4. Predict alpha scores at T
5. Estimate Ledoit-Wolf covariance from log returns up to T
6. Solve the mean-variance QP for portfolio weights
7. Compute simple portfolio return for `[T, T+1]`

The 13-month burn-in covers the 12-month momentum lookback plus one safety month. Execution is assumed frictionless at end-of-month closing prices.

### Evaluation

All metrics are hand-rolled:

| Metric | Formula | Convention |
|---|---|---|
| Sharpe ratio | `mean(r − rf_m) / std(r − rf_m) × √12` | Monthly series; rf = 2% p.a. |
| Max drawdown | `min((cum_value − cum_peak) / cum_peak)` | Most negative trough |
| IC | `Pearson(rank(scores), rank(returns))` per date | Spearman = Pearson of ranks |
| IC IR | `mean_IC / std_IC × √12` | Signal consistency |

Annualization uses **×√12** (monthly returns), not ×√252 which applies to daily series. Risk-free rate is a constant 2% p.a. — a defensible approximation for a multi-year backtest where the key requirement is that it is stated and applied consistently.

---

## Design Decisions

### Why reversal instead of P/E

`yfinance.Ticker.info["trailingPE"]` returns the *current* price-to-earnings ratio, not a historical time series. Using it in a backtest would write today's valuation data into every historical rebalance date — a direct form of lookahead bias. Reversal uses only the same price data as momentum and is strictly point-in-time safe.

### Expanding window, not rolling

An expanding training set (all data up to T) is simpler and accumulates more observations over time. A fixed-lookback rolling window reduces distribution shift at the cost of throwing away early data; it is the natural next iteration.

### No execution stage

Transaction cost modelling is a separate skill set and less relevant to research-track roles. The pipeline assumes frictionless execution. Real implementation would require market-impact and bid-ask spread models, which vary by asset size and broker.

---

## Known Limitations

**Survivorship bias** — the universe is the *current* S&P 500 membership applied to historical prices. Stocks that were constituents during the sample period but were subsequently removed (bankruptcy, merger, reconstitution) are absent. This imparts an upward bias to all reported returns. True point-in-time universe construction requires a historical constituents database (Compustat, Bloomberg).

**Statistical uncertainty** — 47 independent monthly periods over 5 years is a small sample. At a typical monthly IC of 0.03–0.06 the standard error of the mean IC is ≈ 0.008, giving wide confidence intervals around all reported Sharpe and IC estimates. These numbers characterize the pipeline's behavior on this sample; they are not a robust claim about forward performance.

**Universe size** — 29 tickers after download failures limits cross-sectional variation and makes it harder for any ranking model to demonstrate reliable IC.

---

## Future Extensions

| Extension | Notes |
|---|---|
| Rolling window | Fixed lookback reduces distribution shift at late periods |
| Walk-forward XGBoost tuning | Re-tune `max_depth`, `learning_rate` at each step |
| Additional factors | Quality (ROE stability), value (book-to-market from financial statements) |
| Sector / country constraints | Linear inequality constraints in the cvxpy QP |
| Risk parity | Equalize marginal risk contributions; no matrix inversion needed |
| Black-Litterman | Blend market-implied returns with factor views |
| HRP | Cluster-based allocation via hierarchical linkage |
| Point-in-time universe | Historical S&P 500 constituents from Compustat or a vendor feed |
| Transaction cost model | Market-impact + bid-ask spread; net-of-cost returns |
| VaR / CVaR constraints | Tail-risk budget added to the QP |
