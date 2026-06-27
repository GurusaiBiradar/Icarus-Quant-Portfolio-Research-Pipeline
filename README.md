# Icarus — Quant Portfolio Research Pipeline

An end-to-end quantitative equity research pipeline built as a portfolio project
targeting quant research internships (equity factor research, multi-asset/tactical
allocation). Every stage is hand-rolled and explainable in an interview.

**Guiding principle**: one example of each idea, not every variation — two factors,
one risk model, one optimizer, hand-rolled metrics.

---

## Stack

| Library | Role |
|---|---|
| `pandas`, `numpy`, `yfinance`, `pyarrow` | Data ingestion and storage |
| `scikit-learn` | Ledoit-Wolf covariance estimation |
| `xgboost` | Conditional ML alpha model |
| `cvxpy` | Mean-variance portfolio optimizer |
| `scipy.stats` | IC validation (Spearman, ground-truth check) |
| `streamlit` | Dashboard (reads precomputed files only) |

Python 3.11, conda environment `icarus`.

---

## Setup

```bash
conda create -n icarus python=3.11 -y
conda activate icarus
pip install pandas numpy yfinance pyarrow scikit-learn xgboost cvxpy scipy streamlit pytest
```

---

## Running the pipeline

```bash
# 1. Download data and run both backtests (~15 min total)
python run_pipeline.py

# 2. Launch dashboard
streamlit run app.py

# 3. Run test suite (excludes slow backtest integration tests)
pytest tests/ --ignore=tests/test_backtest_loop.py

# Full test suite including backtest integration (~7 min)
pytest tests/
```

---

## Pipeline stages

```
Stage 1  universe.py + data_pipeline.py   Universe construction, filters, Parquet cache
Stage 2  factors.py                        Momentum (12-1 month) + reversal (1 month)
Stage 3  alpha_model.py                    Rank-average baseline + XGBoost (conditional)
Stage 4  risk_model.py                     Ledoit-Wolf shrinkage covariance
Stage 5  optimizer.py                      Mean-variance QP via cvxpy
Stage 6  backtest_loop.py                  Monthly expanding-window rebalancing loop
Stage 7  evaluation.py                     Sharpe, max drawdown, IC — all hand-rolled
Stage 8  app.py                            Streamlit dashboard
         run_pipeline.py                   End-to-end orchestration script
Stage 5  Execution                         Skipped (see below)
```

### Universe (Stage 1)

Approximately 50 stocks sampled from the current S&P 500 constituent list
(scraped from Wikipedia). Three liquidity/history filters are applied:

| Filter | Threshold |
|---|---|
| Minimum average closing price | $5 |
| Minimum average dollar volume (ADV) | $5 M/day |
| Minimum trading history | 756 days (~3 years) |

Five years of daily OHLCV data are downloaded via `yfinance`.

### Factors (Stage 2)

**Momentum** — 12-1 month return (skip-month convention):

```
score(T) = price(T − 1 month) / price(T − 12 months) − 1
```

The skip-month convention excludes the most recent month from the numerator
because the last month's return is dominated by short-term reversal noise that
would partially cancel the trend signal.

**Reversal** — 1-month return:

```
score(T) = price(T) / price(T − 1 month) − 1
```

Both factors use calendar-date lookbacks (`pd.DateOffset`) resolved to the last
available trading day via `Series.asof()`, not positional row offsets. This
prevents silent lookahead bias when any rows are missing.

### Alpha model (Stage 3)

Two models compete. The backtest is run independently for each:

**Rank-average baseline** (no training): cross-sectionally rank each factor as a
percentile in (0, 1], with momentum ranked ascending (higher = stronger buy) and
reversal ranked descending (lower/negative = stronger contrarian buy). Average the
two percentile ranks into a combined alpha score.

**XGBoost** (conditional): trained at each rebalance date on all historical
`(factor_scores_at_T, forward_return_T→T+1)` pairs using an expanding window.
Predicts cross-sectional forward returns from factor scores.

Both models expose the same `AlphaModel` interface (`fit()` / `predict_scores()`),
so the backtest loop substitutes them without branching.

#### XGBoost decision

XGBoost is retained only if it outperforms the rank-average baseline on IC
(Spearman rank correlation between predicted scores and realised returns) across
multiple out-of-sample rebalance periods. The decision is made automatically
by `run_pipeline.py` and printed to the console.

**Result from this run (29 tickers, 47 out-of-sample periods, 2019-2024):**

| Model | Sharpe (ann.) | Mean IC | IC IR |
|---|---|---|---|
| Rank-Average (baseline) | 1.13 | −0.0105 | −0.13 |
| XGBoost | 0.94 | −0.0366 | −0.52 |

**Decision: Rank-Average baseline retained.**  XGBoost shows worse IC and lower
Sharpe — adding model complexity here is not justified.

**Interpreting the results:**

*Positive Sharpe despite negative IC* is expected and not a contradiction.
The portfolio is long-only and fully invested, so it carries a positive beta to
the market.  The sample period (2019–2024) included a strong equity bull run, so
the portfolio earned a market premium regardless of how well the factor model
ranked individual stocks cross-sectionally.  IC measures cross-sectional ranking
accuracy; Sharpe measures total return risk-adjusted.

*Near-zero IC* is also expected given the setup constraints:
1. The universe is small (29 tickers after one download failure) — cross-sectional
   variation in returns is limited with fewer stocks.
2. The IC standard error is `std_IC / √N ≈ 0.10 / √47 ≈ 0.015`, so a mean IC of
   −0.0105 is well within one standard error of zero — statistically indistinguishable
   from a null signal.
3. Survivorship bias inflates realised returns (market prices went up) without
   improving cross-sectional predictive accuracy, which is what IC measures.

If the margin is small or inconsistent, the simpler baseline wins — that is an
honest finding, not a failure.  A model complexity increase must be justified by
a reliable improvement in out-of-sample predictive accuracy.

### Risk model (Stage 4)

Ledoit-Wolf shrinkage covariance estimated from daily log returns, scaled by ×21
(≈ trading days per month) to produce a monthly covariance consistent with the
monthly rebalancing frequency.

**Why Ledoit-Wolf**: with ~50 assets and ~750 daily observations the condition
number of the sample covariance can be in the thousands, making the optimizer
unstable. Ledoit-Wolf shrinks toward a scaled identity matrix with an analytically
optimal coefficient and guarantees a positive semi-definite result (required by
the cvxpy solver).

A PSD assertion (`check_psd`) is run after every estimation as a defensive guard.

### Optimizer (Stage 5)

Mean-variance QP solved via cvxpy:

```
maximise  w^T μ − risk_aversion × w^T Σ w
subject to
  Σ w_i = 1    (fully invested)
  w_i ≥ 0      (long-only, no shorting)
  w_i ≤ 0.10   (max 10% per stock)
```

`μ` is the cross-sectional alpha score vector; `Σ` is the monthly Ledoit-Wolf
covariance. Tickers with NaN alpha scores (insufficient lookback history) are
excluded from the QP and assigned weight 0; the remaining weights still sum to 1.

### Backtest loop (Stage 6)

Monthly rebalancing with an expanding window:

1. At each rebalance date T (after a 13-month burn-in), slice all data to `[start, T]`.
2. Compute factor panels on the slice — no future prices used.
3. Fit the alpha model on all observable `(factor_scores_at_t, forward_return)` pairs
   strictly before T.
4. Predict alpha scores at T.
5. Estimate the Ledoit-Wolf covariance from log returns up to T.
6. Solve the mean-variance QP for portfolio weights.
7. Compute the simple portfolio return for `[T, T+1]`.

**Expanding vs rolling window**: this implementation uses a growing training set
(more data at each step). Rolling window with a fixed lookback is a future extension.

### Evaluation (Stage 7)

All metrics are hand-rolled from first principles:

| Metric | Formula | Note |
|---|---|---|
| Sharpe ratio | `mean(r − rf_m) / std(r − rf_m) × √12` | Monthly series, annualised ×√12 |
| Max drawdown | `min((cum_value − cum_peak) / cum_peak)` | Most negative value |
| IC | `Pearson(rank(scores), rank(returns))` per date | Spearman = Pearson of ranks |
| IC IR | `mean_IC / std_IC × √12` | Signal consistency |

---

## Key design decisions

### Why reversal instead of P/E

`yfinance.Ticker.info["trailingPE"]` returns the *current* price-to-earnings ratio,
not the historical P/E as of any past date. Using it in a backtest would leak today's
valuation data into every historical period — a straightforward form of lookahead bias.
Reversal uses the same price data as momentum and is strictly point-in-time safe.

### Why execution is skipped

Stage 5 (execution / transaction cost modelling) is out of scope for research-track
roles. A one-sentence callout is the appropriate treatment: the pipeline assumes
frictionless execution at end-of-month closing prices. Real implementation would
require market-impact and bid-ask models, which vary by asset size and broker.

---

## Known limitations

### Survivorship bias

The universe is constructed from the **current** S&P 500 constituent list, then
historical prices are fetched for those tickers. Stocks that were members of the
index during the sample period but were subsequently removed (due to bankruptcy,
merger, or index reconstitution) are absent. This imparts an upward bias to all
reported returns because the universe is tilted toward companies that survived and
remained large-cap throughout the sample.

True point-in-time universe construction requires a historical constituents database
(e.g., Compustat or a commercial vendor feed), which is outside the scope of this
project.

### yfinance download failures

Occasional tickers fail to download (e.g., `MRK: OperationalError('database is locked')`).
This is a transient `yfinance` issue and does not indicate a code bug.  Failed tickers
are silently dropped; the universe shrinks by the number of failures.  Re-running the
pipeline usually resolves them.  Set `FORCE_REFRESH = True` in `run_pipeline.py` to
re-attempt all downloads.

### Statistical uncertainty (sample size)

With a 5-year data window and 13-month burn-in, the backtest produces approximately
**35–50 independent monthly rebalance periods**. At a typical monthly IC of 0.03–0.06,
the standard error of the mean IC is `std_IC / √N ≈ 0.05 / √40 ≈ 0.008`. Confidence
intervals around all reported Sharpe ratios and IC estimates are therefore wide.
The results illustrate the pipeline architecture and methodology; they are not a
statistically robust claim about live performance.

---

## Conventions

**Risk-free rate**: constant 2% per annum → `rf_monthly = 0.02 / 12 ≈ 0.00167`.
A constant approximation is appropriate here; the key requirement is that it is
stated explicitly and applied consistently.

**Sharpe annualisation**: monthly excess-return Sharpe × **√12** (not ×√252,
which is the correct factor for daily returns). The return series is monthly, so
monthly compounding applies.

**Portfolio returns**: simple returns are used for portfolio arithmetic
(`Σ w_i × r_i`), because simple returns aggregate linearly across assets.
Log returns are used for the covariance estimation (more statistically stable),
then scaled to monthly units before being passed to the optimizer.

---

## Future extensions

| Extension | Notes |
|---|---|
| Rolling window | Fixed lookback instead of expanding; reduces distribution shift |
| Walk-forward XGBoost tuning | Re-tune `max_depth`, `learning_rate` at each step |
| Additional factors | Quality (ROE stability), value (book-to-market from financial statements) |
| Sector / country constraints | Add linear inequality constraints to the cvxpy QP |
| Risk parity | Equalise marginal risk contributions instead of mean-variance |
| Black-Litterman | Blend market-implied returns with factor views |
| HRP (Hierarchical Risk Parity) | Cluster-based allocation, no matrix inversion |
| True point-in-time universe | Historical S&P 500 constituents database |
| Transaction cost model | Market-impact + bid-ask spread; compute net-of-cost returns |
| VaR / CVaR constraints | Add tail-risk budget to the QP |
