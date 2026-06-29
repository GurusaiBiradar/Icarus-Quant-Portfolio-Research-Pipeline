# Icarus — Quantitative Portfolio Research Pipeline

End-to-end equity factor research: universe → signals → risk → optimizer → walk-forward backtest → evaluation. Every metric is hand-rolled; no pyfolio wrappers.

---

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  yfinance  ·  ~50 S&P 500 stocks  ·  5 years daily OHLCV       │
  └──────────────────────────┬──────────────────────────────────────┘
                             │  price > $5  ·  ADV > $5M/day
                             ▼
             ┌───────────────────────────────┐
             │          Alpha Signals        │
             │  Momentum   12-1 month ret    │
             │  Reversal    1-month  ret     │
             └──────────────┬────────────────┘
                            │ cross-sectional factor scores
                   ┌────────┴─────────┐
                   ▼                  ▼
          ┌─────────────┐    ┌─────────────────┐
          │  Rank-Avg   │    │    XGBoost      │
          │  (baseline) │    │ expanding train │
          └──────┬──────┘    └────────┬────────┘
                 └──────────┬─────────┘
                            │  α  +  Ledoit-Wolf Σ
                            ▼
             ┌───────────────────────────────┐
             │   Mean-Variance Optimizer     │
             │   fully invested  ·  long-only│
             │   max 10% per stock  (cvxpy)  │
             └──────────────┬────────────────┘
                            │ weights
                            ▼
             ┌───────────────────────────────┐
             │   Monthly Rebalancing Loop    │
             │   expanding window            │
             │   47 out-of-sample periods    │
             └──────────────┬────────────────┘
                            │
                            ▼
            Sharpe  ·  Max Drawdown  ·  IC  ·  IC IR
                       Streamlit Dashboard
```

---

## Results

**2019 – 2024 · 29 tickers · 47 independent rebalance periods**

| Model | Sharpe (ann.) | Mean IC | IC IR |
|---|---|---|---|
| Rank-Average (baseline) | **1.13** | −0.0105 | −0.13 |
| XGBoost | 0.94 | −0.0366 | −0.52 |

**Baseline retained.** XGBoost underperforms on both Sharpe and IC across all out-of-sample periods. With two factors over a 29-stock cross-section, the model has too little variation to find real signal before overfitting — that is the honest finding.

*Positive Sharpe alongside near-zero IC isn't a contradiction:* the long-only portfolio earns market beta through a strong 2019–2024 equity run. IC measures cross-sectional ranking accuracy; Sharpe measures total risk-adjusted return — these are orthogonal.

---

## Stack

`pandas` · `numpy` · `yfinance` · `pyarrow` · `scikit-learn` (Ledoit-Wolf) · `xgboost` · `cvxpy` · `scipy` · `streamlit` · Python 3.11

---

## Quick Start

```bash
conda create -n icarus python=3.11 -y && conda activate icarus
pip install pandas numpy yfinance pyarrow scikit-learn xgboost cvxpy scipy streamlit pytest

python run_pipeline.py   # download data + run both backtests (~15 min first run)
streamlit run app.py     # launch dashboard

pytest tests/ --ignore=tests/test_backtest_loop.py   # fast tests
pytest tests/                                         # full suite (~7 min)
```

---

## Notable Design Choices

**`asof()` over `shift(N)` for factor lookbacks** — `shift(21)` is position-based and silently lands on the wrong date when rows are missing. `DateOffset` computes an exact calendar date; `asof()` resolves it to the last available trading day, returning `NaN` if before the series start. This eliminates silent lookahead bias in factor computation.

**Reversal instead of P/E** — `yfinance` returns only the *current* P/E, not a historical series. Using it in a backtest writes today's valuation into every past rebalance date. Reversal uses the same price data as momentum and is strictly point-in-time safe.

**Ledoit-Wolf shrinkage** — with ~50 assets and ~750 observations the sample covariance condition number reaches into the thousands. Ledoit-Wolf shrinks toward a scaled identity matrix with an analytically optimal coefficient, guaranteeing a PSD result required by the QP solver.

---

## Known Limitations

- **Survivorship bias** — universe is current S&P 500 membership applied to historical prices; delisted stocks are absent, inflating reported returns.
- **Small sample** — 47 monthly periods give a mean IC standard error of ≈ 0.015; all Sharpe and IC estimates carry wide confidence intervals.
- **Frictionless execution** — end-of-month closing prices; no transaction cost or market-impact model.

---

## Future Extensions

Rolling window · Walk-forward XGBoost tuning · Additional factors (quality, value) · Sector constraints · Risk parity · Black-Litterman · HRP · Point-in-time universe · Transaction cost model · VaR/CVaR constraints
