# 🧠📈 EigenAlpha — Multi-Factor Quant Engine

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=EigenAlpha&fontSize=40&fontColor=ffffff&animation=fadeIn" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Quant-Factor%20Model-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Backtesting-Engine-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Research%20Stage-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/github/stars/davex-ai/EigenAlpha?style=for-the-badge"/>
</p>

---

## ⚡ Overview

**EigenAlpha** is a modular **multi-factor quantitative research engine** built to simulate how real-world hedge funds construct, test, and evaluate investment strategies.

It combines:

* Factor modeling
* Portfolio construction
* Backtesting with transaction costs
* Alpha decomposition
* Risk-aware optimization

👉 Designed as a **mini quant lab**, not just a script.

---

## 🧩 Core Features

### 🧠 Factor Engine

* Momentum
* Volatility
* Value (proxy-based)
* Size
* Volume Momentum
* Mean Reversion
* Residual Momentum

All factors are:

* 📦 Plug-and-play via decorator system
* ⚙️ Automatically computed with dynamic argument injection

---

### 📊 Portfolio Construction

* Monthly rebalancing
* Long-only / configurable exposure
* Ranking-based selection
* Supports:

  * Equal Weighting
  * Risk Parity
  * Mean-Variance Optimization

---

### 🧪 Backtesting Engine

* Forward returns (`shift(-5)`)
* Transaction cost modeling
* Turnover calculation
* Volatility targeting

---

### 📈 Factor Evaluation

* Information Coefficient (IC)
* Factor Sharpe Ratio
* IC time-series tracking
* Alpha decomposition

---

### ⚙️ Research System

* Config-driven experiments
* Train/Test split (Out-of-sample validation)
* Factor orthogonalization (optional)
* Dynamic factor weighting based on IC × Sharpe

---

## 🧬 Architecture

```
EigenAlpha/
│
├── main.py              # Research pipeline
├── strategy.py          # Factor definitions + portfolio logic
├── utilities.py         # Data + backtesting engine
├── metrics.py           # Risk & performance metrics
├── validation.py        # Factor evaluation + alpha analysis
└── data.py              # Universe loader
```

---

## 🚀 Example Workflow

```python
# Load data
prices, volumes = load_data(tickers)
returns = compute_returns(prices)

# Compute factors
factors = compute_factors(prices, returns, volumes)

# Evaluate signals
print(evaluate_factors(factors, returns))

# Combine factors
scores = combine_factors(factors, weights)

# Build portfolio
portfolio = rebalance_portfolio(scores)

# Backtest
performance = backtest(returns, portfolio)
```

---

## 📊 Sample Output

* 📈 Strategy vs Benchmark performance
* 🧠 Factor contribution breakdown
* 📉 Drawdown analysis
* 🔍 IC stability plots

---

## 🧠 Key Concepts Implemented

* Cross-sectional Z-score normalization
* Multi-factor modeling
* Alpha signal evaluation
* Portfolio optimization techniques
* Out-of-sample validation
* Transaction cost modeling

---

## ⚠️ Important Notes

* Uses **proxy data** for some factors (e.g., value, size)
* Designed for **research and experimentation**, not live trading
* Results depend heavily on:

  * Data quality
  * Factor design
  * Market regime

---

## 🔥 What Makes This Different

This isn’t a toy project.

EigenAlpha replicates the **core workflow used in quant funds**, including:

* Factor research loops
* Signal validation (IC)
* Risk-aware portfolio construction
* Alpha attribution

---

## 🛠️ Tech Stack

* Python 🐍
* pandas / numpy
* matplotlib
* yfinance

---

## 📌 Future Improvements

* Real fundamental data (P/E, earnings, etc.)
* Larger universe (100+ assets)
* Factor neutralization (sector/market)
* Machine learning signal integration
* Web dashboard (FastAPI + React)

---

## 👤 Author

**Dave (davex-ai)**

> Building toward elite-level quantitative systems & research

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,100:0f2027&height=120&section=footer"/>
</p>
