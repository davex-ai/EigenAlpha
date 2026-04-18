import yfinance as yf
import numpy as np
import pandas as pd


def load_data(tickers, start="2020-01-01", end="2025-01-01"):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    data = data.dropna(axis=1, how="any")
    if 'Close' in data.columns:
        data = data['Close']
    else:
        data = data['Adj Close']
    data = data.dropna(axis=1, how="any")
    return data


def compute_turnover(portfolio):
    return portfolio.diff().abs().sum(axis=1)

def compute_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def apply_transaction_costs(returns, portfolio, cost=0.001):
    turnover = compute_turnover(portfolio)
    cost_series = turnover * cost

    return returns - cost_series

def backtest(returns, portfolio, cost=0.001, weighting="equal"):
    weights = pd.DataFrame(index=portfolio.index, columns=portfolio.columns)

    for date in portfolio.index:
        active = portfolio.loc[date] != 0

        if active.sum() == 0:
            weights.loc[date] = 0
            continue

        subset_returns = returns.loc[:date, active]

        if weighting == "risk_parity":
            w = risk_parity_weights(subset_returns)
        elif weighting == "mean_variance":
            w = mean_variance_weights(subset_returns)
        else:
            w = pd.Series(1/active.sum(), index=portfolio.columns[active])

        full_w = pd.Series(0, index=portfolio.columns)
        full_w[active] = w

        weights.loc[date] = full_w

    weights = weights.fillna(0)

    gross_returns = (weights * returns.shift(-1)).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1)
    costs = turnover * cost

    return (gross_returns - costs).dropna()

FACTORS = {}

def register_factor(name):
    def wrapper(fn):
        FACTORS[name] = fn
        return fn
    return wrapper


def compute_factors(prices, returns):
    results = {}
    context = {"prices": prices, "returns": returns}

    for name, fn in FACTORS.items():
        # Only pass what the function is asking for
        import inspect
        sig = inspect.signature(fn)
        # Filter the context to only include keys that match function parameters
        kwargs = {k: v for k, v in context.items() if k in sig.parameters}
        results[name] = fn(**kwargs)

    return results

def volatility_targeting(returns, target_vol=0.15):
    vol = returns.rolling(20).std() * np.sqrt(252)

    scaling = target_vol / vol
    return returns * scaling

def risk_parity_weights(returns):
    vol = returns.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    return weights

def mean_variance_weights(returns):
    mu = returns.mean()
    cov = returns.cov()

    inv_cov = np.linalg.pinv(cov.values)

    weights = inv_cov @ mu.values
    weights /= weights.sum()

    return pd.Series(weights, index=returns.columns)