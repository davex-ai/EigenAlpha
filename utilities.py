import yfinance as yf
import pandas as pd

from main import strategy_returns, cost, apply_transaction_costs


def load_data(tickers, start="2020-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
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

def backtest(returns, portfolio, cost=0.001):
    shifted_returns = returns.shift(-1)
    weights = portfolio.div(portfolio.sum(axis=1), axis=0).fillna(0)
    gross_returns = (weights * shifted_returns).sum(axis=1)
    turnover = portfolio.diff().abs().sum(axis=1)
    costs = turnover * cost
    net_returns = gross_returns - costs
    return net_returns

FACTORS = {}

def register_factor(name):
    def wrapper(fn):
        FACTORS[name] = fn
        return fn
    return wrapper

def compute_factors(prices, returns):
    results = {}

    for name, fn in FACTORS.items():
        results[name] = fn(prices, returns)

    return results

