import numpy as np
import pandas as pd

from metrics import zscore
from utilities import register_factor
from metrics import get_rebalance_dates

@register_factor("momentum")
def momentum_factor(prices, window=60):
    return prices.pct_change(window)

@register_factor("size")
def size_factor(prices):
    return np.log(prices)

@register_factor("value")
def value_factor(prices):
    return -prices / prices.rolling(252).mean()

@register_factor("volatility")
def volatility_factor(returns, window=60):
    return returns.rolling(window).std()

def combine_factors(factors, weights):
    score = None

    for name, factor in factors.items():
        z = zscore(factor)

        if name == "volatility":
            z = -z

        w = weights.get(name, 0)

        score = z * w if score is None else score + z * w

    return score

def rebalance_portfolio(scores, freq="ME", top_n=5):
    rebalance_dates = get_rebalance_dates(scores.index, freq)

    portfolio = pd.DataFrame(0, index=scores.index, columns=scores.columns)

    for date in rebalance_dates:
        if date not in scores.index:
            continue

        ranks = scores.loc[date].rank(ascending=False)

        long = ranks <= top_n
        short = ranks >= (len(ranks) - top_n + 1)

        weights = long.astype(int) - short.astype(int)

        portfolio.loc[date] = weights

    # forward fill positions until next rebalance
    portfolio = portfolio.replace(0, np.nan).ffill().fillna(0)

    return portfolio
