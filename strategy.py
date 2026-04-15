import numpy as np
import pandas as pd

from metrics import zscore
from utilities import register_factor
from validation import get_rebalance_dates

@register_factor("momentum")
def momentum_factor(prices, window=60):
    return prices.pct_change(window)

@register_factor("volatility")
def volatility_factor(returns, window=60):
    return returns.rolling(window).std()

def combine_factors(momentum, volatility, weights):
    mom_z = zscore(momentum)
    vol_z = zscore(volatility)

    score = weights["momentum"] * mom_z + weights["volatility"] * vol_z
    return score

# def select_portfolio(scores, top_n=3, ):
#     ranks_desc = scores.rank(axis=1, ascending=False)
#     ranks_asc = scores.rank(axis=1, ascending=True)
#
#     long_portfolio = (ranks_desc <= top_n).astype(int)
#     short_portfolio = (ranks_asc <= top_n).astype(int)
#     portfolio = long_portfolio - short_portfolio
#     return portfolio

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
