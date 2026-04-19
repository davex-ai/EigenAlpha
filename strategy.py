import numpy as np
import pandas as pd

from metrics import zscore
from utilities import register_factor
from metrics import get_rebalance_dates

@register_factor("momentum")
def momentum_factor(prices, window=60):
    return prices.pct_change(window).rolling(10).mean()

@register_factor("size")
def size_factor(prices):
    return np.log(prices)

@register_factor("value")
def value_factor(prices):
    return -prices / prices.rolling(252).mean()

@register_factor("volume_momentum")
def volume_momentum(vol):
    return vol.rolling(20).mean() / vol.rolling(60).mean()

@register_factor("mean_reversion")
def mean_reversion(prices):
    return -prices.pct_change(5)

@register_factor("residual_momentum")
def residual_momentum(returns):
    market = returns.mean(axis=1)

    residuals = returns.sub(market, axis=0)

    return residuals.rolling(60).mean().rolling(5).mean()

@register_factor("volatility")
def volatility_factor(returns, window=60):
    return -returns.rolling(window).std()

def combine_factors(factors, weights):
    score = None
    for name, factor in factors.items():
        z = zscore(factor)

        w = weights.get(name, 0)

        score = z * w if score is None else score + z * w

    return score

def rebalance_portfolio(scores, freq="ME", top_n=8):
    rebalance_dates = get_rebalance_dates(scores.index, freq)
    portfolio = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)

    for date in rebalance_dates:
        if date not in scores.index:
            continue

        ranks = scores.loc[date].rank(ascending=False)
        long_mask = ranks <= top_n

        # Long-Only Weights
        weights = long_mask.astype(float)
        if weights.sum() > 0:
            weights /= weights.sum()

        portfolio.loc[date] = weights

    return portfolio.ffill().fillna(0)
