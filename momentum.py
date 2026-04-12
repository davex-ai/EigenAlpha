import numpy as np

def momentum_factor(prices, window=60):
    return prices.pct_change(window)

def volatility_factor(returns, window=60):
    return returns.rolling(window).std()

def zscore(df):
    return (df - df.mean(axis=1, skipna=True).values.reshape(-1,1)) / df.std(axis=1, skipna=True).values.reshape(-1,1)

def combine_factors(momentum, volatility):
    mom_z = zscore(momentum)
    vol_z = zscore(volatility)

    score = mom_z - vol_z
    return score

def select_portfolio(scores, top_n=10):
    ranks = scores.rank(axis=1, ascending=False)
    portfolio = ranks <= top_n
    return portfolio.astype(int)