import numpy as np

def momentum_factor(prices, window=60):
    return prices.pct_change(window)

def volatility_factor(returns, window=60):
    return returns.rolling(window).std()

def zscore(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)

def combine_factors(momentum, volatility, weights):
    mom_z = zscore(momentum)
    vol_z = zscore(volatility)

    score = weights["momentum"] * mom_z - weights["volatility"] * vol_z
    return score

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    std = excess_returns.std()

    if std == 0 or np.isnan(std):
        return 0

    return np.sqrt(252) * excess_returns.mean() / std

def select_portfolio(scores, top_n=3):
    ranks = scores.rank(axis=1, ascending=False)
    portfolio = ranks <= top_n
    return portfolio.astype(int)