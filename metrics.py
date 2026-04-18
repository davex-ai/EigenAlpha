import numpy as np
import pandas as pd

def zscore(df):
    if isinstance(df, pd.Series):
        mean = df.mean()
        std = df.std()
        return (df - mean) / std

    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)

    return df.sub(mean, axis=0).div(std, axis=0)

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / 252
    std = excess_returns.std()

    if std == 0 or np.isnan(std):
        return 0

    return np.sqrt(252) * excess_returns.mean() / std

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

def get_rebalance_dates(dates, freq="ME"):
    return pd.Series(dates, index=dates).resample(freq).last().dropna().values

def orthogonalize_factors(factors):
    # Step 1: Align all factors to same index/columns
    aligned = pd.concat(factors, axis=1)

    # MultiIndex → (date, ticker) x factor
    aligned = aligned.dropna()

    # Split back
    ortho = {}

    for name in factors.keys():
        y = aligned[name]

        X = aligned.drop(columns=name)

        if X.shape[1] == 0:
            ortho[name] = y.unstack()
            continue

        betas = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        y_hat = X.values @ betas
        resid = y.values - y_hat
        if isinstance(ortho[name], pd.Series):
            ortho[name] = ortho[name].to_frame()

        ortho[name] = pd.DataFrame(
            resid,
            index=y.index
        ).unstack()

    return ortho