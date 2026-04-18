import numpy as np
import pandas as pd

def zscore(df):
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
    ortho = {}

    factor_names = list(factors.keys())

    for i, name in enumerate(factor_names):
        target = zscore(factors[name])

        others = [zscore(factors[n]) for n in factor_names if n != name]

        if not others:
            ortho[name] = target
            continue

        X = np.stack([f.values for f in others], axis=2)

        residuals = []

        for t in range(target.shape[0]):
            y = target.iloc[t].values
            x = X[t]

            mask = ~np.isnan(y) & ~np.isnan(x).any(axis=1)

            if mask.sum() < 5:
                residuals.append(np.full_like(y, np.nan))
                continue

            beta = np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]
            y_hat = x @ beta
            resid = y - y_hat

            residuals.append(resid)

        ortho[name] = pd.DataFrame(residuals, index=target.index, columns=target.columns)

    return ortho
