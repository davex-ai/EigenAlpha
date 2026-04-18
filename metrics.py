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
    ortho = {}
    common_index = None
    common_columns = None

    for f in factors.values():
        common_index = f.index if common_index is None else common_index.intersection(f.index)
        common_columns = f.columns if common_columns is None else common_columns.intersection(f.columns)

    aligned = {
        name: f.loc[common_index, common_columns]
        for name, f in factors.items()
    }

    factors = aligned

    factor_names = list(factors.keys())

    for name in factor_names:
        target = zscore(factors[name])
        others = [zscore(factors[n]) for n in factor_names if n != name]

        if not others:
            ortho[name] = target
            continue

        residuals = []

        for date in target.index:
            y = target.loc[date]
            X = pd.concat([f.loc[date] for f in others if date in f.index], axis=1)

            # drop NaNs
            df = pd.concat([y, X], axis=1).dropna()

            if df.shape[0] < 5:
                residuals.append(pd.Series(np.nan, index=y.index))
                continue

            y_clean = df.iloc[:, 0]
            X_clean = df.iloc[:, 1:]

            beta = np.linalg.lstsq(X_clean.values, y_clean.values, rcond=None)[0]
            y_hat = X_clean.values @ beta
            resid = y_clean.values - y_hat

            res_series = pd.Series(np.nan, index=y.index)
            res_series[df.index] = resid

            residuals.append(res_series)

        ortho[name] = pd.DataFrame(residuals, index=target.index, columns=target.columns)

    return ortho