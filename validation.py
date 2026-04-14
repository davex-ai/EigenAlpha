from metrics import sharpe_ratio
from strategy import zscore
import pandas as pd


def information_coefficient(factor, future_returns):
    aligned = factor.align(future_returns, join="inner")[0]
    ic = aligned.corrwith(future_returns, axis=1)
    ic.title("Information Coefficient")
    ic.plot()
    return ic.mean()

def get_rebalance_dates(dates, freq="M"):
    return pd.Series(dates, index=dates).resample(freq).last().dropna().values

def factor_return(factor, returns):
    factor = zscore(factor)

    weights = factor.div(factor.abs().sum(axis=1), axis=0)

    factor_ret = (weights * returns).sum(axis=1)
    return factor_ret
def evaluate_factors(factors, returns):
    results = {}

    future_returns = returns.shift(-1)

    for name, factor in factors.items():
        ic = information_coefficient(factor, future_returns)
        f_ret = factor_return(factor, returns)
        sharpe = sharpe_ratio(f_ret)

        results[name] = {
            "IC": ic,
            "Sharpe": sharpe
        }

    return pd.DataFrame(results).T
