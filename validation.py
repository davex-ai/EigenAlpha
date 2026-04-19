from metrics import sharpe_ratio
from strategy import zscore
import pandas as pd


def information_coefficient_series(factor, future_returns):
    factor_aligned, returns_aligned = factor.align(future_returns, join="inner")
    return factor_aligned.corrwith(returns_aligned, axis=1)

def information_coefficient(factor, future_returns):
    return information_coefficient_series(factor, future_returns).mean()

def factor_return(factor, returns):
    factor = zscore(factor)

    weights = factor.div(factor.abs().sum(axis=1), axis=0)
    future_ret = returns.shift(-1)

    factor_ret = (weights * future_ret).sum(axis=1)
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

def alpha_decomposition(factors, returns):
    results = {}

    for name, factor in factors.items():
        f_ret = factor_return(factor, returns)
        results[name] = f_ret

    return pd.DataFrame(results)