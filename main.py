import yfinance as yf
import pandas as pd


def load_data(tickers, start="2020-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    data = data.dropna(axis=1, how="any")  # remove broken tickers
    return data

def compute_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def backtest(returns, portfolio):
    portfolio_returns = []

    for date in returns.index:
        if date not in portfolio.index:
            continue

        weights = portfolio.loc[date]
        if weights.sum() == 0:
            portfolio_returns.append(0)
            continue

        weights = weights / weights.sum()  # equal weight

        daily_ret = (weights * returns.loc[date]).sum()
        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=returns.index[:len(portfolio_returns)])