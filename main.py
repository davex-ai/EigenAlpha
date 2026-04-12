import yfinance as yf
import pandas as pd
import numpy as np

def load_data(tickers, start="2020-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    data = data.dropna(axis=1, how="any")  # remove broken tickers
    return data

def compute_returns(prices):
    returns = prices.pct_change().dropna()
    return returns