import yfinance as yf


def load_data(tickers, start="2020-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    if 'Close' in data.columns:
        data = data['Close']
    else:
        data = data['Adj Close']
    data = data.dropna(axis=1, how="any")
    return data


def compute_turnover(portfolio):
    return portfolio.diff().abs().sum(axis=1)

def compute_returns(prices):
    returns = prices.pct_change().dropna()
    return returns


def backtest(returns, portfolio, cost=0.001):
    weights = portfolio.div(portfolio.abs().sum(axis=1), axis=0).fillna(0)

    gross_returns = (weights * returns.shift(-1)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    costs = turnover * cost

    net_returns = gross_returns - costs
    return net_returns.dropna()


FACTORS = {}

def register_factor(name):
    def wrapper(fn):
        FACTORS[name] = fn
        return fn
    return wrapper


def compute_factors(prices, returns):
    results = {}
    # Use a dictionary to store the data available
    context = {"prices": prices, "returns": returns}

    for name, fn in FACTORS.items():
        # Only pass what the function is asking for
        import inspect
        sig = inspect.signature(fn)
        # Filter the context to only include keys that match function parameters
        kwargs = {k: v for k, v in context.items() if k in sig.parameters}
        results[name] = fn(**kwargs)

    return results

