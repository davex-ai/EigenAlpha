
import matplotlib.pyplot as plt
import numpy as np

from metrics import sharpe_ratio, max_drawdown, orthogonalize_factors
from strategy import  combine_factors, rebalance_portfolio
from utilities import load_data, compute_returns, backtest, compute_factors, risk_parity_weights, mean_variance_weights, \
    volatility_targeting
from validation import evaluate_factors, alpha_decomposition, information_coefficient_series, information_coefficient
from data import load_local_tickers

tickers = load_local_tickers()
tickers = [t.replace('.','-') for t in tickers]
# tickers = ["AAPL", "MSFT", "GOOG", 'GOOGL', "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH", "HD", "PG"]

prices, volumes = load_data(tickers)
returns = compute_returns(prices)

split_date = "2023-01-01"
train_prices = prices.loc[:split_date]
train_returns = returns.loc[:split_date]
train_volumes = volumes.loc[:split_date]
test_prices = prices.loc[split_date:]
test_returns = returns.loc[split_date:]
test_volumes = volumes.loc[split_date:]

all_factors = compute_factors(prices, returns,volumes)
print("Factor Evaluation:\n", evaluate_factors(all_factors, returns))

benchmark, benchmark_volumes  = load_data(["SPY"])
benchmark_returns = compute_returns(benchmark).squeeze()

def plot_performance(strategy_returns, benchmark_returns):
    (1 + strategy_returns).cumprod().plot(label="Strategy", figsize=(10, 5))
    # Match benchmark dates to strategy dates
    (1 + benchmark_returns.loc[strategy_returns.index]).cumprod().plot(label="Benchmark")
    plt.title("Out-of-Sample Performance")
    plt.legend()
    plt.show()

# Test

configs = [
    {"momentum": 0.7, "volatility": -0.3},
    {"momentum": 0.5, "volatility": -0.5},
    {"momentum": 1.0, "volatility": -0.2},
    {"momentum": 1.5, "volatility": 0.1},
    {"momentum": 2.0, "volatility": 0.4},

]
train_factors = compute_factors(train_prices, train_returns, train_volumes)
# train_factors = orthogonalize_factors(train_factors)
for k, v in train_factors.items():
    print(k, type(v), v.shape if hasattr(v, "shape") else "NO SHAPE")
best_config = None
best_sharpe = -np.inf

for config in configs:
    scores = combine_factors(train_factors, config)
    portfolio = rebalance_portfolio(scores)

    ret = backtest(train_returns, portfolio)
    cumulative = (1 + ret).cumprod()
    print("Max Drawdown:", max_drawdown(cumulative))
    s = sharpe_ratio(ret)

    if s > best_sharpe:
        best_sharpe = s
        best_config = config

    print("Config", config, "Sharpe Ratio", s)


test_factors = compute_factors(test_prices, test_returns, test_volumes)
# test_factors = orthogonalize_factors(test_factors)
weights = {}
eval_df = evaluate_factors(train_factors, train_returns)

for name in train_factors:
    ic = eval_df.loc[name, "IC"]
    sharpe = eval_df.loc[name, "Sharpe"]

    weights[name] = max(ic * sharpe, 0)

scores = combine_factors(test_factors, weights)
# scores = combine_factors(test_factors, best_config)
portfolio = rebalance_portfolio(scores)
print("Net exposure:", portfolio.sum(axis=1).mean())
print("Test Portfolio:", portfolio)
for name, factor in test_factors.items():
    ic_series = information_coefficient_series(factor, test_returns.shift(-5))
    ic_series.plot(title=f"{name} IC Over Time")
    plt.axhline(0, linestyle="--")
    plt.show()

test_perf = backtest(test_returns, portfolio, weighting="equal")
test_perf = volatility_targeting(test_perf)
alpha = alpha_decomposition(test_factors, test_returns)

(alpha.cumsum()).plot(title="Test Factor Contributions")
plt.show()

# evaluate factors
print(evaluate_factors(test_factors, test_returns))

# plot
plot_performance(test_perf, benchmark_returns)

# sharpe
print("Strategy Sharpe:", sharpe_ratio(test_perf))
print("Benchmark Sharpe:", sharpe_ratio(benchmark_returns.loc[test_perf.index]))
