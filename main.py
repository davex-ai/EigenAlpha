
import matplotlib.pyplot as plt
import numpy as np

from metrics import sharpe_ratio, max_drawdown
from strategy import momentum_factor, volatility_factor, combine_factors, rebalance_portfolio
from utilities import load_data, compute_returns, backtest, compute_factors, compute_turnover
from validation import evaluate_factors
from data import load_local_tickers

tickers = load_local_tickers()
prices = load_data(tickers)
returns = compute_returns(prices)
split_date = "2023-01-01"

train_prices = prices.loc[:split_date]
test_prices = prices.loc[split_date:]

train_returns = compute_returns(train_prices)
test_returns = compute_returns(test_prices)

momentum = momentum_factor(prices)
volatility = volatility_factor(returns)
valid = momentum.notna() & volatility.notna()
initial_weights = {"momentum": 0.7, "volatility": 0.3}
scores = combine_factors(momentum, volatility, initial_weights)
scores = scores.where(valid)
portfolio = rebalance_portfolio(scores)

benchmark = load_data(["SPY"])
benchmark_returns = compute_returns(benchmark)

benchmark_cum = (1 + benchmark_returns.squeeze()).cumprod()

portfolio_returns = backtest(returns, portfolio)

cumulative = (1 + portfolio_returns).cumprod()
print("Max Drawdown:", max_drawdown(cumulative))
cost = 0.001  # 0.1%
def apply_transaction_costs(returns, portfolio, cost=0.001):
    turnover = compute_turnover(portfolio)
    cost_series = turnover * cost

    return returns - cost_series

def plot_performance(strategy_returns, benchmark_returns):
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(10,5))
    plt.plot(strategy_cum, label="Strategy")
    plt.plot(benchmark_cum, label="Benchmark")
    plt.title("Strategy vs Benchmark")
    plt.legend()
    plt.show()

# print("Scores: ", scores)
# print("Prices: ", prices)
# print("Returns: ", returns)
# print("Momentum: ", momentum)
# print("Votality: ", volatility)
# print("Returns: ", returns)
# print("Portfolio: ", portfolio)
# print("Portfolio Returns: ", portfolio_returns)
# print("Sharpe: ", sharpe_ratio(portfolio_returns))
# print("Cumulative Returns: ", cumulative)

# Test

configs = [
    {"momentum": 0.7, "volatility": -0.3},
    {"momentum": 0.5, "volatility": -0.5},
    {"momentum": 1.0, "volatility": -0.2},
    {"momentum": 1.5, "volatility": 0.1},
    {"momentum": 2.0, "volatility": 0.4},

]

best_config = None
best_sharpe = -np.inf

for config in configs:
    factors = compute_factors(train_prices, train_returns)
    strategy_returns = backtest(returns, portfolio)
    scores = combine_factors(factors["momentum"], factors["volatility"], config)
    portfolio = rebalance_portfolio(scores)

    ret = backtest(train_returns, portfolio)
    s = sharpe_ratio(ret)

    if s > best_sharpe:
        best_sharpe = s
        best_config = config

    print("Config", config, "Sharpe Ratio", sharpe_ratio(strategy_returns))

prices = load_data(tickers)
returns = compute_returns(prices)
factors = compute_factors(test_prices, test_returns)
scores = combine_factors(factors["momentum"], factors["volatility"], best_config)
portfolio = rebalance_portfolio(scores)

test_perf = backtest(test_returns, portfolio)

print("OUT-OF-SAMPLE SHARPE:", sharpe_ratio(test_perf))

# factors
factors = compute_factors(prices, returns)

# evaluate factors
print(evaluate_factors(factors, returns))

# strategy
weights_config = {"momentum": 0.7, "volatility": 0.3}

scores = combine_factors(factors['momentum'], factors['volatility'], weights_config)
portfolio = rebalance_portfolio(scores, top_n=5)

portfolio_returns = backtest(returns, portfolio)

# benchmark
benchmark = load_data(["SPY"])
benchmark_returns = compute_returns(benchmark).squeeze()

# plot
plot_performance(portfolio_returns, benchmark_returns)

# sharpe
print("Strategy Sharpe:", sharpe_ratio(portfolio_returns))
print("Benchmark Sharpe:", sharpe_ratio(benchmark_returns))

