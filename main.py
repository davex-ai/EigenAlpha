
import matplotlib.pyplot as plt

from strategy import momentum_factor, volatility_factor, combine_factors, sharpe_ratio, max_drawdown, \
    rebalance_portfolio
from utilities import load_data, compute_returns, backtest, compute_factors, compute_turnover
from validation import evaluate_factors

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH", "HD", "PG"]

prices = load_data(tickers)
returns = compute_returns(prices)

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

for config in configs:
    score = combine_factors(momentum, volatility, config)
    portfolio = rebalance_portfolio(score, top_n=3)
    strategy_returns = backtest(returns, portfolio)

    print("Config", config, "Sharpe Ratio", sharpe_ratio(strategy_returns))

prices = load_data(tickers)
returns = compute_returns(prices)

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

