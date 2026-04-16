
import matplotlib.pyplot as plt
import numpy as np

from metrics import sharpe_ratio, max_drawdown
from strategy import momentum_factor, volatility_factor, combine_factors, rebalance_portfolio
from utilities import load_data, compute_returns, backtest, compute_factors, compute_turnover
from validation import evaluate_factors, alpha_decomposition
from data import load_local_tickers

# tickers = load_local_tickers()
# tickers = [t.replace('.','-') for t in tickers]
tickers = ["AAPL", "MSFT", "GOOG", 'GOOGL', "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH", "HD", "PG"]

prices = load_data(tickers)
returns = compute_returns(prices)

split_date = "2023-01-01"
train_prices = prices.loc[:split_date]
train_returns = returns.loc[:split_date]
test_prices = prices.loc[split_date:]
test_returns = returns.loc[split_date:]

all_factors = compute_factors(prices, returns)
print("Factor Evaluation:\n", evaluate_factors(all_factors, returns))

benchmark = load_data(["SPY"])
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
train_factors = compute_factors(train_prices, train_returns)
best_config = None
best_sharpe = -np.inf

for config in configs:
    scores = combine_factors({"momentum": train_factors['momentum'], "volatility": train_factors['volatility']}, config)
    portfolio = rebalance_portfolio(scores)

    ret = backtest(train_returns, portfolio)
    cumulative = (1 + ret).cumprod()
    print("Max Drawdown:", max_drawdown(cumulative))
    s = sharpe_ratio(ret)

    if s > best_sharpe:
        best_sharpe = s
        best_config = config

    print("Config", config, "Sharpe Ratio", s)


test_factors = compute_factors(test_prices, test_returns)
scores = combine_factors({"momentum": test_factors['momentum'], "volatility": test_factors['volatility']}, best_config)
portfolio = rebalance_portfolio(scores)

test_perf = backtest(test_returns, portfolio)
alpha = alpha_decomposition(test_factors, test_returns)

(alpha.cumsum()).plot(title="Test Factor Contributions")
plt.show()

print("OUT-OF-SAMPLE SHARPE:", sharpe_ratio(test_perf))

# evaluate factors
print(evaluate_factors(test_factors, test_perf))

# plot
plot_performance(test_perf, benchmark_returns)

# sharpe
print("Strategy Sharpe:", sharpe_ratio(test_perf))
print("Benchmark Sharpe:", sharpe_ratio(benchmark_returns))
