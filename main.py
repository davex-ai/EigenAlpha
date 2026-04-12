
import matplotlib.pyplot as plt

from momentum import momentum_factor, volatility_factor, combine_factors, select_portfolio, sharpe_ratio
from utilities import load_data, compute_returns, backtest

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH", "HD", "PG"]

prices = load_data(tickers)
returns = compute_returns(prices)

momentum = momentum_factor(prices)
volatility = volatility_factor(returns)
valid = momentum.notna() & volatility.notna()

scores = combine_factors(momentum, volatility)
scores = scores.where(valid)
portfolio = select_portfolio(scores)

benchmark = load_data(["SPY"])
benchmark_returns = compute_returns(benchmark)

benchmark_cum = (1 + benchmark_returns.squeeze()).cumprod()

portfolio_returns = backtest(returns, portfolio)

cumulative = (1 + portfolio_returns).cumprod()

plt.figure()
plt.plot(cumulative)
plt.title("Portfolio Performance")
plt.show()
plt.plot(cumulative, label="Strategy")
plt.plot(benchmark_cum, label="Benchmark")
plt.legend()

print("Scores: ", scores)
print("Prices: ", prices)
print("Returns: ", returns)
print("Momentum: ", momentum)
print("Votality: ", volatility)
print("Returns: ", returns)
print("Portfolio: ", portfolio)
print("Portfolio Returns: ", portfolio_returns)
print("Sharpe: ", sharpe_ratio(portfolio_returns))
print("Cumulative Returns: ", cumulative)