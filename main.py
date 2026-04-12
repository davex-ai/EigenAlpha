
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "UNH", "HD", "PG"]

prices = load_data(tickers)
returns = compute_returns(prices)

momentum = momentum_factor(prices)
volatility = volatility_factor(returns)

scores = combine_factors(momentum, volatility)
portfolio = select_portfolio(scores)

portfolio_returns = backtest(returns, portfolio)

cumulative = (1 + portfolio_returns).cumprod()

plt.figure()
plt.plot(cumulative)
plt.title("Portfolio Performance")
plt.show()