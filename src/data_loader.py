import pandas as pd
import yfinance as yf

def fetch_market_data(tickers, start="2020-01-01", end="2024-01-01"):
    """
    Fetches historical adjusted close prices for given tickers.
    """
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data

def compute_features(price_data):
    """
    Computes basic financial features: daily returns, volatility, momentum.
    """
    returns = price_data.pct_change().dropna()
    volatility = returns.rolling(window=30).std()
    momentum = price_data / price_data.shift(30) - 1

    features = pd.DataFrame({
        "returns": returns.mean(axis=1),
        "volatility": volatility.mean(axis=1),
        "momentum": momentum.mean(axis=1)
    }, index=returns.index)

    return features

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    market_data = fetch_market_data(tickers)
    features = compute_features(market_data)
    
    market_data.to_csv("data/market_data.csv")
    features.to_csv("data/market_features.csv")

    print("Market data and features saved.")