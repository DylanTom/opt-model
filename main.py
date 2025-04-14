from src.data_loader import fetch_market_data, compute_features
from src.feature_extraction import extract_text_features
import pandas as pd

# Define stock tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Step 1: Load Market Data
market_data = fetch_market_data(tickers)
features = compute_features(market_data)

# Step 2: Extract ChatGPT Investment Insights
insights_df = extract_text_features(tickers)

# Step 3: Save Data
market_data.to_csv("data/market_data.csv")
features.to_csv("data/market_features.csv")
insights_df.to_csv("data/chatgpt_insights.csv", index=False)

print("✅ Data pipeline complete. Market data & ChatGPT insights saved.")