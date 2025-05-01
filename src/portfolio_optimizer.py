import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

# --- Parameters ---
PRICE_LOOKBACK_DAYS = 60
RISK_AVERSION = 0.1  # lambda

# --- 1. Load data ---
insight_df = pd.read_csv("data/insight_features.csv")

# Ensure needed columns exist
required_cols = [col for col in insight_df.columns if col.startswith("Insight_PC")]

tickers = insight_df["Ticker"].tolist()

# --- 2. Pull price data ---
price_data = yf.download(tickers, period=f"{PRICE_LOOKBACK_DAYS}d")["Close"]
returns = price_data.pct_change().dropna()

# Align price data to your tickers
returns = returns[tickers]

# --- 3. Add real next-period return to training set ---
latest_returns = returns.iloc[-1]
insight_df = insight_df.set_index("Ticker")
insight_df["NextReturn"] = latest_returns

# --- 4. Train model to predict returns from insights ---
X = insight_df[required_cols + ["Sentiment_Score"]]
y = insight_df["NextReturn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train_scaled, y_train)

# --- 5. Predict expected returns (μ vector) ---
mu = model.predict(X_all_scaled)

# --- 6. Compute covariance matrix (Σ) ---
cov_matrix = returns.cov().values

# --- 7. Mean-Variance Optimization with CVXPY ---
n = len(tickers)
w = cp.Variable(n)

objective = cp.Maximize(mu @ w - RISK_AVERSION * cp.quad_form(w, cov_matrix))
constraints = [cp.sum(w) == 1, w >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()

# --- 8. Output results ---
weights = w.value
portfolio = pd.DataFrame({"Ticker": tickers, "Weight": weights})
portfolio = portfolio.sort_values("Weight", ascending=False)

print("\n📊 Optimized Portfolio Allocation:\n")
print(portfolio.to_string(index=False))

portfolio.to_csv("data/optimized_portfolio.csv", index=False)
print("\n✅ Portfolio saved to data/optimized_portfolio.csv")