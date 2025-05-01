import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

# === PARAMETERS ===
LOOKBACK_DAYS = 60
TAU = 0.05  # uncertainty in prior
RISK_AVERSION = 2.5  # typical lambda

# === 1. LOAD INSIGHT DATA ===
insight_df = pd.read_csv("data/insight_features.csv")
tickers = insight_df["Ticker"].tolist()

# === 2. GET PRICE DATA ===
price_data = yf.download(tickers, period=f"{LOOKBACK_DAYS}d")["Close"]
returns = price_data.pct_change().dropna()
returns = returns[tickers]

# === 3. ADD REALIZED RETURN ===
latest_returns = returns.iloc[-1]
insight_df = insight_df.set_index("Ticker")
insight_df["NextReturn"] = latest_returns

# === 4. PREDICT VIEWS (Q) ===
features = [col for col in insight_df.columns if col.startswith("Insight_PC")]
X = insight_df[features + ["Sentiment_Score"]]
y = insight_df["NextReturn"]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_all_scaled = scaler.transform(X)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train_scaled, y_train)

Q = model.predict(X_all_scaled)  # ChatGPT/sentiment-based "views"

# === 5. BLACK-LITTERMAN COMPONENTS ===

# Σ = sample covariance matrix
Σ = returns.cov().values
n = len(tickers)

# Market weights proxy (uniform or from real data)
w_mkt = np.ones(n) / n

# Implied equilibrium returns (π)
π = RISK_AVERSION * Σ @ w_mkt

# View matrix P (identity = 1 view per asset)
P = np.eye(n)

# Ω = diagonal view uncertainty (variance of model residuals)
Ω = np.diag(np.var(y_train - model.predict(X_train_scaled)) * np.ones(n))

# Compute posterior expected returns: μ_bl
τΣ = TAU * Σ
middle = np.linalg.inv(P @ τΣ @ P.T + Ω)
μ_bl = np.linalg.inv(np.linalg.inv(τΣ) + P.T @ middle @ P) @ (np.linalg.inv(τΣ) @ π + P.T @ middle @ Q)

# === 6. OPTIMIZE PORTFOLIO USING μ_bl ===
w = cp.Variable(n)
objective = cp.Maximize(μ_bl @ w - RISK_AVERSION * cp.quad_form(w, Σ))
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve()

weights = w.value
portfolio = pd.DataFrame({"Ticker": tickers, "Weight": weights})
portfolio = portfolio.sort_values("Weight", ascending=False)

print("\n🧠 Black-Litterman Optimized Portfolio:\n")
print(portfolio.to_string(index=False))
portfolio.to_csv("data/bl_portfolio.csv", index=False)
print("\n✅ Saved to data/bl_portfolio.csv")