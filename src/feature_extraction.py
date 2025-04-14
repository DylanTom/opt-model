import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = "your-api-key-here"

def get_chatgpt_insight(ticker):
    """
    Fetches ChatGPT investment insights for a given stock ticker.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Provide a brief investment summary for {ticker}"}]
    )
    return response["choices"][0]["message"]["content"]

def extract_text_features(tickers):
    """
    Extracts investment insights for multiple stocks and stores in a dataframe.
    """
    insights = {ticker: get_chatgpt_insight(ticker) for ticker in tickers}
    df = pd.DataFrame(list(insights.items()), columns=["Ticker", "ChatGPT_Insight"])
    return df

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    insights_df = extract_text_features(tickers)
    
    insights_df.to_csv("data/chatgpt_insights.csv", index=False)
    print("ChatGPT insights saved.")