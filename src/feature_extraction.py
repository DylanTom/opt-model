import openai
import pandas as pd
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

def get_chatgpt_insight(ticker):
    """
    Fetches ChatGPT investment insights for a given stock ticker.
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": f"What are some key insights about {ticker}?"}]
    )
    return response.choices[0].message.content

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