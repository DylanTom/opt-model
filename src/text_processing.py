import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the required dataset (only needed once)
nltk.download("vader_lexicon")

def compute_sentiment(text):
    """
    Computes sentiment polarity scores for a given text.
    Returns a score between -1 (negative) and +1 (positive).
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)["compound"]
    return sentiment_score

def add_sentiment_scores(df, text_column="ChatGPT_Insight"):
    """
    Adds sentiment scores to a DataFrame containing ChatGPT insights.
    """
    df["Sentiment_Score"] = df[text_column].apply(compute_sentiment)
    return df


if __name__ == "__main__":
    # Load ChatGPT insights
    insights_df = pd.read_csv("data/chatgpt_insights.csv")

    # Compute sentiment scores
    insights_with_sentiment = add_sentiment_scores(insights_df)

    # Save processed data
    insights_with_sentiment.to_csv("data/chatgpt_insights_processed.csv", index=False)
    print("✅ Sentiment scores added and saved.")