import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Optional: Use dotenv if API keys needed
from dotenv import load_dotenv
load_dotenv()

def load_and_process(csv_path):
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records.")

    # 2. Drop any nulls in ChatGPT_Insight
    df.dropna(subset=["ChatGPT_Insight"], inplace=True)

    # 3. Vectorize the text using TF-IDF (could replace with OpenAI embeddings later)
    tfidf = TfidfVectorizer(max_features=200)
    text_vectors = tfidf.fit_transform(df["ChatGPT_Insight"]).toarray()

    # 4. Optional: Standardize and reduce dimension
    scaler = StandardScaler()
    text_vectors_scaled = scaler.fit_transform(text_vectors)

    pca = PCA(n_components=4)  # tweak as needed
    pca_features = pca.fit_transform(text_vectors_scaled)

    # 5. Combine back into DataFrame
    pca_df = pd.DataFrame(pca_features, columns=[f"Insight_PC{i}" for i in range(1, 5)])
    final_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

    return final_df

if __name__ == "__main__":
    csv_path = "data/chatgpt_insights_processed.csv"  # adjust path as needed
    df = load_and_process(csv_path)

    # Preview
    print(df.head())

    # Save or move to modeling
    df.to_csv("data/insight_features.csv", index=False)
    print("Processed feature file saved to data/insight_features.csv")