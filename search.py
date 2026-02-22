from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    #data cleaning and preprocessing
    df = pd.read_csv(path)
    df = df[["title", "overview"]]
    df = df.dropna(subset=["overview"]).reset_index(drop=True)

    #compiling all text fields into a single search field
    df["search_text"] = (
        df["title"].fillna("") + " " +
        df["overview"].fillna("")
    ).str.lower()
    
    return df



#new embedding engine using sentence transformers & search function using cosine similarity on embeddings
def build_embedding_engine(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df["search_text"].tolist(),
        show_progress_bar=True, normalize_embeddings=True
    )
    return model, np.array(embeddings)

def search_movies_embeddings(query, df, model, embeddings, top_k=5):
    query = str(query).lower().strip()
    query_vec = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_vec, embeddings)[0]

    top_k = min(5, len(df))
    top_idx = sims.argsort()[-top_k:][::-1]

    results = df.iloc[top_idx].copy()
    results["confidence"] = sims[top_idx] * 100

    # results["confidence"] = (sims[top_idx] / sims[top_idx].max()) * 100

    return results

#old tf-idf engine & search function
def build_engine(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["search_text"])
    return vectorizer, X

def search_movies(query, df, vectorizer, X, top_k=5):
    query = str(query).lower().strip()
    query_vec = vectorizer.transform([query]) 
    sims = cosine_similarity(query_vec, X)[0]

    top_k = min(5, len(df))
    top_idx = sims.argsort()[-top_k:][::-1]

    results = df.iloc[top_idx].copy()
    results["confidence"] = sims[top_idx] * 100
    
    # results["confidence"] = (sims[top_idx] / sims[top_idx].max()) * 100

    return results
