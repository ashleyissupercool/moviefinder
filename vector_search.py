from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from cortex import CortexClient, DistanceMetric
from sentence_transformers import SentenceTransformer

# ---- Config ----
HOST = "localhost:50051"
COLLECTION = "movies"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384

_model: SentenceTransformer | None = None


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "it", "its", "this", "that", "they", "them",
    "their", "he", "she", "his", "her", "we", "our", "you", "your", "who",
    "what", "which", "as", "up", "out", "about", "into", "over", "after",
    "not", "no", "so", "if", "then", "than", "when", "where", "how",
}

def _remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.lower().split() if w not in STOPWORDS)


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _stable_id(title: str) -> int:
    key = title.strip().lower().encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:15], 16)


def _safe_get(row: pd.Series, col: str) -> str:
    val = row.get(col, "")
    return "" if pd.isna(val) else str(val)


def index_movies(df: pd.DataFrame, batch_size: int = 256, force_reindex: bool = False) -> None:
    """
    Embed df["search_text"] and upsert into VectorAI DB.
    Run once (or when dataset changes), not on every Streamlit rerun.
    """
    with CortexClient(HOST) as client:
        already_indexed = client.has_collection(COLLECTION)
        if already_indexed and not force_reindex:
            print("Collection already exists, skipping indexing.")
            return
        if already_indexed and force_reindex:
            client.delete_collection(COLLECTION)
        client.create_collection(
            name=COLLECTION,
            dimension=DIM,
            distance_metric=DistanceMetric.COSINE,
        )

    if "search_text" not in df.columns:
        raise ValueError("DataFrame must have a 'search_text' column.")

    df = df.copy()
    for col in ["title", "search_text", "overview"]:
        df[col] = df[col].fillna("").astype(str)

    # keep entry with longest overview for duplicate titles
    df["_overview_len"] = df["overview"].str.len()
    df = (df.sort_values("_overview_len", ascending=False)
            .drop_duplicates(subset=["title"])
            .drop(columns=["_overview_len"])
            .reset_index(drop=True))

    df["search_text"] = df["search_text"].apply(_remove_stopwords)

    model = get_model()
    embeddings = model.encode(
        df["search_text"].tolist(),
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )

    ids: List[int] = []
    payloads: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        title = _safe_get(row, "title")
        ids.append(int(idx))  # use row index — guaranteed unique
        payloads.append({
            "title": title,
            "overview": _safe_get(row, "overview"),
        })

    with CortexClient(HOST) as client:
        n = len(ids)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            client.batch_upsert(
                COLLECTION,
                ids=ids[start:end],
                vectors=embeddings[start:end].tolist(),
                payloads=payloads[start:end],
            )
    print(f"Indexed {n} movies into VectorAI DB.")


def search_movies_vector(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Search VectorAI DB for top_k movies matching query.
    Returns list of dicts: [{"title": ..., "overview": ..., "confidence": ...}, ...]
    """
    model = get_model()
    q_vec = model.encode([_remove_stopwords(query)], normalize_embeddings=True)[0].tolist()

    with CortexClient(HOST) as client:
        if not client.has_collection(COLLECTION):
            return []
        results = client.search(COLLECTION, query=q_vec, top_k=top_k, with_payload=True)

    output: List[Dict[str, Any]] = []
    for r in results:
        score = float(getattr(r, "score", 0.0))
        conf = round(max(0.0, min(1.0, score)) * 100, 1)
        payload = getattr(r, "payload", {}) or {}
        output.append({
            "title": payload.get("title", ""),
            "overview": payload.get("overview", ""),
            "confidence": conf,
        })

    return output
