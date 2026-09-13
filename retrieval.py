"""
Cheap, fast retrieval over the historical corpus: TF-IDF + cosine similarity.
No embeddings API needed, so this stays free and instant even at corpus size ~4000.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config as C


class HistoricalRetriever:
    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus["customer_text"].fillna(""))

    @classmethod
    def from_parquet(cls, path=C.CORPUS_PARQUET):
        return cls(pd.read_parquet(path))

    def top_k(self, query_text: str, k: int = 3):
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        results = []
        for i in top_idx:
            row = self.corpus.iloc[i]
            results.append({
                "similarity": float(sims[i]),
                "past_customer_text": row["customer_text"],
                "past_brand_reply": row["brand_reply_text"],
            })
        return results
