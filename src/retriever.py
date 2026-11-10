
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PaperRetriever:
    def __init__(self, chunks_path=None):
        if chunks_path is None:
            chunks_path = PROJECT_ROOT / "data" / "chunks.json"

        with open(chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.chunk_texts
        )

    def retrieve(self, question, top_k=5):
        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector, self.tfidf_matrix
        ).flatten()

        top_indices = scores.argsort()[::-1]
        top_indices = [
            index for index in top_indices
            if scores[index] > 0
        ][:top_k]

        results = []

        for index in top_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results
