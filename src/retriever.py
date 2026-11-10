import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PaperRetriever:
    def __init__(self, chunks_path="data/chunks.json"):
        with open(chunks_path, "r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, question, top_k=5):
        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).flatten()

        top_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in top_indices:
            result = self.chunks[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


if __name__ == "__main__":
    retriever = PaperRetriever()

    question = "What is the attention mechanism in the Transformer?"
    results = retriever.retrieve(question)

    print(f"\nQuestion: {question}\n")

    for i, result in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Paper: {result['title']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text'][:400]}...")
        print()