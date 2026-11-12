
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastembed import TextEmbedding

from src.retriever import PaperRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks_path=None,
        model_name="BAAI/bge-small-en-v1.5",
        rrf_k=60
    ):
        self.baseline = PaperRetriever(chunks_path)
        self.chunks = self.baseline.chunks
        self.rrf_k = rrf_k

        self.embedding_model = TextEmbedding(model_name=model_name)

        self.chunk_embeddings = np.array(
            list(self.embedding_model.embed(self.baseline.chunk_texts))
        )

    def retrieve(self, question, top_k=5):
        # 1. Lexical retrieval using TF-IDF
        query_vector = self.baseline.vectorizer.transform([question])

        lexical_scores = cosine_similarity(
            query_vector,
            self.baseline.tfidf_matrix
        ).flatten()

        lexical_order = [
            i for i in np.argsort(lexical_scores)[::-1]
            if lexical_scores[i] > 0
        ]

        lexical_ranks = {
            index: rank
            for rank, index in enumerate(lexical_order, start=1)
        }

        # 2. Semantic retrieval using BGE
        query_embedding = np.array(
            list(self.embedding_model.query_embed(question))
        )

        semantic_scores = cosine_similarity(
            query_embedding,
            self.chunk_embeddings
        ).flatten()

        semantic_order = np.argsort(semantic_scores)[::-1]

        semantic_ranks = {
            index: rank
            for rank, index in enumerate(semantic_order, start=1)
        }

        # 3. Reciprocal Rank Fusion
        fused_scores = {}

        for index, rank in lexical_ranks.items():
            fused_scores[index] = (
                fused_scores.get(index, 0)
                + 1 / (self.rrf_k + rank)
            )

        for index, rank in semantic_ranks.items():
            fused_scores[index] = (
                fused_scores.get(index, 0)
                + 1 / (self.rrf_k + rank)
            )

        # 4. Return the highest-ranked passages
        ranked_indices = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True
        )[:top_k]

        results = []

        for index in ranked_indices:
            result = self.chunks[index].copy()
            result["score"] = fused_scores[index]
            result["rrf_score"] = fused_scores[index]
            result["lexical_score"] = float(lexical_scores[index])
            result["semantic_score"] = float(semantic_scores[index])
            result["lexical_rank"] = lexical_ranks.get(index)
            result["semantic_rank"] = semantic_ranks.get(index)

            results.append(result)

        return results
