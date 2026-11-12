
import pandas as pd

from src.retriever import PaperRetriever
from src.hybrid_retriever import HybridRetriever

questions = pd.read_csv("experiments/quality_eval.csv")

tfidf = PaperRetriever()
hybrid = HybridRetriever()

expected_papers = {
    "attention": "attention_is_all_you_need",
    "moshi": "moshi",
    "dynamic_chunking": "dynamic_chunking"
}

rows = []

for _, question in questions.iterrows():
    qid = question["question_id"]
    category = question["category"]
    query = question["question"]

    tfidf_results = tfidf.retrieve(query, top_k=3)
    hybrid_results = hybrid.retrieve(query, top_k=3)

    expected = expected_papers.get(category)

    tfidf_papers = [r["paper_id"] for r in tfidf_results]
    hybrid_papers = [r["paper_id"] for r in hybrid_results]

    rows.append({
        "question_id": qid,
        "question": query,
        "category": category,
        "expected_paper": expected,
        "tfidf_top3": ";".join(tfidf_papers),
        "hybrid_top3": ";".join(hybrid_papers),
        "tfidf_hit_at_3": (
            expected in tfidf_papers if expected else None
        ),
        "hybrid_hit_at_3": (
            expected in hybrid_papers if expected else None
        )
    })

results = pd.DataFrame(rows)
results.to_csv("experiments/retrieval_comparison.csv", index=False)

in_scope = results[results["expected_paper"].notna()]

print("\n--- Retrieval comparison ---")
print("In-scope questions:", len(in_scope))
print(
    "TF-IDF Hit@3:",
    in_scope["tfidf_hit_at_3"].sum(),
    "/",
    len(in_scope)
)
print(
    "Hybrid Hit@3:",
    in_scope["hybrid_hit_at_3"].sum(),
    "/",
    len(in_scope)
)

print("\n--- Per-question results ---")
print(
    results[
        ["question_id", "tfidf_hit_at_3", "hybrid_hit_at_3"]
    ].to_string(index=False)
)

print("\nSaved: experiments/retrieval_comparison.csv")
