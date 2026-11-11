import pandas as pd

from src.retriever import PaperRetriever

questions = pd.read_csv("experiments/quality_eval.csv")

retriever = PaperRetriever()

retrieval_results = []

for _, row in questions.iterrows():
    question = row["question"]

    passages = retriever.retrieve(question, top_k=3)

    print(f"\nQuestion: {question}")

    if not passages:
        print("No relevant passages found.")
        continue

    for i, passage in enumerate(passages, 1):
        print(
            f"Source {i}: {passage['title']} | "
            f"Page {passage['page']} | "
            f"Score: {passage['score']:.4f}"
        )

    retrieval_results.append({
        "question": question,
        "top_score": passages[0]["score"] if passages else 0,
        "top_paper": passages[0]["title"] if passages else "",
        "top_page": passages[0]["page"] if passages else "",
        "top_3_scores": [
            round(p["score"], 4) for p in passages
        ],
        "top_3_papers": [
            p["title"] for p in passages
        ]
    })

results_df = pd.DataFrame(retrieval_results)

results_df.to_csv(
    "experiments/retrieval_eval.csv",
    index=False
)

print("\nSaved retrieval results to experiments/retrieval_eval.csv")