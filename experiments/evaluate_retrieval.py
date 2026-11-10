import pandas as pd

from src.retriever import PaperRetriever

questions = pd.read_csv("experiments/quality_eval.csv")

retriever = PaperRetriever()

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