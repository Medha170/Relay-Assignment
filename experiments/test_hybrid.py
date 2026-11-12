
from src.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()

questions = [
    "How does scaled dot-product attention work?",
    "What role does the Mimi audio codec play in Moshi?",
    "What is the current population of Bengaluru?"
]

for question in questions:
    print(f"\nQUESTION: {question}")

    results = retriever.retrieve(question, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(
            f"\nRank {rank} | {result['paper_id']} "
            f"| Page {result['page']}"
        )
        print(f"RRF: {result['rrf_score']:.5f}")
        print(f"TF-IDF rank: {result['lexical_rank']}")
        print(f"Semantic rank: {result['semantic_rank']}")
        print(result["text"][:250])
