
from src.retriever import PaperRetriever

retriever = PaperRetriever()

questions = [
    "What is the current population of Bengaluru?",
    "Who won the 2024 Olympic mens 100m final?",
    "Why is the dot product scaled by the square root of the key dimension?"
]

for question in questions:
    print("\nQUESTION:", question)

    passages = retriever.retrieve(question, top_k=3)

    for i, passage in enumerate(passages, 1):
        print(
            "\nRank", i,
            "|", passage["title"],
            "| Page", passage["page"],
            "| Score", round(passage["score"], 4)
        )

        print(passage["text"][:450].replace("\n", " "))
