
QUESTIONS = [
    # Attention Is All You Need
    ("attention", "How does scaled dot-product attention work?"),
    ("attention", "Why is the dot product scaled by the square root of the key dimension?"),
    ("attention", "What is the purpose of multi-head attention?"),
    ("attention", "How does the Transformer encode positional information?"),

    # Moshi
    ("moshi", "What role does the Mimi audio codec play in Moshi?"),
    ("moshi", "How does Moshi support full-duplex dialogue?"),
    ("moshi", "What is Inner Monologue, and why is it useful?"),
    ("moshi", "How does Moshi generate speech in real time?"),

    # Dynamic Chunking
    ("dynamic_chunking", "What problem is dynamic chunking intended to address?"),
    ("dynamic_chunking", "How does dynamic chunking differ from fixed-size chunking?"),
    ("dynamic_chunking", "What is hierarchical sequence modeling?"),
    ("dynamic_chunking", "What evidence does the paper provide for its approach?"),

    # Questions outside the papers
    ("out_of_scope", "What is the capital of France?"),
    ("out_of_scope", "Who won the 2024 Olympic men's 100m final?"),
    ("out_of_scope", "What is the current population of Bengaluru?"),
]

QUESTIONS = [QUESTIONS[4], QUESTIONS[8], QUESTIONS[12]]

import asyncio
import csv
from pathlib import Path

from src.pipeline import ResearchPipeline
from src.model_client import GroqModelClient
from src.retriever import PaperRetriever


OUTPUT_PATH = Path("experiments/results_v2.csv")


async def main():
    retriever = PaperRetriever()
    model = GroqModelClient(max_completion_tokens=500)
    pipeline = ResearchPipeline(model, retriever)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames = [
                    "request_id",
                    "scenario",
                    "source",
                    "start_time",
                    "first_token_ms",
                    "total_ms",
                    "status",
                    "retries",
                    "tokens_in",
                    "tokens_out",
                    "retrieval_setting",
                    "answer_grade",
                ]
            )
            writer.writeheader()

            for question_id, (category, question) in enumerate(
                QUESTIONS, start=1
            ):
                passages = retriever.retrieve(question, top_k=5)

                sources = [
                    {
                        "paper": p["title"],
                        "page": p["page"],
                        "score": round(p["score"], 4),
                    }
                    for p in passages
                ]

                answer = ""

                print(f"\n[{question_id}/15] {question}")

                model.last_metrics = {
                    "start_time": "",
                    "first_token_ms": None,
                    "total_ms": None,
                    "retries": 0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "status": "error",
                }

                try:
                    async for piece in pipeline.answer(question):
                        answer += piece
                        print(piece, end="", flush=True)

                except Exception as error:
                    answer = f"ERROR: {error}"
                    print(f"\nRequest error: {type(error).__name__}: {error}")

                metrics = model.last_metrics

                print(f"\nStatus: {metrics['status']}")
                print(f"TTFT: {metrics['first_token_ms']} ms")
                print(f"Total: {metrics['total_ms']} ms")
                print(f"Retries: {metrics['retries']}")

                writer.writerow({
                    "request_id": question_id,
                    "scenario": "quality_evaluation",
                    "source": "real",
                    "start_time": metrics["start_time"],
                    "first_token_ms": metrics["first_token_ms"],
                    "total_ms": metrics["total_ms"],
                    "status": metrics["status"],
                    "retries": metrics["retries"],
                    "tokens_in": metrics["tokens_in"],
                    "tokens_out": metrics["tokens_out"],
                    "retrieval_setting": "tfidf_top3",
                    "answer_grade": "",
                })

                file.flush()

                # Avoid sending requests too quickly.
                await asyncio.sleep(2)

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
