
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

import asyncio
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import ResearchPipeline
from src.model_client import GroqModelClient
from src.retriever import PaperRetriever


OUTPUT_PATH = Path("experiments/quality_eval.csv")


async def main():
    retriever = PaperRetriever()
    model = GroqModelClient()
    pipeline = ResearchPipeline(model, retriever)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "question_id",
                    "category",
                    "question",
                    "answer",
                    "retrieved_sources",
                    "total_ms",
                    "answer_grade",
                    "notes",
                ],
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
                start = time.perf_counter()

                print(f"\n[{question_id}/15] {question}")

                try:
                    async for piece in pipeline.answer(question):
                        answer += piece
                        print(piece, end="", flush=True)

                    status = "completed"

                except Exception as error:
                    answer = f"ERROR: {error}"
                    status = "error"

                total_ms = round(
                    (time.perf_counter() - start) * 1000, 2
                )

                print(f"\nTime: {total_ms} ms")

                writer.writerow({
                    "question_id": question_id,
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "retrieved_sources": json.dumps(sources),
                    "total_ms": total_ms,
                    "answer_grade": "",
                    "notes": status,
                })

                file.flush()

                # Avoid sending requests too quickly.
                await asyncio.sleep(2)

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
