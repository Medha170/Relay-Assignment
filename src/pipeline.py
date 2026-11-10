
import asyncio

from src.retriever import PaperRetriever
from src.model_client import GroqModelClient
import os
from src.standin import StandInModelClient


MIN_RELEVANCE_SCORE = 0.10
TOP_K = 3


class ResearchPipeline:
    def __init__(self, model_client, retriever=None):
        self.retriever = retriever or PaperRetriever()
        self.model_client = model_client

    def build_prompt(self, question, passages):
        context = ""

        for i, passage in enumerate(passages, 1):
            context += (
                f"\n[Source {i}]\n"
                f"Paper: {passage['title']}\n"
                f"arXiv: {passage['arxiv']}\n"
                f"Page: {passage['page']}\n"
                f"Passage: {passage['text']}\n"
            )

        return f"""
You are a research paper assistant.

Answer using ONLY the retrieved passages below.

Rules:
- Cite every factual claim using [Source N].
- Never invent facts or citations.
- If the passages do not support an answer, say:
  "The provided papers do not contain enough information to answer this question."
- Be concise but complete.
- Keep the answer concise: aim for 3-5 sentences.
- Complete every sentence; do not stop mid-explanation.
- Do not include details that are not supported by the passages.

Retrieved passages:
{context}

Question: {question}
"""

    async def answer(self, question):
        # Step 1: Retrieve relevant passages
        passages = self.retriever.retrieve(question, top_k=TOP_K)

        # Step 2: Check retrieval relevance
        if not passages or passages[0]["score"] < MIN_RELEVANCE_SCORE:
            yield (
                "The provided papers do not contain enough information "
                "to answer this question."
            )
            return

        # Step 3: Build a grounded prompt
        prompt = self.build_prompt(question, passages)

        # Step 4: Generate and stream the answer
        async for piece in self.model_client.stream_generate(prompt):
            yield piece

    async def close(self):
        await self.model_client.close()


async def main():
    backend = os.getenv("MODEL_BACKEND", "groq").lower()

    if backend == "standin":
        model_client = StandInModelClient()
    elif backend == "groq":
        model_client = GroqModelClient()
    else:
        raise ValueError(f"Unknown model backend: {backend}")

    pipeline = ResearchPipeline(model_client)

    try:
        question = input("Ask a question about the papers: ")

        print("\n--- Answer ---")

        async for piece in pipeline.answer(question):
            print(piece, end="", flush=True)

        print()

    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
