import asyncio

from dotenv import load_dotenv
from groq import AsyncGroq

from src.retriever import PaperRetriever

load_dotenv()

MODEL = "openai/gpt-oss-20b"


async def answer_question(question, retriever, client):
    passages = retriever.retrieve(question, top_k=5)

    context = ""

    for i, passage in enumerate(passages, 1):
        context += (
            f"\n[Source {i}]\n"
            f"Paper: {passage['title']}\n"
            f"arXiv: {passage['arxiv']}\n"
            f"Page: {passage['page']}\n"
            f"Passage: {passage['text']}\n"
        )

    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the supplied passages.

Rules:
- Cite every factual claim using [Source N].
- Never invent information or citations.
- If the passages do not contain enough information, say:
  "The provided papers do not contain enough information to answer this question."
- Be concise and clear.

Retrieved passages:
{context}

Question: {question}
"""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=500,
    )

    return response.choices[0].message.content, passages


async def main():
    retriever = PaperRetriever()

    async with AsyncGroq(max_retries=0) as client:
        question = input("Ask a question about the papers: ")

        answer, sources = await answer_question(
            question, retriever, client
        )

        print("\n--- Answer ---")
        print(answer)

        print("\n--- Retrieved Sources ---")
        for i, source in enumerate(sources, 1):
            print(
                f"[Source {i}] {source['title']} "
                f"(arXiv:{source['arxiv']}, page {source['page']})"
            )


if __name__ == "__main__":
    asyncio.run(main())