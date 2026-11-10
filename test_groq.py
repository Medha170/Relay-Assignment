
import asyncio
import time

from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


async def main():
    async with AsyncGroq(max_retries=0) as client:
        start = time.perf_counter()
        first_token_time = None
        answer = ""
        usage = None

        stream = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": "Explain what a rate limit is in one sentence."
                }
            ],
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content

                if content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    answer += content
                    print(content, end="", flush=True)

        end = time.perf_counter()

        print("\n\n--- Measurements ---")
        print("TTFT:", round((first_token_time - start) * 1000, 2), "ms")
        print("Total latency:", round((end - start) * 1000, 2), "ms")

        if usage:
            print("Prompt tokens:", usage.prompt_tokens)
            print("Completion tokens:", usage.completion_tokens)
            print("Total tokens:", usage.total_tokens)


asyncio.run(main())
