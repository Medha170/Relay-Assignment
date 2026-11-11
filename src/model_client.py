
from dotenv import load_dotenv
from groq import AsyncGroq
import asyncio

from groq import RateLimitError

load_dotenv()


class GroqModelClient:
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        self.client = AsyncGroq(max_retries=0)


    async def stream_generate(self, prompt):
        max_retries = 2

        for attempt in range(max_retries + 1):
            yielded_any = False

            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=500,
                    stream=True,
                )

                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    content = chunk.choices[0].delta.content

                    if content:
                        yielded_any = True
                        yield content

                return

            except RateLimitError as error:
                if yielded_any or attempt == max_retries:
                    raise

                retry_after = error.response.headers.get("retry-after")

                delay = (
                    float(retry_after)
                    if retry_after is not None
                    else 2 ** attempt
                )

                print(
                    f"\nRate limited. Retrying in {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                await asyncio.sleep(delay)

    async def close(self):
        await self.client.close()
