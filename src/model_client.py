
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


class GroqModelClient:
    def __init__(self, model="openai/gpt-oss-20b"):
        self.model = model
        self.client = AsyncGroq(max_retries=0)

    async def stream_generate(self, prompt):
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
                yield content

    async def close(self):
        await self.client.close()
