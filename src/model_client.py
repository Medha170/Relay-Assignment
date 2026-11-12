
from dotenv import load_dotenv
from groq import AsyncGroq
import asyncio
import time

from groq import RateLimitError
from datetime import datetime, timezone

load_dotenv()


class GroqModelClient:
    def __init__(self, model="openai/gpt-oss-20b", max_completion_tokens=500):
        self.model = model
        self.client = AsyncGroq(max_retries=0)
        self.last_metrics = {}
        self.max_completion_tokens = max_completion_tokens


    async def stream_generate(self, prompt):
        max_retries = 2

        request_start = time.perf_counter()

        self.last_metrics = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "first_token_ms": None,
            "total_ms": None,
            "retries": 0,
            "tokens_in": None,
            "tokens_out": None,
            "status": "error",
        }

        for attempt in range(max_retries + 1):
            yielded_any = False

            try:
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_completion_tokens=self.max_completion_tokens,
                    stream=True,
                )

                async for chunk in stream:
                    # Capture token usage if provided by the API.
                    if chunk.usage:
                        self.last_metrics["tokens_in"] = chunk.usage.prompt_tokens
                        self.last_metrics["tokens_out"] = chunk.usage.completion_tokens

                    if not chunk.choices:
                        continue

                    content = chunk.choices[0].delta.content

                    if content:
                        if not yielded_any:
                            self.last_metrics["first_token_ms"] = round(
                                (time.perf_counter() - request_start) * 1000, 2
                            )

                        yielded_any = True
                        yield content

                self.last_metrics["status"] = "ok"
                self.last_metrics["total_ms"] = round(
                    (time.perf_counter() - request_start) * 1000, 2
                )
                return

            except RateLimitError as error:
                if yielded_any or attempt == max_retries:
                    self.last_metrics["status"] = "rate_limited"
                    self.last_metrics["total_ms"] = round(
                        (time.perf_counter() - request_start) * 1000, 2
                    )
                    raise

                self.last_metrics["retries"] += 1

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

            except Exception:
                self.last_metrics["status"] = "error"
                self.last_metrics["total_ms"] = round(
                    (time.perf_counter() - request_start) * 1000, 2
                )
                raise

    async def close(self):
        await self.client.close()
