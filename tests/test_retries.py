
import asyncio
import httpx
from types import SimpleNamespace
from groq import RateLimitError

from src.model_client import GroqModelClient


def make_rate_limit_error():
    response = httpx.Response(
        429,
        headers={"retry-after": "0"},
        request=httpx.Request("POST", "https://example.com")
    )
    return RateLimitError(
        "Rate limited",
        response=response,
        body=None
    )


class FakeCompletions:
    def __init__(self, fail_count):
        self.fail_count = fail_count
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1

        if self.calls <= self.fail_count:
            raise make_rate_limit_error()

        async def fake_stream():
            chunk = SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Test response")
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=3,
                    completion_tokens=2
                )
            )
            yield chunk

        return fake_stream()


async def test_retry_success():
    fake = FakeCompletions(fail_count=1)

    model = GroqModelClient.__new__(GroqModelClient)
    model.model = "fake-model"
    model.max_completion_tokens = 500
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=fake
        )
    )

    result = "".join([
        piece async for piece in model.stream_generate("test prompt")
    ])

    assert result == "Test response"
    assert fake.calls == 2

    print("PASS: Retry succeeded on second attempt")

    assert model.last_metrics["status"] == "ok"
    assert model.last_metrics["retries"] == 1
    assert model.last_metrics["first_token_ms"] is not None
    print("DEBUG TTFT:", model.last_metrics["first_token_ms"])
    assert model.last_metrics["total_ms"] is not None
    assert model.last_metrics["tokens_in"] == 3
    assert model.last_metrics["tokens_out"] == 2


async def test_retry_exhaustion():
    fake = FakeCompletions(fail_count=3)

    model = GroqModelClient.__new__(GroqModelClient)
    model.model = "fake-model"
    model.max_completion_tokens = 500
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=fake
        )
    )

    try:
        async for _ in model.stream_generate("test prompt"):
            pass
    except RateLimitError:
        assert fake.calls == 3
        print("PASS: Retry stopped after three attempts")
        assert model.last_metrics["status"] == "rate_limited"
        assert model.last_metrics["retries"] == 2
        assert model.last_metrics["total_ms"] is not None
    else:
        raise AssertionError("Expected RateLimitError was not raised")

async def test_generic_error():
    class FailingCompletions:
        async def create(self, **kwargs):
            raise RuntimeError("Simulated service failure")

    model = GroqModelClient.__new__(GroqModelClient)
    model.model = "fake-model"
    model.max_completion_tokens = 500
    model.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FailingCompletions()
        )
    )

    try:
        async for _ in model.stream_generate("test prompt"):
            pass
    except RuntimeError:
        assert model.last_metrics["status"] == "error"
        assert model.last_metrics["total_ms"] is not None
        print("PASS: Generic error recorded correctly")
    else:
        raise AssertionError("Expected RuntimeError was not raised")

async def main():
    await test_retry_success()
    await test_retry_exhaustion()
    await test_generic_error()


if __name__ == "__main__":
    asyncio.run(main())
