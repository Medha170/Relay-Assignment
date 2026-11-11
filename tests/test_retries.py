
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
                ]
            )
            yield chunk

        return fake_stream()


async def test_retry_success():
    fake = FakeCompletions(fail_count=1)

    model = GroqModelClient.__new__(GroqModelClient)
    model.model = "fake-model"
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


async def test_retry_exhaustion():
    fake = FakeCompletions(fail_count=3)

    model = GroqModelClient.__new__(GroqModelClient)
    model.model = "fake-model"
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
    else:
        raise AssertionError("Expected RateLimitError was not raised")


async def main():
    await test_retry_success()
    await test_retry_exhaustion()


if __name__ == "__main__":
    asyncio.run(main())
