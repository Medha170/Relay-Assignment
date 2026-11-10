
import asyncio

from src.standin import (
    StandInModelClient,
    StandInRateLimitError,
    StandInServiceError,
    StandInTimeoutError,
)


async def collect_response(client):
    response = ""

    async for piece in client.stream_generate("test prompt"):
        response += piece

    return response


async def main():
    # Test 1: Normal response
    client = StandInModelClient(base_latency_ms=50)
    response = await collect_response(client)
    print("Normal response:", response[:50])

    # Test 2: Rate limit
    client = StandInModelClient(
        base_latency_ms=10,
        rate_limit_requests=1,
        rate_limit_window_seconds=10,
    )

    await collect_response(client)

    try:
        await collect_response(client)
    except StandInRateLimitError as error:
        print("Rate limit test: PASSED", error)

    # Test 3: Timeout
    client = StandInModelClient(
        base_latency_ms=500,
        timeout_ms=50,
    )

    try:
        await collect_response(client)
    except StandInTimeoutError as error:
        print("Timeout test: PASSED", error)

    # Test 4: Service failure
    client = StandInModelClient(
        failure_probability=1.0,
    )

    try:
        await collect_response(client)
    except StandInServiceError as error:
        print("Failure test: PASSED", error)


if __name__ == "__main__":
    asyncio.run(main())
