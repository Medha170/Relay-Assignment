
import asyncio
import random
import time
from collections import deque


class StandInRateLimitError(Exception):
    def __init__(self, retry_after):
        self.retry_after = retry_after
        super().__init__(
            f"Stand-in rate limit reached. Retry after {retry_after:.2f}s"
        )


class StandInServiceError(Exception):
    pass


class StandInTimeoutError(Exception):
    pass


class StandInModelClient:
    def __init__(
        self,
        base_latency_ms=500,
        chunk_interval_ms=30,
        chunk_size=25,
        failure_probability=0.0,
        rate_limit_requests=30,
        rate_limit_window_seconds=60,
        max_concurrency=5,
        overload_delay_ms=1000,
        timeout_ms=None,
        seed=42,
    ):
        self.base_latency_ms = base_latency_ms
        self.chunk_interval_ms = chunk_interval_ms
        self.chunk_size = chunk_size
        self.failure_probability = failure_probability

        self.rate_limit_requests = rate_limit_requests
        self.rate_limit_window_seconds = rate_limit_window_seconds

        self.max_concurrency = max_concurrency
        self.overload_delay_ms = overload_delay_ms
        self.timeout_ms = timeout_ms

        self.request_times = deque()
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self.random = random.Random(seed)

        self.response_text = (
            "This is a simulated response from the stand-in model. "
            "It does not perform actual language model inference. "
            "Its purpose is to help us test latency, streaming, "
            "failures, rate limits, and concurrent request handling."
        )

    async def _begin_request(self):
        async with self.lock:
            now = time.monotonic()

            while (
                self.request_times
                and now - self.request_times[0]
                >= self.rate_limit_window_seconds
            ):
                self.request_times.popleft()

            if len(self.request_times) >= self.rate_limit_requests:
                retry_after = (
                    self.rate_limit_window_seconds
                    - (now - self.request_times[0])
                )
                raise StandInRateLimitError(retry_after)

            self.request_times.append(now)
            self.active_requests += 1

            return self.active_requests > self.max_concurrency

    async def _end_request(self):
        async with self.lock:
            self.active_requests -= 1

    async def _wait(self, delay_ms):
        try:
            sleep = asyncio.sleep(delay_ms / 1000)

            if self.timeout_ms is None:
                await sleep
            else:
                await asyncio.wait_for(
                    sleep,
                    timeout=self.timeout_ms / 1000
                )

        except asyncio.TimeoutError:
            raise StandInTimeoutError("Stand-in request timed out")

    async def stream_generate(self, prompt):
        overloaded = await self._begin_request()

        try:
            delay = self.base_latency_ms

            if overloaded:
                delay += self.overload_delay_ms

            await self._wait(delay)

            if self.random.random() < self.failure_probability:
                raise StandInServiceError(
                    "Simulated stand-in service failure"
                )

            for start in range(0, len(self.response_text), self.chunk_size):
                if start > 0:
                    await asyncio.sleep(
                        self.chunk_interval_ms / 1000
                    )

                yield self.response_text[
                    start:start + self.chunk_size
                ]

        finally:
            await self._end_request()

    async def close(self):
        pass
