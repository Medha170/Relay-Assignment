
import asyncio
import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline import ResearchPipeline
from src.standin import (
    StandInModelClient,
    StandInRateLimitError,
    StandInTimeoutError,
    StandInServiceError,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "experiments" / "results.csv"

QUESTION = "How does the Transformer use attention?"

CSV_FIELDS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


async def run_one(pipeline, scenario):
    start_time = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    first_token_ms = ""

    try:
        async for piece in pipeline.answer(QUESTION):
            if first_token_ms == "":
                first_token_ms = round(
                    (time.perf_counter() - start) * 1000, 2
                )

        status = "ok"

    except StandInRateLimitError:
        status = "rate_limited"

    except StandInTimeoutError:
        status = "timeout"

    except StandInServiceError:
        status = "error"

    except Exception as error:
        print(f"Unexpected error: {error}")
        status = "error"

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "request_id": str(uuid.uuid4()),
        "scenario": scenario,
        "source": "standin",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "tfidf_top5_min0.05",
        "answer_grade": "",
    }


async def main():
    client = StandInModelClient(
        base_latency_ms=300,
        chunk_interval_ms=10,
        max_concurrency=3,
        overload_delay_ms=500,
        rate_limit_requests=100,
        failure_probability=0,
    )

    pipeline = ResearchPipeline(client)
    results = []

    try:
        # 1. Steady: one request after another
        print("Running steady scenario...")
        for _ in range(5):
            results.append(await run_one(pipeline, "steady"))

        # 2. Burst: ten requests launched together
        print("Running burst scenario...")
        burst = await asyncio.gather(
            *[
                run_one(pipeline, "burst")
                for _ in range(10)
            ]
        )
        results.extend(burst)

        # 3. Sustained overload: requests launched every 100 ms
        print("Running sustained overload scenario...")
        tasks = []

        for _ in range(20):
            tasks.append(
                asyncio.create_task(
                    run_one(pipeline, "sustained_overload")
                )
            )
            await asyncio.sleep(0.1)

        results.extend(await asyncio.gather(*tasks))

    finally:
        await pipeline.close()

    # Append results so previous experiment runs are preserved.
    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")
    print("\nScenario summary:")

    for scenario in ("steady", "burst", "sustained_overload"):
        rows = [r for r in results if r["scenario"] == scenario]
        successful = sum(r["status"] == "ok" for r in rows)

        print(
            f"{scenario}: {successful}/{len(rows)} successful"
        )


if __name__ == "__main__":
    asyncio.run(main())
