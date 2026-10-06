"""
Simulates N chat users hitting the chatbot at the same time.

Each user sends a series of messages with think time in between, the way
real users do. Records outcome, time-to-first-token and total time per request.

Usage:
    python run.py --users 20 --messages 5 --label naive
"""
import argparse
import asyncio
import json
import random
import statistics
import time

import httpx

MESSAGES = [
    "Hi, I need help with my order.",
    "It was supposed to arrive yesterday but tracking shows nothing.",
    "Can you check if there is a delay in my region?",
    "What are my options if it does not arrive this week?",
    "Okay, how do I request a refund?",
    "Thanks. Is there anything else I should know?",
]


async def user_session(client, args, user_id: str, results: list):
    for i in range(args.messages):
        message = MESSAGES[i % len(MESSAGES)]
        start = time.perf_counter()
        record = {"user": user_id, "turn": i + 1, "status": None, "ttft_s": None, "total_s": None}
        try:
            async with client.stream("POST", args.url, json={"user_id": user_id, "message": message}) as r:
                if r.status_code != 200:
                    await r.aread()
                    record["status"] = "rate_limited" if r.status_code == 429 else f"http_{r.status_code}"
                else:
                    async for _ in r.aiter_bytes():
                        if record["ttft_s"] is None:
                            record["ttft_s"] = time.perf_counter() - start
                    record["status"] = "success"
        except httpx.HTTPError as e:
            record["status"] = f"error_{type(e).__name__}"
        record["total_s"] = time.perf_counter() - start
        results.append(record)
        await asyncio.sleep(random.uniform(args.think_min, args.think_max))


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(int(len(values) * p), len(values) - 1)]


def summarize(results, label, elapsed, provider_stats):
    total = len(results)
    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    ttfts = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    success = by_status.get("success", 0)

    print(f"\n=== {label} ===")
    print(f"Requests:        {total} in {elapsed:.1f}s")
    print(f"Success:         {success} ({success / total:.0%})")
    for status, count in sorted(by_status.items()):
        if status != "success":
            print(f"{status + ':':<17}{count} ({count / total:.0%})")
    if ttfts:
        print(f"TTFT p50 / p95:  {statistics.median(ttfts):.2f}s / {percentile(ttfts, 0.95):.2f}s")
    if provider_stats:
        print(f"Provider:        {provider_stats}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/chat")
    p.add_argument("--provider", default="http://localhost:8001")
    p.add_argument("--users", type=int, default=20)
    p.add_argument("--messages", type=int, default=5)
    p.add_argument("--think-min", type=float, default=3)
    p.add_argument("--think-max", type=float, default=10)
    p.add_argument("--label", default="run")
    args = p.parse_args()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        await client.post(f"{args.provider}/reset")
        await client.post(args.url.rsplit("/", 1)[0] + "/reset")

        results: list[dict] = []
        start = time.perf_counter()
        # All users start at once: the worst-case burst.
        await asyncio.gather(*(user_session(client, args, f"user-{n}", results) for n in range(args.users)))
        elapsed = time.perf_counter() - start

        provider_stats = (await client.get(f"{args.provider}/stats")).json()

    summarize(results, args.label, elapsed, provider_stats)
    out = f"results_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "elapsed_s": elapsed, "provider": provider_stats, "requests": results}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
