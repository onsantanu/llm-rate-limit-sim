"""
Simulates N chat users hitting the chatbot at the same time.

Each user sends a series of messages with think time in between, the way
real users do. Records outcome, time-to-first-token and total time per request.

Usage:
    python run.py --users 20 --messages 5 --label naive --seed 42

With --seed, each user's think times are repeatable (set SEED on the mock
provider too, so reply lengths are repeatable as well).

Chaos testing:
    --error-rate 0.05            5% of provider calls fail with a server error
    --outage mock-gpt:15:20      mock-gpt returns 503s from second 15 for 20 seconds
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

STATUS_NAMES = {429: "rate_limited", 503: "busy", 409: "in_progress"}


async def user_session(client, args, user_id: str, results: list):
    # One random stream per user, so concurrency can't change who gets which value.
    rng = random.Random(f"{args.seed}:{user_id}") if args.seed is not None else random.Random()
    for i in range(args.messages):
        message = MESSAGES[i % len(MESSAGES)]
        start = time.perf_counter()
        record = {"user": user_id, "turn": i + 1, "status": None, "model": None, "ttft_s": None, "total_s": None}
        try:
            async with client.stream("POST", args.url, json={"user_id": user_id, "message": message}) as r:
                if r.status_code != 200:
                    await r.aread()
                    record["status"] = STATUS_NAMES.get(r.status_code, f"http_{r.status_code}")
                else:
                    record["model"] = r.headers.get("x-model")
                    async for _ in r.aiter_bytes():
                        if record["ttft_s"] is None:
                            record["ttft_s"] = time.perf_counter() - start
                    record["status"] = "success"
        except httpx.HTTPError as e:
            record["status"] = f"error_{type(e).__name__}"
        record["total_s"] = time.perf_counter() - start
        results.append(record)
        await asyncio.sleep(rng.uniform(args.think_min, args.think_max))


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[min(int(len(values) * p), len(values) - 1)]


def summarize(results, label, elapsed, provider_stats, gateway_stats):
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
    models = {}
    for r in results:
        if r["model"]:
            models[r["model"]] = models.get(r["model"], 0) + 1
    if models:
        print(f"Served by:       {models}")
    if ttfts:
        print(f"TTFT p50 / p95:  {statistics.median(ttfts):.2f}s / {percentile(ttfts, 0.95):.2f}s")
    if gateway_stats:
        print(f"Gateway:         {gateway_stats}")
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
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--error-rate", type=float, default=0.0)
    p.add_argument("--outage", default=None, help="MODEL:START_S:DURATION_S")
    args = p.parse_args()

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        await client.post(f"{args.provider}/reset")
        app_base = args.url.rsplit("/", 1)[0]
        await client.post(app_base + "/reset")
        await client.post(f"{args.provider}/chaos", json={"error_rate": args.error_rate})

        async def schedule_outage():
            model, start_s, duration_s = args.outage.split(":")
            await asyncio.sleep(float(start_s))
            await client.post(f"{args.provider}/outage", json={"model": model, "seconds": float(duration_s)})

        outage_task = asyncio.create_task(schedule_outage()) if args.outage else None

        results: list[dict] = []
        start = time.perf_counter()
        # All users start at once: the worst-case burst.
        await asyncio.gather(*(user_session(client, args, f"user-{n}", results) for n in range(args.users)))
        elapsed = time.perf_counter() - start

        if outage_task:
            outage_task.cancel()
        provider_stats = (await client.get(f"{args.provider}/stats")).json()
        gateway_response = await client.get(app_base + "/stats")
        gateway_stats = gateway_response.json() if gateway_response.status_code == 200 else None

    summarize(results, args.label, elapsed, provider_stats, gateway_stats)
    out = f"results_{args.label}.json"
    with open(out, "w") as f:
        json.dump({"label": args.label, "seed": args.seed, "error_rate": args.error_rate, "outage": args.outage,
                   "elapsed_s": elapsed, "provider": provider_stats, "gateway": gateway_stats, "requests": results}, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
