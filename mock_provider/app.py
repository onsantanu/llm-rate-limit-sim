"""
Mock OpenAI-style chat completions provider.

Enforces RPM and TPM per model over a rolling window, returns 429s with
OpenAI-like rate-limit headers, and streams tokens with realistic latency.
Free and deterministic, so load tests are repeatable.

Set SEED to make reply lengths and latencies repeatable: each end user's n-th
admitted call to a model always gets the same values, whatever the arrival order.

Chaos controls for resilience testing: POST /chaos sets a random server-error
rate, POST /outage makes one model return 503s for a number of seconds.

Two models, each with its own separate limit pool (as real providers do):
  - mock-gpt:  the main chat model (tight limits)
  - mock-mini: a smaller, cheaper model (higher limits), e.g. for summaries
"""
import asyncio
import json
import os
import random
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

LIMITS = {
    "mock-gpt": (int(os.getenv("RPM_LIMIT", "500")), int(os.getenv("TPM_LIMIT", "30000"))),
    "mock-mini": (int(os.getenv("MINI_RPM_LIMIT", "500")), int(os.getenv("MINI_TPM_LIMIT", "200000"))),
}
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "60"))
TTFT_MS = (int(os.getenv("TTFT_MIN_MS", "300")), int(os.getenv("TTFT_MAX_MS", "800")))
TOKENS_PER_SECOND = float(os.getenv("TOKENS_PER_SECOND", "60"))
OUTPUT_TOKENS = (int(os.getenv("OUTPUT_MIN", "150")), int(os.getenv("OUTPUT_MAX", "450")))
SEED = os.getenv("SEED")

app = FastAPI()

# Per model: rolling window of (timestamp, tokens_charged) for admitted requests.
windows: dict[str, deque[tuple[float, int]]] = {m: deque() for m in LIMITS}
stats = {m: {"admitted": 0, "rejected_rpm": 0, "rejected_tpm": 0, "errors": 0} for m in LIMITS}
lock = asyncio.Lock()
calls_per_user: dict[tuple[str, str], int] = defaultdict(int)  # (model, user) -> admitted calls
chaos = {"error_rate": 0.0}
outage_until: dict[str, float] = {}


def rng_for(model: str, user: str, n: int) -> random.Random:
    if SEED is None:
        return random.Random()
    return random.Random(f"{SEED}:{model}:{user}:{n}")


def estimate_input_tokens(messages: list[dict]) -> int:
    # Rough heuristic: ~4 characters per token.
    return sum(len(m.get("content", "")) for m in messages) // 4


def prune(window: deque, now: float) -> None:
    while window and now - window[0][0] >= WINDOW_SECONDS:
        window.popleft()


def rate_limit_headers(model: str, now: float, used_requests: int, used_tokens: int) -> dict:
    rpm, tpm = LIMITS[model]
    window = windows[model]
    reset = WINDOW_SECONDS - (now - window[0][0]) if window else 0.0
    return {
        "x-ratelimit-limit-requests": str(rpm),
        "x-ratelimit-remaining-requests": str(max(rpm - used_requests, 0)),
        "x-ratelimit-limit-tokens": str(tpm),
        "x-ratelimit-remaining-tokens": str(max(tpm - used_tokens, 0)),
        "x-ratelimit-reset-tokens": f"{reset:.2f}s",
    }


def too_many_requests(kind: str, headers: dict, retry_after: float) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        headers={**headers, "retry-after": f"{retry_after:.2f}"},
        content={"error": {"type": "rate_limit_exceeded", "message": f"Rate limit reached ({kind})."}},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "mock-gpt")
    if model not in LIMITS:
        return JSONResponse(status_code=404, content={"error": {"type": "model_not_found", "message": model}})
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 1000))
    rpm, tpm = LIMITS[model]
    window = windows[model]

    # Failures happen before admission, so they don't use rate-limit budget.
    if time.monotonic() < outage_until.get(model, 0):
        stats[model]["errors"] += 1
        return JSONResponse(status_code=503, content={"error": {"type": "service_unavailable", "message": "Model overloaded."}})
    if random.random() < chaos["error_rate"]:
        stats[model]["errors"] += 1
        return JSONResponse(status_code=500, content={"error": {"type": "server_error", "message": "Internal error."}})

    # Providers count the request's reserved budget, not just what it ends up generating.
    # Charging input estimate + max_tokens approximates this.
    charge = estimate_input_tokens(messages) + max_tokens

    async with lock:
        now = time.monotonic()
        prune(window, now)
        used_requests = len(window)
        used_tokens = sum(t for _, t in window)
        headers = rate_limit_headers(model, now, used_requests, used_tokens)
        retry_after = WINDOW_SECONDS - (now - window[0][0]) if window else 1.0

        if used_requests + 1 > rpm:
            stats[model]["rejected_rpm"] += 1
            return too_many_requests("requests per minute", headers, retry_after)
        if used_tokens + charge > tpm:
            stats[model]["rejected_tpm"] += 1
            return too_many_requests("tokens per minute", headers, retry_after)

        window.append((now, charge))
        stats[model]["admitted"] += 1
        headers = rate_limit_headers(model, now, used_requests + 1, used_tokens + charge)
        user = body.get("user", "anonymous")
        calls_per_user[(model, user)] += 1
        rng = rng_for(model, user, calls_per_user[(model, user)])

    n_out = min(rng.randint(*OUTPUT_TOKENS), max_tokens)
    ttft_s = rng.randint(*TTFT_MS) / 1000
    usage = {"prompt_tokens": charge - max_tokens, "completion_tokens": n_out}

    if not body.get("stream"):
        await asyncio.sleep(ttft_s + n_out / TOKENS_PER_SECOND)
        content = "".join(f"tok{i} " for i in range(n_out))
        return JSONResponse(
            headers=headers,
            content={"choices": [{"message": {"role": "assistant", "content": content}, "index": 0}], "usage": usage},
        )

    async def stream():
        await asyncio.sleep(ttft_s)
        for i in range(n_out):
            chunk = {"choices": [{"delta": {"content": f"tok{i} "}, "index": 0}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(1 / TOKENS_PER_SECOND)
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.get("/stats")
async def get_stats():
    return {m: {**stats[m], "rpm": LIMITS[m][0], "tpm": LIMITS[m][1]} for m in LIMITS}


@app.post("/reset")
async def reset():
    for m in LIMITS:
        windows[m].clear()
        stats[m] = {"admitted": 0, "rejected_rpm": 0, "rejected_tpm": 0, "errors": 0}
    calls_per_user.clear()
    chaos["error_rate"] = 0.0
    outage_until.clear()
    return {"ok": True}


@app.post("/chaos")
async def set_chaos(request: Request):
    chaos["error_rate"] = float((await request.json())["error_rate"])
    return chaos


@app.post("/outage")
async def start_outage(request: Request):
    body = await request.json()
    outage_until[body["model"]] = time.monotonic() + float(body["seconds"])
    return {"ok": True}
