"""
Mock OpenAI-style chat completions provider.

Enforces RPM and TPM over a rolling window, returns 429s with OpenAI-like
rate-limit headers, and streams tokens with realistic latency. Free and
deterministic, so load tests are repeatable.
"""
import asyncio
import json
import os
import random
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

RPM_LIMIT = int(os.getenv("RPM_LIMIT", "500"))
TPM_LIMIT = int(os.getenv("TPM_LIMIT", "30000"))
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "60"))
TTFT_MS = (int(os.getenv("TTFT_MIN_MS", "300")), int(os.getenv("TTFT_MAX_MS", "800")))
TOKENS_PER_SECOND = float(os.getenv("TOKENS_PER_SECOND", "60"))
OUTPUT_TOKENS = (int(os.getenv("OUTPUT_MIN", "150")), int(os.getenv("OUTPUT_MAX", "450")))

app = FastAPI()

# Rolling window of (timestamp, tokens_charged) for admitted requests.
window: deque[tuple[float, int]] = deque()
lock = asyncio.Lock()
stats = {"admitted": 0, "rejected_rpm": 0, "rejected_tpm": 0}


def estimate_input_tokens(messages: list[dict]) -> int:
    # Rough heuristic: ~4 characters per token.
    return sum(len(m.get("content", "")) for m in messages) // 4


def prune(now: float) -> None:
    while window and now - window[0][0] >= WINDOW_SECONDS:
        window.popleft()


def rate_limit_headers(now: float, used_requests: int, used_tokens: int) -> dict:
    reset = WINDOW_SECONDS - (now - window[0][0]) if window else 0.0
    return {
        "x-ratelimit-limit-requests": str(RPM_LIMIT),
        "x-ratelimit-remaining-requests": str(max(RPM_LIMIT - used_requests, 0)),
        "x-ratelimit-limit-tokens": str(TPM_LIMIT),
        "x-ratelimit-remaining-tokens": str(max(TPM_LIMIT - used_tokens, 0)),
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
    messages = body.get("messages", [])
    max_tokens = int(body.get("max_tokens", 1000))

    # Providers count the request's reserved budget, not just what it ends up generating.
    # Charging input estimate + max_tokens approximates this.
    charge = estimate_input_tokens(messages) + max_tokens

    async with lock:
        now = time.monotonic()
        prune(now)
        used_requests = len(window)
        used_tokens = sum(t for _, t in window)
        headers = rate_limit_headers(now, used_requests, used_tokens)
        retry_after = WINDOW_SECONDS - (now - window[0][0]) if window else 1.0

        if used_requests + 1 > RPM_LIMIT:
            stats["rejected_rpm"] += 1
            return too_many_requests("requests per minute", headers, retry_after)
        if used_tokens + charge > TPM_LIMIT:
            stats["rejected_tpm"] += 1
            return too_many_requests("tokens per minute", headers, retry_after)

        window.append((now, charge))
        stats["admitted"] += 1
        headers = rate_limit_headers(now, used_requests + 1, used_tokens + charge)

    n_out = min(random.randint(*OUTPUT_TOKENS), max_tokens)

    async def stream():
        await asyncio.sleep(random.randint(*TTFT_MS) / 1000)
        for i in range(n_out):
            chunk = {"choices": [{"delta": {"content": f"tok{i} "}, "index": 0}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(1 / TOKENS_PER_SECOND)
        usage = {"prompt_tokens": charge - max_tokens, "completion_tokens": n_out}
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.get("/stats")
async def get_stats():
    return {**stats, "limits": {"rpm": RPM_LIMIT, "tpm": TPM_LIMIT, "window_s": WINDOW_SECONDS}}


@app.post("/reset")
async def reset():
    window.clear()
    for k in stats:
        stats[k] = 0
    return {"ok": True}
