"""
Gateway chatbot backend.

Step 1 - request shaping: spend fewer tokens per request so more requests fit
in the same provider limit.
  - Lean system prompt instead of a bloated one.
  - History trimmed to a token budget (most recent turns kept).
  - max_tokens sized to what replies actually need, because the provider
    reserves max_tokens against TPM whether or not they are generated.
  - Rolling summary: turns that fall outside the history budget are folded
    into a short summary instead of being dropped. Summaries are made in the
    background, after the reply has streamed, by a smaller model that has its
    own limit pool, so they don't compete with chat for capacity.

Step 2 - distributed limiter: never send a request the provider would reject.
  - Each request waits in a shared FIFO queue (Redis) until its tokens fit.
  - Waiting is bounded by WAIT_DEADLINE_S; after that the user gets a clean
    "busy" response instead of a raw provider error.
  - One in-flight message per user, so nobody can flood the shared budget.
  - A 429 that slips through (estimate drift) corrects the limiter and the
    request goes back to waiting, invisibly to the user.

Step 3 - resilience: fast replies even when the main model is full or failing.
  - Fallback model: if the chat model can't admit a request within
    PRIMARY_WAIT_S, the smaller model (separate limit pool) answers instead.
    When the limiter already knows the wait would be longer, the request skips
    the chat model immediately instead of waiting the full PRIMARY_WAIT_S.
  - Retries: server errors and dropped connections are retried with
    exponential backoff and jitter, only before the first token, within the deadline.
  - Circuit breaker: after repeated failures a model is skipped for a cooldown,
    so requests go straight to the fallback instead of waiting on a broken provider.
  - A stream that breaks mid-reply ends with an explicit notice, never silently.

Privacy: the raw user ID never leaves the system. Optionally (SEND_END_USER_ID),
a keyed hash of it is sent so the provider can attribute abuse to one end user
instead of the whole account.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import time
from collections import defaultdict

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from breaker import CircuitBreaker
from limiter import LLMRateLimiter

PROVIDER_URL = os.getenv("PROVIDER_URL", "http://localhost:8001/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mock-gpt")
# The smaller model serves both summaries and fallback replies, sharing its limit pool.
MINI_MODEL = os.getenv("MINI_MODEL", "mock-mini")
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "150"))
SUMMARY_DEADLINE_S = 30  # background work can wait longer than a user can
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
HISTORY_TOKEN_BUDGET = int(os.getenv("HISTORY_TOKEN_BUDGET", "600"))
WAIT_DEADLINE_S = float(os.getenv("WAIT_DEADLINE_S", "10"))
PRIMARY_WAIT_S = float(os.getenv("PRIMARY_WAIT_S", "3"))  # then switch to the fallback model
MAX_ATTEMPTS = 3  # per model, for server errors and dropped connections
SEND_END_USER_ID = os.getenv("SEND_END_USER_ID", "false").lower() == "true"
END_USER_ID_SECRET = os.getenv("END_USER_ID_SECRET", "")
if SEND_END_USER_ID and not END_USER_ID_SECRET:
    raise RuntimeError("SEND_END_USER_ID needs END_USER_ID_SECRET, or the ID could be reversed by guessing.")
INFLIGHT_TTL_S = 120  # safety net if a stream never finishes

# ~100 tokens: the same instructions, without repetition.
SYSTEM_PROMPT = (
    "You are a helpful support assistant for an online store. Answer order, delivery and refund "
    "questions concisely. Follow company policy: never promise dates you cannot verify, offer a "
    "refund only after confirming the order is lost or late beyond 7 days, and escalate anything "
    "involving payments to a human agent. Keep answers under 150 words."
)

SUMMARY_PROMPT = (
    "Update the conversation summary with the new turns. Keep facts the assistant will need later: "
    "names, order and ticket numbers, dates, decisions made, and open questions. Drop small talk. "
    "Reply with the updated summary only, under 120 words."
)

app = FastAPI()
# Drop idle connections before the provider does: requests now wait in a queue,
# so pooled connections sit idle longer and could be reused just as they close.
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0), limits=httpx.Limits(keepalive_expiry=2.0))
history: dict[str, list[dict]] = defaultdict(list)   # turns not yet folded into the summary
summaries: dict[str, str] = {}
summarizing: set[str] = set()      # one summary job per user at a time
background_tasks: set[asyncio.Task] = set()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "60"))
limiter = LLMRateLimiter(
    redis_url=REDIS_URL,
    model=CHAT_MODEL,
    rpm=int(os.getenv("RPM_LIMIT", "500")),
    tpm=int(os.getenv("TPM_LIMIT", "30000")),
    window_s=WINDOW_SECONDS,
)
mini_limiter = LLMRateLimiter(
    redis_url=REDIS_URL,
    model=MINI_MODEL,
    rpm=int(os.getenv("MINI_RPM_LIMIT", "500")),
    tpm=int(os.getenv("MINI_TPM_LIMIT", "200000")),
    window_s=WINDOW_SECONDS,
)
chat_breaker = CircuitBreaker(limiter.r, CHAT_MODEL)
mini_breaker = CircuitBreaker(limiter.r, MINI_MODEL)
stats = {"fallbacks": 0, "primary_skipped": 0, "retries": 0, "breaker_trips": 0, "interrupted_streams": 0}


class ChatRequest(BaseModel):
    user_id: str
    message: str


def end_user_fields(user_id: str) -> dict:
    """Pseudonymous end-user ID for the provider: stable per user, meaningless without our secret.

    Field name follows OpenAI's `user` parameter; check your provider's docs for the current one.
    """
    if not SEND_END_USER_ID:
        return {}
    digest = hmac.new(END_USER_ID_SECRET.encode(), user_id.encode(), hashlib.sha256).hexdigest()
    return {"user": digest[:32]}


def estimate_tokens(text: str) -> int:
    # ~4 characters per token; use a real tokenizer (e.g. tiktoken) in production.
    return len(text) // 4 + 1


def trim_history(turns: list[dict], budget: int) -> list[dict]:
    """Keep the most recent turns that fit within the token budget."""
    kept, used = [], 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn["content"])
        if used + cost > budget and kept:
            break
        kept.append(turn)
        used += cost
    return list(reversed(kept))


def provider_charge(messages: list[dict], max_tokens: int) -> int:
    # Must match how the provider counts usage, or the limiter drifts from reality.
    return sum(len(m["content"]) for m in messages) // 4 + max_tokens


def build_messages(user_id: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if user_id in summaries:
        messages.append({"role": "system", "content": f"Summary of the earlier conversation: {summaries[user_id]}"})
    messages.extend(trim_history(history[user_id], HISTORY_TOKEN_BUDGET))
    return messages


async def summarize_older_turns(user_id: str) -> None:
    """Fold turns that no longer fit the history budget into the user's summary."""
    turns = history[user_id]
    older = turns[: len(turns) - len(trim_history(turns, HISTORY_TOKEN_BUDGET))]
    if not older:
        return
    transcript = "\n".join(f"{t['role']}: {t['content']}" for t in older)
    prompt = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"Current summary: {summaries.get(user_id, '(none)')}\n\nNew turns:\n{transcript}"},
    ]
    if not await mini_limiter.acquire(provider_charge(prompt, SUMMARY_MAX_TOKENS), SUMMARY_DEADLINE_S):
        return  # no capacity now; the turns stay in history and we try again after the next reply
    payload = {"model": MINI_MODEL, "messages": prompt, "max_tokens": SUMMARY_MAX_TOKENS, **end_user_fields(user_id)}
    response = await client.post(PROVIDER_URL, json=payload)
    await mini_limiter.sync_from_headers(response.headers)
    if response.status_code != 200:
        return
    summaries[user_id] = response.json()["choices"][0]["message"]["content"]
    # New turns are only ever appended, so the summarized ones are still at the front.
    del history[user_id][: len(older)]


async def run_summary_job(user_id: str) -> None:
    try:
        await summarize_older_turns(user_id)
    except Exception:
        logging.exception("Summary job failed for %s", user_id)
    finally:
        summarizing.discard(user_id)


def schedule_summary(user_id: str) -> None:
    if user_id in summarizing:
        return
    summarizing.add(user_id)
    task = asyncio.create_task(run_summary_job(user_id))
    background_tasks.add(task)  # keep a reference so the task isn't garbage-collected
    task.add_done_callback(background_tasks.discard)


async def open_stream(
    model: str, model_limiter: LLMRateLimiter, breaker: CircuitBreaker, payload: dict, charge: int, stop_at: float
) -> httpx.Response | None:
    """Reserve capacity on `model` and open a reply stream, retrying transient failures.

    Returns None if the model can't be used before `stop_at` (busy, circuit open, or failing).
    """
    payload = {**payload, "model": model}
    attempt = 0
    while True:
        if await breaker.is_open():
            return None
        if not await model_limiter.acquire(charge, max(stop_at - time.monotonic(), 0)):
            return None
        try:
            upstream = await client.send(client.build_request("POST", PROVIDER_URL, json=payload), stream=True)
        except httpx.TransportError:
            upstream = None
        if upstream is not None:
            await model_limiter.sync_from_headers(upstream.headers)
            if upstream.status_code == 200:
                await breaker.record_success()
                return upstream
            await upstream.aclose()
            if upstream.status_code == 429:
                # Our estimate drifted; the limiter is now corrected. The rejected reservation
                # stays in our window until it expires, which errs on the safe side.
                continue
            if upstream.status_code < 500:
                logging.error("Non-retryable %s from %s", upstream.status_code, model)
                return None
        # Server error or dropped connection: counts towards the circuit breaker, then retry.
        if await breaker.record_failure():
            stats["breaker_trips"] += 1
            logging.warning("Circuit opened for %s", model)
        attempt += 1
        backoff = min(0.25 * 2**attempt, 2.0) * random.uniform(0.5, 1.5)  # exponential, with jitter
        if attempt >= MAX_ATTEMPTS or time.monotonic() + backoff >= stop_at:
            return None
        stats["retries"] += 1
        await asyncio.sleep(backoff)


def busy_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": {"type": "busy", "message": "High demand right now. Please try again in a moment."}},
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    inflight_key = f"llm:inflight:{req.user_id}"
    if not await limiter.r.set(inflight_key, 1, nx=True, ex=INFLIGHT_TTL_S):
        return JSONResponse(
            status_code=409,
            content={"error": {"type": "in_progress", "message": "Your previous message is still being answered."}},
        )
    handed_off = False  # once streaming starts, the stream releases the in-flight lock
    try:
        history[req.user_id].append({"role": "user", "content": req.message})
        messages = build_messages(req.user_id)
        payload = {"messages": messages, "max_tokens": MAX_TOKENS, "stream": True, **end_user_fields(req.user_id)}
        charge = provider_charge(messages, MAX_TOKENS)
        start = time.monotonic()
        give_up_at = start + WAIT_DEADLINE_S

        model_used = CHAT_MODEL
        upstream = None
        if await limiter.estimate_wait(charge) <= PRIMARY_WAIT_S:
            upstream = await open_stream(
                CHAT_MODEL, limiter, chat_breaker, payload, charge, min(start + PRIMARY_WAIT_S, give_up_at)
            )
        else:
            stats["primary_skipped"] += 1  # known to be full: don't make the user wait for nothing
        if upstream is None:
            model_used = MINI_MODEL
            upstream = await open_stream(MINI_MODEL, mini_limiter, mini_breaker, payload, charge, give_up_at)
        if upstream is None:
            history[req.user_id].pop()  # unanswered: don't leave it in the conversation
            return busy_response()
        if model_used != CHAT_MODEL:
            stats["fallbacks"] += 1

        async def relay():
            reply = []
            try:
                async for line in upstream.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    for choice in json.loads(line[6:]).get("choices", []):
                        text = choice["delta"].get("content", "")
                        reply.append(text)
                        yield text
            except httpx.TransportError:
                # Tokens already reached the user, so a silent retry would duplicate them.
                stats["interrupted_streams"] += 1
                yield "\n[The reply was interrupted. Please send your message again.]"
            finally:
                await upstream.aclose()
                history[req.user_id].append({"role": "assistant", "content": "".join(reply)})
                await limiter.r.delete(inflight_key)
                schedule_summary(req.user_id)

        handed_off = True
        return StreamingResponse(relay(), media_type="text/plain", headers={"x-model": model_used})
    finally:
        if not handed_off:
            await limiter.r.delete(inflight_key)


@app.get("/stats")
async def get_stats():
    return stats


@app.post("/reset")
async def reset():
    history.clear()
    summaries.clear()
    await limiter.reset()
    await mini_limiter.reset()
    await chat_breaker.reset()
    await mini_breaker.reset()
    for key in stats:
        stats[key] = 0
    async for key in limiter.r.scan_iter("llm:inflight:*"):
        await limiter.r.delete(key)
    return {"ok": True}
