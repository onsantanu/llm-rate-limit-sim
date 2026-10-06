"""
Gateway chatbot backend.

Step 1 - request shaping: spend fewer tokens per request so more requests fit
in the same provider limit.
  - Lean system prompt instead of a bloated one.
  - History trimmed to a token budget (most recent turns kept).
  - max_tokens sized to what replies actually need, because the provider
    reserves max_tokens against TPM whether or not they are generated.
"""
import json
import os
from collections import defaultdict

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

PROVIDER_URL = os.getenv("PROVIDER_URL", "http://localhost:8001/v1/chat/completions")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
HISTORY_TOKEN_BUDGET = int(os.getenv("HISTORY_TOKEN_BUDGET", "600"))

# ~100 tokens: the same instructions, without repetition.
SYSTEM_PROMPT = (
    "You are a helpful support assistant for an online store. Answer order, delivery and refund "
    "questions concisely. Follow company policy: never promise dates you cannot verify, offer a "
    "refund only after confirming the order is lost or late beyond 7 days, and escalate anything "
    "involving payments to a human agent. Keep answers under 150 words."
)

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
history: dict[str, list[dict]] = defaultdict(list)


class ChatRequest(BaseModel):
    user_id: str
    message: str


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


@app.post("/chat")
async def chat(req: ChatRequest):
    history[req.user_id].append({"role": "user", "content": req.message})
    recent = trim_history(history[req.user_id], HISTORY_TOKEN_BUDGET)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *recent]
    payload = {"model": "mock-gpt", "messages": messages, "max_tokens": MAX_TOKENS, "stream": True}

    upstream = await client.send(client.build_request("POST", PROVIDER_URL, json=payload), stream=True)

    if upstream.status_code != 200:
        # Still no protection in step 1: the provider's error reaches the user.
        body = json.loads(await upstream.aread())
        await upstream.aclose()
        return JSONResponse(status_code=upstream.status_code, content=body)

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
        finally:
            await upstream.aclose()
            history[req.user_id].append({"role": "assistant", "content": "".join(reply)})

    return StreamingResponse(relay(), media_type="text/plain")


@app.post("/reset")
async def reset():
    history.clear()
    return {"ok": True}
