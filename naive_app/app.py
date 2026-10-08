"""
Naive chatbot backend: every user message goes straight to the LLM provider.

No limiter, no queue, no retries, full history every turn, generous max_tokens.
This is the baseline that breaks under concurrent load.
"""
import hashlib
import hmac
import json
import os
from collections import defaultdict

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

PROVIDER_URL = os.getenv("PROVIDER_URL", "http://localhost:8001/v1/chat/completions")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
SEND_END_USER_ID = os.getenv("SEND_END_USER_ID", "false").lower() == "true"
END_USER_ID_SECRET = os.getenv("END_USER_ID_SECRET", "")

# ~400 tokens: a realistic production system prompt.
SYSTEM_PROMPT = ("You are a helpful support assistant. Follow the company policy carefully. " * 22).strip()

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
history: dict[str, list[dict]] = defaultdict(list)


def end_user_fields(user_id: str) -> dict:
    # Same pseudonymous ID as the gateway, so the mock's seeding works for both apps.
    if not (SEND_END_USER_ID and END_USER_ID_SECRET):
        return {}
    return {"user": hmac.new(END_USER_ID_SECRET.encode(), user_id.encode(), hashlib.sha256).hexdigest()[:32]}


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    history[req.user_id].append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[req.user_id]]
    payload = {"model": "mock-gpt", "messages": messages, "max_tokens": MAX_TOKENS, "stream": True, **end_user_fields(req.user_id)}

    upstream = await client.send(client.build_request("POST", PROVIDER_URL, json=payload), stream=True)

    if upstream.status_code != 200:
        # Naive behaviour: the provider's error goes straight to the user.
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
