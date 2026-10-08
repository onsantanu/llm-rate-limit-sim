# LLM Rate Limit Simulation

Simulates a chatbot whose backbone is a rate-limited LLM API, under concurrent users.

- `mock_provider/` – OpenAI-style chat completions mock. Enforces RPM and TPM per model over a
  rolling window, returns 429s with rate-limit headers, streams tokens. Free and repeatable.
  Two models with separate limit pools: `mock-gpt` (chat) and `mock-mini` (summaries).
- `naive_app/` – chatbot backend that calls the provider directly (no limiter, no retries,
  full history, `max_tokens=1000`). The baseline that breaks.
- `gateway_app/` – the protected backend, built up step by step.
  Step 1 (request shaping): lean system prompt, history trimmed to a token budget,
  `max_tokens` sized to real replies. Turns that fall outside the history budget are folded
  into a rolling summary by `mock-mini` in the background, so context isn't simply lost and
  summarising doesn't use chat capacity.
  Step 2 (distributed limiter): Redis-backed rolling window that mirrors the provider's
  accounting, a shared FIFO queue with a wait deadline, one in-flight message per user,
  self-correction from rate-limit headers, and a clean "busy" reply instead of raw 429s.
  Step 3 (resilience): falls back to `mock-mini` when the chat model can't admit a request
  within `PRIMARY_WAIT_S` (or immediately, when the limiter already knows the wait would be
  longer), retries server errors and dropped connections with backoff and jitter (before the
  first token only), and a Redis-backed circuit breaker skips a failing model for a cooldown.
- `load_test/` – 20 users start at once, each sends 5 messages with think time.

## Run locally (no Docker)

Needs Python 3.10+ and a running Redis (the demo uses DB 15, so it won't touch your other data).

```bash
pip install -r requirements.txt
python run_local.py                      # starts provider :8001, naive app :8000, gateway :8002
python run_local.py WAIT_DEADLINE_S=600  # override any setting; Ctrl+C stops everything
```

Logs go to `./logs`. In another terminal:

```bash
python load_test/run.py --users 20 --messages 5 --label naive --seed 42
python load_test/run.py --url http://localhost:8002/chat --users 20 --messages 5 --label step3 --seed 42

# Chaos: 5% random server errors on both models
python load_test/run.py --url http://localhost:8002/chat --label step3_chaos --seed 42 --error-rate 0.05
```

To watch the circuit breaker handle an outage, give the chat model enough capacity that it
carries all the traffic, then take it down (restart `run_local.py` with the override):

```bash
python run_local.py TPM_LIMIT=200000
python load_test/run.py --url http://localhost:8002/chat --label breaker --seed 42 --outage mock-gpt:20:25
```

With the default tight limit, the chat model is mostly full, so an outage barely touches it:
requests are already being routed to the fallback.

## Run with Docker (alternative)

```bash
docker compose up --build -d
```

Same ports and settings (see `docker-compose.yml`), so the load-test commands above work unchanged.

The gateway runs the latest step.

Change provider limits (`TPM_LIMIT`, `RPM_LIMIT`) via `run_local.py` overrides or
`docker-compose.yml` to simulate different tiers. Results are saved to `results_<label>.json` for charting.

## Repeatable results

With `--seed` on the load test and `SEED` on the mock provider, each user's think times and
each user's n-th reply length are fixed, regardless of arrival order. Timing still varies
slightly (scheduling, the wait deadline), so report each step as the median of 3 runs.

## Privacy

The raw user ID never leaves the system. With `SEND_END_USER_ID=true`, the apps send a keyed
hash of it (HMAC with `END_USER_ID_SECRET`) in the provider's end-user field, so the provider
can attribute abuse to one end user instead of the whole account. It's off by default in the
code; the demo turns it on because the mock uses it to keep each user's reply lengths repeatable.

## Token accounting

The mock charges each request `estimated input tokens + max_tokens` against TPM at
admission time, approximating how providers reserve budget up front. Input tokens are
estimated at ~4 characters per token.
