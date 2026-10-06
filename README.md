# LLM Rate Limit Simulation

Simulates a chatbot whose backbone is a rate-limited LLM API, under concurrent users.

- `mock_provider/` – OpenAI-style chat completions mock. Enforces RPM and TPM over a
  rolling window, returns 429s with rate-limit headers, streams tokens. Free and repeatable.
- `naive_app/` – chatbot backend that calls the provider directly (no limiter, no retries,
  full history, `max_tokens=1000`). The baseline that breaks.
- `gateway_app/` – the protected backend, built up step by step.
  Step 1 (request shaping): lean system prompt, history trimmed to a token budget,
  `max_tokens` sized to real replies.
- `load_test/` – 20 users start at once, each sends 5 messages with think time.

## Run

```bash
docker compose up --build -d
pip install -r load_test/requirements.txt
python load_test/run.py --users 20 --messages 5 --label naive
python load_test/run.py --url http://localhost:8002/chat --users 20 --messages 5 --label step1_shaping
```

Change provider limits in `docker-compose.yml` (`TPM_LIMIT`, `RPM_LIMIT`) to simulate
different tiers. Results are saved to `results_<label>.json` for charting.

## Token accounting

The mock charges each request `estimated input tokens + max_tokens` against TPM at
admission time, approximating how providers reserve budget up front. Input tokens are
estimated at ~4 characters per token.
