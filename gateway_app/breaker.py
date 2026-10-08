"""
Circuit breaker per model, shared across gateway instances via Redis.

After `threshold` consecutive failures (server errors or dropped connections)
within `window_s`, the circuit opens for `cooldown_s`: no traffic goes to that
model, so requests switch to the fallback immediately instead of waiting and
retrying against a provider that is down. After the cooldown, traffic flows
again; if it keeps failing, the circuit re-opens after `threshold` failures.

Rate limits (429) are not failures: they mean "busy", which the limiter handles.
"""
import redis.asyncio as redis


class CircuitBreaker:
    def __init__(self, r: redis.Redis, model: str, threshold: int = 5, window_s: int = 30, cooldown_s: int = 15):
        self.r = r
        self.failures_key = f"llm:{model}:cb:failures"
        self.open_key = f"llm:{model}:cb:open"
        self.threshold, self.window_s, self.cooldown_s = threshold, window_s, cooldown_s

    async def is_open(self) -> bool:
        return await self.r.exists(self.open_key) == 1

    async def record_success(self) -> None:
        await self.r.delete(self.failures_key)

    async def record_failure(self) -> bool:
        """Returns True if this failure opened the circuit."""
        failures = await self.r.incr(self.failures_key)
        if failures == 1:
            await self.r.expire(self.failures_key, self.window_s)
        if failures >= self.threshold:
            await self.r.delete(self.failures_key)
            return bool(await self.r.set(self.open_key, 1, ex=self.cooldown_s, nx=True))
        return False

    async def reset(self) -> None:
        await self.r.delete(self.failures_key, self.open_key)
