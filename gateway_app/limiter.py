"""
Distributed LLM rate limiter backed by Redis.

- Mirrors the provider's accounting: a rolling window of admitted requests,
  each charged input estimate + max_tokens (no refund, because the provider
  doesn't refund either).
- FIFO waiting queue shared by all gateway instances: only the head of the
  queue may claim capacity, so nobody gets starved by luckier pollers.
- Self-correcting: after every provider response, the rate-limit headers are
  compared with our view and any extra usage the provider reports is recorded.
- One limiter per model, because each model has its own limit pool.
- Wait estimates: callers can ask how long a request would wait (tokens in use
  plus everything already queued ahead), and skip a model that is known to be full.
"""
import asyncio
import uuid

import redis.asyncio as redis

HEARTBEAT_PREFIX = "llm:hb:"   # waiter liveness, so a crashed instance can't block the queue
HEARTBEAT_TTL_S = 5
MAX_POLL_S = 0.25
NOT_HEAD_POLL_S = 0.05     # poll fast when behind, so the queue drains without gaps

# Returns "0" when admitted, "-1" when not at the head of the queue,
# or the seconds until the oldest window entry expires.
ACQUIRE_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local window_s = tonumber(ARGV[5])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window_s)

while true do
  local head = redis.call('ZRANGE', KEYS[2], 0, 0)[1]
  if not head or redis.call('EXISTS', ARGV[6] .. head) == 1 then break end
  redis.call('ZREM', KEYS[2], head)
  redis.call('HDEL', KEYS[3], head)
end
if redis.call('ZRANGE', KEYS[2], 0, 0)[1] ~= ARGV[1] then return '-1' end

local entries = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
local used = 0
for i = 1, #entries, 2 do
  used = used + tonumber(string.match(entries[i], ':(%d+)$'))
end
local count = #entries / 2
local charge = tonumber(ARGV[2])
if count + 1 <= tonumber(ARGV[3]) and used + charge <= tonumber(ARGV[4]) then
  redis.call('ZADD', KEYS[1], now, ARGV[1] .. ':' .. charge)
  redis.call('ZREM', KEYS[2], ARGV[1])
  redis.call('HDEL', KEYS[3], ARGV[1])
  return '0'
end
return tostring(tonumber(entries[2]) + window_s - now)
"""

# Seconds until a new request of ARGV[1] tokens would fit, after the window has
# freed enough for it and for everything already queued ahead of it.
# Only TPM is estimated: with chat-sized requests, RPM is never the binding limit.
ESTIMATE_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
local window_s = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window_s)
local entries = redis.call('ZRANGE', KEYS[1], 0, -1, 'WITHSCORES')
local used = 0
for i = 1, #entries, 2 do
  used = used + tonumber(string.match(entries[i], ':(%d+)$'))
end
local queued = 0
for _, charge in ipairs(redis.call('HVALS', KEYS[2])) do
  queued = queued + tonumber(charge)
end
local need = used + queued + tonumber(ARGV[1]) - tonumber(ARGV[2])
if need <= 0 then return '0' end
for i = 1, #entries, 2 do
  need = need - tonumber(string.match(entries[i], ':(%d+)$'))
  if need <= 0 then return tostring(tonumber(entries[i + 1]) + window_s - now) end
end
return tostring(window_s)
"""

# Records the gap when the provider reports more usage than our window holds.
SYNC_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1e6
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - tonumber(ARGV[2]))
local entries = redis.call('ZRANGE', KEYS[1], 0, -1)
local used = 0
for i = 1, #entries do
  used = used + tonumber(string.match(entries[i], ':(%d+)$'))
end
local gap = tonumber(ARGV[1]) - used
if gap > 0 then
  redis.call('ZADD', KEYS[1], now, 'corr-' .. ARGV[3] .. ':' .. gap)
end
return gap
"""


class LLMRateLimiter:
    def __init__(self, redis_url: str, model: str, rpm: int, tpm: int, window_s: float):
        self.r = redis.from_url(redis_url, decode_responses=True)
        self.window_key = f"llm:{model}:window"   # zset: member "<id>:<tokens>", score = admitted-at
        self.queue_key = f"llm:{model}:queue"     # zset: member waiter id, score = ticket number
        self.ticket_key = f"llm:{model}:ticket"
        self.charges_key = f"llm:{model}:queue_charges"  # hash: waiter id -> tokens it needs
        self.rpm, self.tpm, self.window_s = rpm, tpm, window_s
        self._acquire = self.r.register_script(ACQUIRE_LUA)
        self._sync = self.r.register_script(SYNC_LUA)
        self._estimate = self.r.register_script(ESTIMATE_LUA)

    async def acquire(self, charge: int, timeout_s: float) -> bool:
        """Wait in the shared FIFO queue until `charge` tokens fit, or give up after timeout_s."""
        if charge > self.tpm:
            raise ValueError(f"Request needs {charge} tokens; limit is {self.tpm} per window.")
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + timeout_s
        waiter = uuid.uuid4().hex
        ticket = await self.r.incr(self.ticket_key)
        await self.r.set(HEARTBEAT_PREFIX + waiter, 1, ex=HEARTBEAT_TTL_S)
        await self.r.hset(self.charges_key, waiter, charge)
        await self.r.zadd(self.queue_key, {waiter: ticket})
        try:
            while True:
                await self.r.set(HEARTBEAT_PREFIX + waiter, 1, ex=HEARTBEAT_TTL_S)
                wait = float(await self._acquire(
                    keys=[self.window_key, self.queue_key, self.charges_key],
                    args=[waiter, charge, self.rpm, self.tpm, self.window_s, HEARTBEAT_PREFIX],
                ))
                if wait == 0:
                    return True
                remaining = give_up_at - loop.time()
                if remaining <= 0:
                    return False
                await asyncio.sleep(min(wait if wait > 0 else NOT_HEAD_POLL_S, MAX_POLL_S, remaining))
        finally:
            await self.r.zrem(self.queue_key, waiter)
            await self.r.hdel(self.charges_key, waiter)
            await self.r.delete(HEARTBEAT_PREFIX + waiter)

    async def estimate_wait(self, charge: int) -> float:
        """Seconds a request of `charge` tokens would wait, counting the queue ahead of it."""
        return float(await self._estimate(
            keys=[self.window_key, self.charges_key], args=[charge, self.tpm, self.window_s]
        ))

    async def sync_from_headers(self, headers) -> None:
        """Align our window with the provider's own view of token usage."""
        limit = headers.get("x-ratelimit-limit-tokens")
        remaining = headers.get("x-ratelimit-remaining-tokens")
        if limit is None or remaining is None:
            return
        provider_used = int(limit) - int(remaining)
        await self._sync(keys=[self.window_key], args=[provider_used, self.window_s, uuid.uuid4().hex])

    async def reset(self) -> None:
        await self.r.delete(self.window_key, self.queue_key, self.ticket_key, self.charges_key)
