"""
Run the whole simulation locally, without Docker.

Starts the mock provider (8001), naive app (8000) and gateway (8002) with the
same settings as docker-compose.yml, using your local Redis. Ctrl+C stops all.

Usage:
    python run_local.py
    python run_local.py WAIT_DEADLINE_S=600 TPM_LIMIT=200000   # override any setting
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import redis

ROOT = Path(__file__).parent

SETTINGS = {
    # Provider limits; the gateway must use the same values.
    "RPM_LIMIT": "500",
    "TPM_LIMIT": "30000",
    "MINI_RPM_LIMIT": "500",
    "MINI_TPM_LIMIT": "200000",
    "WINDOW_SECONDS": "60",
    "SEED": "42",
    # Gateway
    "REDIS_URL": "redis://localhost:6379/15",  # separate DB so the demo can't touch your other data
    "MAX_TOKENS": "500",
    "HISTORY_TOKEN_BUDGET": "600",
    "WAIT_DEADLINE_S": "10",
    "PRIMARY_WAIT_S": "3",
    # Both apps: send a keyed hash of the user ID (never the raw ID). Off by default in the
    # code; on here because the mock uses it to keep each user's reply lengths repeatable.
    "SEND_END_USER_ID": "true",
    "END_USER_ID_SECRET": "demo-secret-change-me",
}

# The naive app keeps its own bloated settings.
SERVICES = [
    ("mock_provider", 8001, {}),
    ("naive_app", 8000, {"MAX_TOKENS": "1000"}),
    ("gateway_app", 8002, {}),
]


def apply_overrides(args: list[str]) -> None:
    for arg in args:
        key, sep, value = arg.partition("=")
        if not sep or key not in SETTINGS:
            sys.exit(f"Unknown setting '{arg}'. Use KEY=VALUE with one of: {', '.join(SETTINGS)}")
        SETTINGS[key] = value


def check_redis() -> None:
    try:
        redis.from_url(SETTINGS["REDIS_URL"]).ping()
    except redis.RedisError as e:
        sys.exit(f"Can't reach Redis at {SETTINGS['REDIS_URL']}: {e}")


def wait_until_up(port: int, timeout_s: float = 15) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://localhost:{port}/docs", timeout=1)
            return True
        except httpx.HTTPError:
            time.sleep(0.3)
    return False


def stop_on_sigterm(*_) -> None:
    raise KeyboardInterrupt


def main() -> None:
    signal.signal(signal.SIGTERM, stop_on_sigterm)  # `kill` stops everything too, not just Ctrl+C
    apply_overrides(sys.argv[1:])
    check_redis()
    (ROOT / "logs").mkdir(exist_ok=True)

    processes = []
    try:
        for name, port, extra in SERVICES:
            env = {**os.environ, **SETTINGS, **extra}
            log = open(ROOT / "logs" / f"{name}.log", "w")
            cmd = [sys.executable, "-m", "uvicorn", "app:app", "--port", str(port), "--log-level", "warning"]
            processes.append(subprocess.Popen(cmd, cwd=ROOT / name, env=env, stdout=log, stderr=subprocess.STDOUT))
            if not wait_until_up(port):
                sys.exit(f"{name} didn't start on port {port}; see logs/{name}.log")
            print(f"{name:<14} http://localhost:{port}")

        print("\nSettings:", ", ".join(f"{k}={v}" for k, v in SETTINGS.items()))
        print("Logs in ./logs. Ctrl+C to stop.\n")
        print("In another terminal:")
        print("  python load_test/run.py --label naive --seed 42")
        print("  python load_test/run.py --url http://localhost:8002/chat --label step3 --seed 42")
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        print("A service exited unexpectedly; see ./logs.")
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait(timeout=10)
        print("Stopped.")


if __name__ == "__main__":
    main()
