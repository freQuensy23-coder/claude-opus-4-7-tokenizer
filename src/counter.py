"""Async sandwich-counter for Claude count_tokens API with adaptive rate limiting.

Sandwich protocol: count(x) = raw(§ + x + §) - raw(§§).
§ (U+00A7) has a stable framing overhead and doesn't merge with adjacent
chars. Subtracting the §§ baseline removes framing.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx

API_URL = "https://api.anthropic.com/v1/messages/count_tokens"
MARKER = "\u00A7"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class Config:
    api_key: str
    model: str = "claude-opus-4-7"
    target_rpm: int = 3000
    max_concurrency: int = 300
    min_concurrency: int = 4
    initial_concurrency: int = 64
    timeout: float = 60.0
    max_retries: int = 8
    state_dir: Path = Path("state")

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set")
        return cls(api_key=api_key)


class AdaptiveLimiter:
    """Condition-based limiter with soft RPM ceiling and adaptive concurrency.

    - `_concurrency` is the current cap on in-flight requests; we adjust it
      multiplicatively in response to 429/5xx (shrink) or streaks of success
      (grow), bounded by [min_concurrency, max_concurrency].
    - `target_rpm` acts as a soft global ceiling: if we've made `target_rpm`
      requests in the last 60s, new acquires sleep until the oldest drops out.
    """

    def __init__(self, target_rpm: int, max_concurrency: int, min_concurrency: int,
                 initial_concurrency: int):
        self.target_rpm = target_rpm
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self._concurrency = max(min_concurrency, min(initial_concurrency, max_concurrency))
        self._inflight = 0
        self._cond = asyncio.Condition()
        self._timestamps: deque[float] = deque(maxlen=target_rpm)
        self._recent_429 = 0
        self._recent_ok = 0
        self._total_calls = 0
        self._total_429 = 0

    async def acquire(self):
        while True:
            async with self._cond:
                # Concurrency gate.
                while self._inflight >= self._concurrency:
                    await self._cond.wait()
                # Soft RPM gate.
                now = time.monotonic()
                if len(self._timestamps) >= self.target_rpm:
                    oldest = self._timestamps[0]
                    wait = oldest + 60.0 - now
                    if wait > 0:
                        # Release the cond while sleeping so others can check too.
                        self._cond.notify_all()
                        # We sleep *outside* the lock; break and re-check.
                        break_flag = ("wait", wait)
                    else:
                        self._timestamps.popleft()
                        break_flag = None
                else:
                    break_flag = None

                if break_flag is None:
                    self._inflight += 1
                    self._timestamps.append(now)
                    self._total_calls += 1
                    return
            # RPM wait (outside lock).
            _, wait = break_flag
            await asyncio.sleep(wait)

    async def release(self):
        async with self._cond:
            self._inflight -= 1
            self._cond.notify()

    async def note_429(self):
        async with self._cond:
            self._total_429 += 1
            self._recent_429 += 1
            self._recent_ok = 0
            if self._recent_429 >= 3 and self._concurrency > self.min_concurrency:
                self._concurrency = max(self.min_concurrency, self._concurrency // 2)
                self._recent_429 = 0
                self._cond.notify_all()

    async def note_ok(self):
        async with self._cond:
            self._recent_ok += 1
            if self._recent_ok >= 400 and self._concurrency < self.max_concurrency:
                bump = max(4, self._concurrency // 4)
                self._concurrency = min(self.max_concurrency, self._concurrency + bump)
                self._recent_ok = 0
                self._cond.notify_all()

    def stats(self) -> dict:
        return {
            "concurrency": self._concurrency,
            "inflight": self._inflight,
            "calls": self._total_calls,
            "429s": self._total_429,
            "rpm_window": len(self._timestamps),
        }


class SandwichCounter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.limiter = AdaptiveLimiter(
            target_rpm=cfg.target_rpm,
            max_concurrency=cfg.max_concurrency,
            min_concurrency=cfg.min_concurrency,
            initial_concurrency=cfg.initial_concurrency,
        )
        self._baseline: int | None = None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SandwichCounter":
        limits = httpx.Limits(
            max_connections=self.cfg.max_concurrency * 2,
            max_keepalive_connections=self.cfg.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            timeout=self.cfg.timeout,
            limits=limits,
            headers={
                "x-api-key": self.cfg.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        self._baseline = await self._raw_count(MARKER + MARKER)
        print(f"[counter] model={self.cfg.model} baseline(§§)={self._baseline}", flush=True)
        return self

    async def __aexit__(self, *_):
        await self._client.aclose()

    async def _raw_count(self, text: str) -> int:
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": text}],
        }
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            await self.limiter.acquire()
            try:
                r = await self._client.post(API_URL, json=payload)
                if r.status_code == 200:
                    await self.limiter.note_ok()
                    return r.json()["input_tokens"]
                if r.status_code == 400:
                    raise ValueError(f"400: {r.text[:200]}")
                if r.status_code in (401, 403):
                    raise RuntimeError(f"auth error {r.status_code}: {r.text[:200]}")
                # Transient: 429, 5xx.
                await self.limiter.note_429()
                retry_after_hdr = r.headers.get("retry-after")
                retry_after = float(retry_after_hdr) if retry_after_hdr else backoff
                await asyncio.sleep(min(30.0, retry_after))
                backoff = min(30.0, backoff * 2)
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                last_exc = e
                await asyncio.sleep(min(30.0, backoff))
                backoff = min(30.0, backoff * 2)
            finally:
                await self.limiter.release()
        raise RuntimeError(f"exhausted retries for {text!r}: {last_exc}")

    async def count_tokens(self, text: str) -> int:
        if text == "":
            return 0
        raw = await self._raw_count(MARKER + text + MARKER)
        return raw - self._baseline

    async def is_single_token(self, text: str) -> bool:
        return (await self.count_tokens(text)) == 1

    @property
    def baseline(self) -> int:
        assert self._baseline is not None
        return self._baseline


async def batch_probe(
    counter: SandwichCounter,
    candidates: Iterable[str],
    out_prefix: Path,
    desc: str = "probe",
    progress_every: int = 500,
    chunk_size: int = 5000,
) -> None:
    """Probe each candidate, write to a rotating CSV store. Resume-safe.

    `out_prefix` is a path-like that the Store appends `.NNNN.csv[.gz]` to.
    Prior runs (including legacy JSONL) are read for resume-deduplication.
    """
    from store import Store, load_checked  # local import to avoid cycles
    out_prefix = Path(out_prefix)
    seen = load_checked(out_prefix)
    todo = [c for c in candidates if c and c not in seen]
    total = len(todo)
    print(f"[{desc}] resume: {len(seen)} already done, {total} pending",
          flush=True)
    if total == 0:
        return

    store = Store(out_prefix)
    write_lock = asyncio.Lock()
    progress = {"done": 0, "hits": 0}
    start = time.monotonic()

    async def one(text: str):
        try:
            n = await counter.count_tokens(text)
            res = (n, None)
        except Exception as e:
            res = (None, str(e)[:200])
        async with write_lock:
            store.write(text, count=res[0], err=res[1])
            progress["done"] += 1
            if res[0] == 1:
                progress["hits"] += 1
            if progress["done"] % progress_every == 0:
                elapsed = time.monotonic() - start
                rate = progress["done"] / max(1e-6, elapsed) * 60.0
                s = counter.limiter.stats()
                print(
                    f"[{desc}] {progress['done']}/{total} hits={progress['hits']} "
                    f"rpm~{rate:.0f} conc={s['concurrency']} inflight={s['inflight']} "
                    f"429={s['429s']}",
                    flush=True,
                )

    try:
        for i in range(0, len(todo), chunk_size):
            batch = todo[i:i + chunk_size]
            await asyncio.gather(*(one(s) for s in batch))
    finally:
        store.close()

    print(f"[{desc}] done. probed={progress['done']} hits={progress['hits']} "
          f"checked_total={len(seen)+total}", flush=True)
