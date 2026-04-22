"""Probe adjacent-pair candidates from /tmp/corpus_candidates.txt.

These are concatenations of adjacent greedy tokens that aren't yet in our
probed set. Yield should be higher than random kgrams because these are
likely BPE merges (adjacent tokens + merge candidate structure).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from counter import Config, SandwichCounter, batch_probe


async def main():
    cfg = Config.from_env()
    cfg.model = "claude-opus-4-7"
    cfg.target_rpm = 3000
    cfg.max_concurrency = 300

    with open("/tmp/corpus_candidates.txt") as f:
        cands = [line.rstrip("\n") for line in f if line.strip()]
    print(f"loaded {len(cands)} candidates")

    out = Path("state") / "phase_adjacent"
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, cands, out, desc="adjacent")


if __name__ == "__main__":
    asyncio.run(main())
