"""Probe smart-miner candidates from /tmp/smart_cands.txt."""
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

    with open("/tmp/smart_cands.txt") as f:
        cands = [line.rstrip("\n") for line in f if line.strip()]
    print(f"loaded {len(cands)} candidates")

    out = Path("state") / "phase_smart_miner"
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, cands, out, desc="smart_miner", progress_every=500)


if __name__ == "__main__":
    asyncio.run(main())
