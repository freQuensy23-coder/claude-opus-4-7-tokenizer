"""Aggressive continued probing: push HF models harder, mine Wikipedia deep."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from counter import Config, SandwichCounter, batch_probe
from store import load_hits, load_checked
import candidates as cands
import corpus


def filter_new(items, already):
    seen = set()
    out = []
    for s in items:
        if s and s not in already and s not in seen:
            seen.add(s)
            out.append(s)
    return out


async def main():
    cfg = Config.from_env()
    cfg.model = "claude-opus-4-7"
    cfg.target_rpm = 3000
    cfg.max_concurrency = 300

    state = Path("state")
    already = set()
    for p in ["port", "context", "multilingual", "handcrafted", "kgram", "hf",
              "ngram", "dict", "tiktoken", "corpus", "boundary", "smart",
              "adjacent", "probe_more"]:
        already |= load_checked(state / f"phase_{p}")

    known = set()
    for p in ["port", "context", "multilingual", "handcrafted", "kgram", "hf",
              "ngram", "dict", "tiktoken", "corpus", "boundary", "smart",
              "adjacent"]:
        known |= load_hits(state / f"phase_{p}")
    print(f"already probed: {len(already)}, known vocab: {len(known)}")

    # 1. Full HF tokenizer mining (no cross-freq filter).
    all_hf = []
    for m in cands.HF_MODELS:
        print(f"[probe_more] loading {m}", flush=True)
        try:
            for s in cands.hf_candidates(m):
                if s and 1 <= len(s) <= 20 and "\ufffd" not in s:
                    all_hf.append(s)
        except Exception as e:
            print(f"  skip {m}: {e}")
    print(f"[probe_more] total HF items: {len(all_hf)}")

    # 2. Deep Wikipedia / Gutenberg kgrams at very wide lengths (15-25).
    text = corpus.huge_corpus()
    from collections import Counter
    cnt: Counter[str] = Counter()
    # Pure word boundaries
    import re
    for m in re.finditer(r"\S{3,20}", text):
        cnt[m.group()] += 1
    for m in re.finditer(r" \S{3,20}", text):
        cnt[m.group()] += 1
    real_words = [w for w, f in cnt.most_common() if f >= 2]
    print(f"[probe_more] real-word candidates: {len(real_words)}")

    # 3. Also try variant forms of known vocab (from context).
    variants = list(cands.context_variants_of(known))

    # Combine + dedupe + filter
    combined = all_hf + real_words + variants
    todo = filter_new(combined, already)
    print(f"[probe_more] new to probe: {len(todo)}")

    # Limit to a sane number
    if len(todo) > 800_000:
        todo = todo[:800_000]
        print(f"[probe_more] capped to {len(todo)}")

    out = state / "phase_probe_more"
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="probe_more", progress_every=1000,
                          chunk_size=4000)


if __name__ == "__main__":
    asyncio.run(main())
