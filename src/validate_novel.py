"""Re-validate the ``only47`` set (tokens in vocab_47.json \\ vocab_46.json)
via live count_tokens sandwich probes on both claude-opus-4-7 and
claude-opus-4-6. Produces VALIDATE_NOVEL.md.

For each of the ~5,624 candidates we answer two questions:

  1. Is this a single token on 4.7?   (sandwich count == 1)
  2. Is this also a single token on 4.6?
     If yes → not truly novel; Gupta just never probed it.
     If no  → genuinely new in 4.7.

Inputs:  vocab_46.json, vocab_47.json
Outputs: state/validate_47.csv, state/validate_46.csv (resume-safe)
         VALIDATE_NOVEL.md  (summary)
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from counter import Config, SandwichCounter  # noqa: E402


def load_verified(p: Path) -> list[str]:
    with p.open(encoding="utf-8") as f:
        return json.load(f)["verified"]


def read_done(path: Path) -> dict[str, int | None]:
    """Return {text: count} from a prior run's CSV (resume support)."""
    if not path.exists():
        return {}
    out: dict[str, int | None] = {}
    with path.open(encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row or len(row) < 2:
                continue
            text = bytes(row[0], "utf-8").decode("unicode_escape")
            try:
                out[text] = int(row[1])
            except ValueError:
                out[text] = None
    return out


def esc(s: str) -> str:
    return s.encode("unicode_escape").decode("ascii")


async def run_pass(
    *,
    model: str,
    candidates: list[str],
    out_csv: Path,
    desc: str,
    target_rpm: int,
    max_concurrency: int,
) -> dict[str, int | None]:
    done = read_done(out_csv)
    todo = [c for c in candidates if c not in done]
    print(f"[{desc}] {len(done):,} cached, {len(todo):,} pending "
          f"(total {len(candidates):,})", flush=True)
    if not todo:
        return done

    cfg = Config(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=model,
        target_rpm=target_rpm,
        max_concurrency=max_concurrency,
    )
    results: dict[str, int | None] = dict(done)
    progress = {"done": 0, "hits": 0}
    start = time.monotonic()
    write_lock = asyncio.Lock()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fh = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fh)

    async with SandwichCounter(cfg) as counter:
        async def one(text: str) -> None:
            try:
                n = await counter.count_tokens(text)
            except Exception as e:
                n = None
                err = str(e)[:120]
            else:
                err = ""
            async with write_lock:
                writer.writerow([esc(text), "" if n is None else n, err])
                results[text] = n
                progress["done"] += 1
                if n == 1:
                    progress["hits"] += 1
                if progress["done"] % 500 == 0:
                    elapsed = time.monotonic() - start
                    rate = progress["done"] / max(1e-6, elapsed) * 60.0
                    s = counter.limiter.stats()
                    print(f"[{desc}] {progress['done']:,}/{len(todo):,} "
                          f"hits={progress['hits']:,} "
                          f"rpm~{rate:.0f} conc={s['concurrency']} "
                          f"inflight={s['inflight']} 429={s['429s']}",
                          flush=True)
                    fh.flush()

        CHUNK = 2000
        for i in range(0, len(todo), CHUNK):
            await asyncio.gather(*(one(t) for t in todo[i:i + CHUNK]))

    fh.flush()
    fh.close()
    elapsed = time.monotonic() - start
    print(f"[{desc}] finished {progress['done']:,} probes in {elapsed:.1f}s "
          f"({progress['done']/max(1e-6,elapsed)*60.0:.0f} rpm); "
          f"hits={progress['hits']:,}", flush=True)
    return results


def hist(results: dict[str, int | None]) -> Counter:
    c: Counter = Counter()
    for v in results.values():
        if v is None:
            c["err"] += 1
        elif v == 1:
            c["single"] += 1
        elif v == 0:
            c["zero"] += 1
        else:
            c[f">=2 (got {min(v,5)})" if v <= 5 else ">=6"] += 1
    return c


async def main() -> None:
    v46 = set(load_verified(ROOT / "vocab_46.json"))
    v47_list = load_verified(ROOT / "vocab_47.json")
    only47 = [t for t in v47_list if t not in v46]
    print(f"[main] |only47| = {len(only47):,}")

    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    csv47 = state / "validate_only47_on_47.csv"
    csv46 = state / "validate_only47_on_46.csv"

    # Pass 1: on 4.7 (the important one)
    r47 = await run_pass(
        model="claude-opus-4-7",
        candidates=only47,
        out_csv=csv47,
        desc="on 4.7",
        target_rpm=3000,
        max_concurrency=300,
    )

    # Pass 2: on 4.6 (to see what fraction are truly novel)
    r46 = await run_pass(
        model="claude-opus-4-6",
        candidates=only47,
        out_csv=csv46,
        desc="on 4.6",
        target_rpm=3000,
        max_concurrency=300,
    )

    # Build summary
    h47 = hist(r47)
    h46 = hist(r46)

    # Joint: truly novel to 4.7 (1 on 4.7, not 1 on 4.6)
    joint = {"truly_novel": 0, "shared_single": 0,
             "not_single_47": 0, "err": 0}
    not_single_examples: list[str] = []
    shared_examples: list[str] = []
    novel_examples: list[str] = []
    for t in only47:
        n47 = r47.get(t)
        n46 = r46.get(t)
        if n47 is None or n46 is None:
            joint["err"] += 1
        elif n47 != 1:
            joint["not_single_47"] += 1
            if len(not_single_examples) < 30:
                not_single_examples.append((t, n47))
        elif n46 == 1:
            joint["shared_single"] += 1
            if len(shared_examples) < 30:
                shared_examples.append(t)
        else:
            joint["truly_novel"] += 1
            if len(novel_examples) < 30:
                novel_examples.append((t, n46))

    # Write markdown
    def fmt(s: str) -> str:
        r = repr(s)
        return r if len(r) <= 60 else r[:57] + "..."

    md = ROOT / "VALIDATE_NOVEL.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# Validation of 4.7-novel tokens via live count_tokens\n\n")
        f.write(f"Re-probed all **{len(only47):,} tokens** in "
                f"`vocab_47.json \\ vocab_46.json` via the sandwich\n"
                f"protocol against *both* models' `count_tokens` endpoints.\n\n")
        f.write("## 1. Per-model sandwich count distribution\n\n")
        f.write("| count on model | 4.7 | 4.6 |\n|---|---:|---:|\n")
        keys = sorted(set(h47) | set(h46))
        for k in keys:
            f.write(f"| `{k}` | {h47.get(k,0):,} | {h46.get(k,0):,} |\n")
        f.write("\n")
        f.write("## 2. Joint classification\n\n")
        total = len(only47)
        for k, label in [
            ("truly_novel", "Single on 4.7, NOT single on 4.6 (genuinely novel)"),
            ("shared_single", "Single on 4.7 AND on 4.6 (Gupta missed in 4.6)"),
            ("not_single_47", "Not single on 4.7 (false positive in our vocab)"),
            ("err", "Error on either pass"),
        ]:
            n = joint[k]
            f.write(f"- **{label}**: {n:,} ({n/total:.1%})\n")
        f.write("\n")

        f.write("## 3. Examples\n\n")
        f.write("### Truly novel in 4.7 (single on 4.7, >1 on 4.6)\n\n")
        for t, n46 in novel_examples:
            f.write(f"- `{fmt(t)}` — on 4.6: {n46} tokens\n")
        f.write("\n### Shared: single on both (Gupta missed in 4.6 probing)\n\n")
        for t in shared_examples:
            f.write(f"- `{fmt(t)}`\n")
        f.write("\n### False positives (claimed single, actually not)\n\n")
        for t, n47 in not_single_examples:
            f.write(f"- `{fmt(t)}` — on 4.7: {n47} tokens\n")
    print(f"[main] wrote {md}")


if __name__ == "__main__":
    asyncio.run(main())
