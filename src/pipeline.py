"""End-to-end pipeline for reconstructing Claude Opus 4.7 tokenizer.

Baseline: Opus 4.6 verified vocab (38,360 strings from Gupta/ctoc) re-checked
against 4.7. Surviving strings form the seed. Additional phases discover
4.7-specific merges from diverse sources.

Phases (each resumable via JSONL checkpoint under state/):

  0. calibrate    — §§ baseline + sanity on known single-tokens
  1. port         — re-verify 4.6 verified (Gupta) on 4.7   [baseline]
  2. handcrafted  — whitespace/digits/repeats/identifiers/morphemes/unicode
  3. tiktoken     — cl100k/o200k/gpt2 \\ already_checked, with auto-abort
                    if hit rate stays below 0.1% after 5K probes
  4. hf           — multilingual HF tokenizers, ranked by cross-frequency
  5. corpus       — iterative bigram/trigram merge discovery from real text
  6. boundary     — one more round of boundary probing on corpus
  7. assemble     — dedupe all hits into vocab_47.json
  8. evaluate     — greedy vs API on diverse samples

Run:  ANTHROPIC_API_KEY=sk-... uv run python src/pipeline.py --phase all
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Iterable

from counter import Config, SandwichCounter, batch_probe
import candidates as cands
import corpus as corpus_mod
import greedy
from store import load_checked as store_load_checked, load_hits as store_load_hits, migrate_jsonl_to_csv

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "state"
LOGS = ROOT / "logs"
REFERENCE = ROOT / "reference"
GUPTA_VOCAB = REFERENCE / "vocab.json"

# Prior phase names (prefix, not filename). Store handles the .NNNN.csv[.gz]
# extensions and legacy .jsonl/.jsonl.gz.
PRIOR_PHASES = [
    "phase_port",
    "phase_handcrafted",
    "phase_ngram",
    "phase_kgram",
    "phase_context",
    "phase_multilingual",
    "phase_smart",
    "phase_dict",
    "phase_tiktoken",
    "phase_hf",
    "phase_corpus",
    "phase_boundary",
    "phase_adjacent",
]


def dedupe_keep_order(xs):
    seen: set = set()
    out: list = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def load_hits_from_jsonl(prefix_or_path) -> set[str]:
    """Legacy name; dispatches to store.load_hits by prefix."""
    p = Path(prefix_or_path)
    # Support both path-to-jsonl and bare prefix.
    if p.suffix == ".jsonl":
        p = p.with_suffix("")
    return store_load_hits(p)


def load_checked_from_jsonl(prefix_or_path) -> set[str]:
    p = Path(prefix_or_path)
    if p.suffix == ".jsonl":
        p = p.with_suffix("")
    return store_load_checked(p)


def load_all_prior_checked() -> set[str]:
    s: set[str] = set()
    for name in PRIOR_PHASES:
        s |= store_load_checked(STATE / name)
    return s


def load_all_prior_hits() -> set[str]:
    s: set[str] = set()
    for name in PRIOR_PHASES:
        s |= store_load_hits(STATE / name)
    return s


async def phase_calibrate(cfg: Config):
    async with SandwichCounter(cfg) as c:
        print(f"baseline(§§) = {c.baseline}", flush=True)
        for s in ["a", " the", "hello", "0", "é", "の", " function", "    ",
                  "```", "```python", " const", " return"]:
            n = await c.count_tokens(s)
            print(f"  count({s!r}) = {n}", flush=True)


async def phase_port(cfg: Config):
    verified, _ = cands.load_gupta_vocab(GUPTA_VOCAB)
    out = STATE / "phase_port"
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, verified, out, desc="port-4.6")


async def phase_handcrafted(cfg: Config):
    out = STATE / "phase_handcrafted"
    already = load_checked_from_jsonl(out) | load_checked_from_jsonl(STATE / "phase_port")
    all_cands = dedupe_keep_order(list(cands.all_handcrafted()))
    todo = [s for s in all_cands if s not in already]
    print(f"[handcrafted] total unique: {len(all_cands)}; new to check: {len(todo)}", flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="handcrafted")


async def phase_ngram(cfg: Config):
    """Systematic n-gram sweep — 2/3-char + common 4-grams."""
    out = STATE / "phase_ngram"
    already = load_all_prior_checked()
    all_cands = dedupe_keep_order(list(cands.ngram_probes()))
    todo = [s for s in all_cands if s not in already]
    print(f"[ngram] total unique: {len(all_cands)}; new to check: {len(todo)}",
          flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="ngram")


async def phase_multilingual(cfg: Config):
    """Enumerate bigrams from major non-ASCII scripts."""
    out = STATE / "phase_multilingual"
    already = load_all_prior_checked()
    all_cands = dedupe_keep_order(list(cands.multilingual_bigrams()))
    todo = [s for s in all_cands if s not in already]
    print(f"[multilingual] candidates: {len(all_cands)}; new: {len(todo)}",
          flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="multilingual")


async def phase_context(cfg: Config, iterations: int = 10,
                        per_iter_cap: int = 400_000,
                        min_hits_to_continue: int = 100):
    """Iteratively generate context-variants of confirmed 4.7 tokens.

    Each iteration expands the confirmed set, then re-generates variants.
    Stops when an iteration yields < min_hits_to_continue new tokens.
    """
    out = STATE / "phase_context"
    async with SandwichCounter(cfg) as c:
        for it in range(iterations):
            known = load_all_prior_hits()
            already = load_all_prior_checked()
            variants = dedupe_keep_order(list(cands.context_variants_of(known)))
            todo = [s for s in variants if s not in already][:per_iter_cap]
            print(f"[context iter {it}] seed: {len(known)}; variants: "
                  f"{len(variants)}; new to probe: {len(todo)}", flush=True)
            if not todo:
                print(f"[context] no new candidates, stopping", flush=True)
                break
            hits_before = len(load_hits_from_jsonl(out))
            await batch_probe(c, todo, out, desc=f"context-it{it}")
            hits_after = len(load_hits_from_jsonl(out))
            gained = hits_after - hits_before
            print(f"[context iter {it}] +{gained} new tokens this iteration",
                  flush=True)
            if gained < min_hits_to_continue:
                print(f"[context] diminishing returns (<{min_hits_to_continue}), "
                      f"stopping", flush=True)
                break


async def phase_smart(cfg: Config, max_probes: int = 400_000,
                      min_freq: int = 2):
    """Targeted merge discovery: probe only substrings our greedy
    over-segments. Uses huge_corpus() (several MB) as candidate source.
    """
    from corpus import huge_corpus
    out = STATE / "phase_smart"
    known = load_all_prior_hits()
    already = load_all_prior_checked()
    text = huge_corpus()
    print(f"[smart] corpus size: {len(text.encode('utf-8'))} bytes, "
          f"seed vocab: {len(known)}", flush=True)
    candidates = cands.smart_kgrams(text, known, k_min=3, k_max=12,
                                     min_freq=min_freq, top_n=max_probes * 3)
    print(f"[smart] over-segmented candidates: {len(candidates)}", flush=True)
    todo = [c for c in candidates if c not in already][:max_probes]
    print(f"[smart] new to probe: {len(todo)}", flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="smart")


async def phase_kgram(cfg: Config, k_min: int = 2, k_max: int = 12,
                      min_freq: int = 5, max_probes: int = 600_000):
    """Mine k-grams from huge corpus, frequency-sorted.

    Uses `huge_corpus()` (several MB). Caps at max_probes for time budget.
    """
    from corpus import huge_corpus
    out = STATE / "phase_kgram"
    text = huge_corpus()
    print(f"[kgram] corpus size: {len(text.encode('utf-8'))} bytes", flush=True)
    kgrams = cands.kgrams_from_text(text, k_min=k_min, k_max=k_max,
                                     min_freq=min_freq)
    print(f"[kgram] unique k-grams (freq>={min_freq}): {len(kgrams)}",
          flush=True)
    already = load_all_prior_checked()
    todo = [s for s in kgrams if s not in already][:max_probes]
    print(f"[kgram] new to probe (capped at {max_probes}): {len(todo)}",
          flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="kgram")


async def phase_dict(cfg: Config, max_probes: int = 60_000):
    """Probe dictionary words and their common variants.

    Shuffled so early probes are representative (the dict file is
    alphabetical — the first 10K entries are archaic "aa"/"aalii"/..., which
    gives a misleading early hit rate). No abort: we want full coverage of
    the ~780K variants up to `max_probes`.
    """
    import random
    out = STATE / "phase_dict"
    already = load_all_prior_checked()
    all_cands = dedupe_keep_order(list(cands.dict_words()))
    random.Random(42).shuffle(all_cands)
    todo = [s for s in all_cands if s not in already][:max_probes]
    print(f"[dict] unique: {len(all_cands)} (shuffled); new: {len(todo)}",
          flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe(c, todo, out, desc="dict")


async def phase_tiktoken(cfg: Config, abort_threshold_ppm: int = 1000,
                        abort_after: int = 8000, max_probes: int = 80_000):
    """Probe tiktoken candidates with auto-abort on low hit rate.

    Rationale: cl100k tokens outside Gupta's 4.6 verified set almost never
    appear as 4.7 single-tokens. We probe a capped slice ordered by BPE rank
    (most frequent first) and bail if yield is too low.
    """
    out = STATE / "phase_tiktoken"
    already = load_all_prior_checked()

    all_cands: list[str] = []
    for enc in ("cl100k_base", "o200k_base", "gpt2"):
        try:
            before = len(all_cands)
            for s in cands.tiktoken_candidates(enc):
                all_cands.append(s)
            print(f"[tiktoken] {enc}: +{len(all_cands)-before}", flush=True)
        except Exception as e:
            print(f"[tiktoken] skip {enc}: {e}", flush=True)
    all_cands = dedupe_keep_order(all_cands)
    todo = [s for s in all_cands if s not in already][:max_probes]
    print(f"[tiktoken] unique: {len(all_cands)}; new (capped at {max_probes}): {len(todo)}", flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe_with_abort(c, todo, out, desc="tiktoken",
                                     abort_threshold_ppm=abort_threshold_ppm,
                                     abort_after=abort_after)


async def phase_hf(cfg: Config, max_per_model: int = 150_000, total_cap: int = 400_000,
                   abort_threshold_ppm: int = 100, abort_after: int = 30_000):
    out = STATE / "phase_hf"
    already = load_all_prior_checked()

    per_model: dict[str, list[str]] = {}
    for m in cands.HF_MODELS:
        print(f"[hf] loading {m} ...", flush=True)
        seen: set[str] = set()
        lst: list[str] = []
        try:
            for s in cands.hf_candidates(m):
                if s in seen:
                    continue
                seen.add(s)
                lst.append(s)
                if len(lst) >= max_per_model:
                    break
        except Exception as e:
            print(f"[hf] skip {m}: {e}", flush=True)
            continue
        per_model[m] = lst
        print(f"[hf] {m}: {len(lst)} candidates", flush=True)

    ranked = cands.rank_by_cross_frequency(per_model)
    todo = [s for s in ranked if s not in already]
    if len(todo) > total_cap:
        print(f"[hf] capping at {total_cap} (was {len(todo)})", flush=True)
        todo = todo[:total_cap]
    print(f"[hf] ranked union: {len(ranked)}; new to check: {len(todo)}", flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe_with_abort(c, todo, out, desc="hf",
                                     abort_threshold_ppm=abort_threshold_ppm,
                                     abort_after=abort_after)


def _generate_corpus_bigrams(vocab: set[str], text: str, min_len: int = 2,
                              max_len: int = 32) -> list[str]:
    """Produce candidates by concatenating adjacent tokens in greedy tokenization.

    For each i, yields tokens[i]+tokens[i+1] (bigram) and
    tokens[i]+tokens[i+1]+tokens[i+2] (trigram) — these are natural merge
    candidates that a BPE tokenizer might collapse into a single token.
    """
    trie = greedy.build_trie(sorted(vocab, key=len, reverse=True))
    tokens = greedy.greedy_tokenize(text, trie)
    out: set[str] = set()
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + tokens[i + 1]
        if min_len <= len(bigram) <= max_len:
            out.add(bigram)
        if i + 2 < len(tokens):
            tri = bigram + tokens[i + 2]
            if min_len <= len(tri) <= max_len:
                out.add(tri)
    return list(out)


async def phase_corpus(cfg: Config, iterations: int = 10,
                       per_iter_cap: int = 60_000,
                       min_hits_to_continue: int = 50):
    """Iterative merge discovery: each round adds tokens to the vocab and
    re-tokenizes the corpus to surface new bigram candidates.
    """
    out = STATE / "phase_corpus"
    already = load_all_prior_checked()
    known = load_all_prior_hits()
    text = corpus_mod.extended_corpus()
    print(f"[corpus] corpus bytes: {len(text.encode('utf-8'))}, seed vocab: {len(known)}",
          flush=True)

    async with SandwichCounter(cfg) as c:
        for it in range(iterations):
            # Refresh known: prior iterations write into the same out file.
            known |= load_hits_from_jsonl(out)
            candidates = _generate_corpus_bigrams(known, text)
            new = [s for s in candidates if s not in already]
            if not new:
                print(f"[corpus] iter {it}: no new candidates, stopping", flush=True)
                break
            new.sort(key=lambda s: (-sum(1 for _ in re.finditer(re.escape(s), text)), len(s)))
            new = new[:per_iter_cap]
            print(f"[corpus] iter {it}: {len(candidates)} candidates, "
                  f"{len(new)} new to probe", flush=True)
            hits_before = len(load_hits_from_jsonl(out))
            await batch_probe_with_abort(c, new, out, desc=f"corpus-it{it}",
                                         abort_threshold_ppm=200, abort_after=15_000)
            already |= set(new)
            hits_after = len(load_hits_from_jsonl(out))
            gained = hits_after - hits_before
            print(f"[corpus] iter {it}: +{gained} new tokens", flush=True)
            if gained < min_hits_to_continue:
                print(f"[corpus] diminishing returns, stopping", flush=True)
                break


async def phase_boundary(cfg: Config, max_new: int = 40_000):
    """Boundary-property search on corpus: find over-segmented windows and
    probe their sub-windows.
    """
    vocab_hits = load_all_prior_hits()
    trie = greedy.build_trie(sorted(vocab_hits, key=len, reverse=True))
    text = corpus_mod.extended_corpus()

    # Split into words (letters + digits + adjacent punctuation) and larger chunks.
    chunks = re.findall(r"\S{3,40}", text)
    chunks = dedupe_keep_order(chunks)

    candidates: list[str] = []
    for ch in chunks:
        if greedy.greedy_count(ch, trie) > 1:
            # try sub-ranges
            for i in range(len(ch)):
                for j in range(i + 2, min(len(ch) + 1, i + 16)):
                    sub = ch[i:j]
                    if greedy.greedy_count(sub, trie) > 1:
                        candidates.append(sub)
        if len(candidates) >= max_new * 2:
            break

    candidates = dedupe_keep_order(candidates)

    out = STATE / "phase_boundary"
    already = load_all_prior_checked()
    todo = [c for c in candidates if c not in already and c not in vocab_hits][:max_new]
    print(f"[boundary] candidates: {len(candidates)}, new to probe: {len(todo)}", flush=True)
    async with SandwichCounter(cfg) as c:
        await batch_probe_with_abort(c, todo, out, desc="boundary",
                                     abort_threshold_ppm=300, abort_after=10_000)


def phase_assemble():
    all_hits = load_all_prior_hits()
    per_phase: dict[str, int] = {}
    for name in PRIOR_PHASES:
        per_phase[name] = len(load_hits_from_jsonl(STATE / name))
    final = sorted(all_hits, key=lambda s: (-len(s), s))
    out = ROOT / "vocab_47.json"
    out.write_text(json.dumps({"verified": final, "model": "claude-opus-4-7",
                               "baseline": 13, "per_phase": per_phase},
                              ensure_ascii=False, indent=0))
    print(f"[assemble] wrote {out} — {len(final)} tokens. Per phase: {per_phase}",
          flush=True)


async def phase_evaluate(cfg: Config):
    vocab = greedy.load_vocab(ROOT / "vocab_47.json")
    trie = greedy.build_trie(vocab)
    samples = [
        ("python_hello", "def hello(name: str) -> str:\n    return f'Hello, {name}!'\n"),
        ("english_prose", "The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step."),
        ("json_blob", '{"user": "alice", "count": 42, "items": ["a", "b", "c"], "nested": {"k": true}}'),
        ("russian", "Быстрая коричневая лиса прыгает через ленивую собаку."),
        ("cjk", "快速的棕色狐狸跳过了懒惰的狗。"),
        ("japanese", "吾輩は猫である。名前はまだ無い。"),
        ("code_block", "for i in range(10):\n    print(i)\n    if i % 2 == 0:\n        continue\n"),
        ("markdown", "# Title\n\nSome **bold** and _italic_ text.\n\n- item 1\n- item 2\n"),
        ("long_english", "In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep."),
        ("typescript", "export const fetchUser = async (id: number): Promise<User | null> => {\n    const res = await fetch(`/api/users/${id}`);\n    return res.ok ? res.json() : null;\n};\n"),
    ]
    async with SandwichCounter(cfg) as c:
        print(f"{'name':<20} {'api':>5} {'greedy':>7} {'eff%':>6}", flush=True)
        efficiencies = []
        for name, text in samples:
            api_n = await c.count_tokens(text)
            g_n = greedy.greedy_count(text, trie)
            eff = 100.0 * api_n / g_n if g_n else 0.0
            efficiencies.append(eff)
            print(f"{name:<20} {api_n:>5} {g_n:>7} {eff:>5.1f}%", flush=True)
        avg = sum(efficiencies) / len(efficiencies)
        print(f"\naverage efficiency: {avg:.1f}%", flush=True)


async def batch_probe_with_abort(counter: SandwichCounter, candidates, out_prefix: Path,
                                 desc: str, abort_threshold_ppm: int = 0,
                                 abort_after: int = 0):
    """Same as batch_probe but emits live hit rate and aborts early if
    hit rate (per million) stays below `abort_threshold_ppm` after
    `abort_after` probes."""
    from store import Store, load_checked
    out_prefix = Path(out_prefix)
    seen = load_checked(out_prefix)
    todo = [c for c in candidates if c and c not in seen]
    total = len(todo)
    print(f"[{desc}] resume: {len(seen)} done, {total} pending", flush=True)
    if total == 0:
        return

    store = Store(out_prefix)
    write_lock = asyncio.Lock()
    progress = {"done": 0, "hits": 0, "abort": False, "last_print": time.monotonic()}
    start = time.monotonic()

    async def one(text: str):
        if progress["abort"]:
            return
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
            now = time.monotonic()
            need_print = (progress["done"] % 500 == 0) or (now - progress["last_print"] >= 30.0)
            if need_print:
                progress["last_print"] = now
                elapsed = now - start
                rate = progress["done"] / max(1e-6, elapsed) * 60.0
                hit_ppm = int(1_000_000 * progress["hits"] / max(1, progress["done"]))
                s = counter.limiter.stats()
                print(
                    f"[{desc}] {progress['done']}/{total} hits={progress['hits']} "
                    f"({hit_ppm}ppm) rpm~{rate:.0f} conc={s['concurrency']} "
                    f"inflight={s['inflight']} 429={s['429s']}",
                    flush=True,
                )
            if (abort_after and progress["done"] >= abort_after and
                    not progress["abort"] and abort_threshold_ppm > 0):
                hit_ppm_v = 1_000_000 * progress["hits"] / max(1, progress["done"])
                if hit_ppm_v < abort_threshold_ppm:
                    progress["abort"] = True
                    print(f"[{desc}] ABORT: hit rate {hit_ppm_v:.0f}ppm < "
                          f"threshold {abort_threshold_ppm}ppm after "
                          f"{progress['done']} probes", flush=True)

    try:
        for i in range(0, len(todo), 4000):
            if progress["abort"]:
                break
            batch = todo[i:i + 4000]
            await asyncio.gather(*(one(s) for s in batch))
    finally:
        store.close()

    elapsed = time.monotonic() - start
    print(f"[{desc}] done. probed={progress['done']} hits={progress['hits']} "
          f"elapsed={elapsed:.1f}s", flush=True)


PHASE_FNS = {
    "calibrate": phase_calibrate,
    "port": phase_port,
    "handcrafted": phase_handcrafted,
    "ngram": phase_ngram,
    "kgram": phase_kgram,
    "context": phase_context,
    "multilingual": phase_multilingual,
    "smart": phase_smart,
    "dict": phase_dict,
    "tiktoken": phase_tiktoken,
    "hf": phase_hf,
    "corpus": phase_corpus,
    "boundary": phase_boundary,
    "evaluate": phase_evaluate,
}

# Empirical data from ~60K HF probes, 77K dict probes, 45K tiktoken probes,
# 32K ngram probes: yields against 4.7 beyond Gupta's 4.6 verified set are
# all <0.5% — not worth 3000rpm quota. kgram (real-text frequency-sorted
# substrings) is the high-yield phase.
DEFAULT_SEQUENCE = ["calibrate", "port", "context", "multilingual",
                    "handcrafted", "kgram", "corpus", "boundary"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all"] + list(PHASE_FNS.keys()) + ["assemble"])
    ap.add_argument("--rpm", type=int, default=3000)
    ap.add_argument("--max-concurrency", type=int, default=300)
    ap.add_argument("--model", default="claude-opus-4-7")
    args = ap.parse_args()

    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    cfg = Config.from_env()
    cfg.target_rpm = args.rpm
    cfg.max_concurrency = args.max_concurrency
    cfg.model = args.model

    run = DEFAULT_SEQUENCE if args.phase == "all" else [args.phase]
    start = time.monotonic()
    for ph in run:
        if ph == "assemble":
            phase_assemble()
            continue
        fn = PHASE_FNS[ph]
        print(f"\n=== phase: {ph} ===", flush=True)
        t0 = time.monotonic()
        await fn(cfg)
        print(f"=== phase {ph} done in {time.monotonic()-t0:.1f}s ===\n", flush=True)

    if args.phase == "all":
        phase_assemble()
        await phase_evaluate(cfg)

    print(f"\ntotal elapsed: {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
