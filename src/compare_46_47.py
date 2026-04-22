"""Gupta-style comparison of claude-opus-4-6 vs claude-opus-4-7 vocabs.

Loads vocab_46.json (Gupta's verified 4.6 set) and vocab_47.json (this project's
verified 4.7 set), then produces:

  1. Set-diff summary (|A∩B|, |A\\B|, |B\\A|)
  2. Length distribution per vocab
  3. Script/character-class breakdown (ASCII / Latin / Cyrillic / CJK / etc.)
  4. Space-prefix counts
  5. Sample tokens 4.7-only and 4.6-only (short + long)
  6. Greedy longest-match head-to-head on every file in corpus_data/
     using each vocab's trie (Gupta's ctoc tokenizer). Reports token counts
     and efficiency relative to each other (4.6 tokens / 4.7 tokens).

No API calls — comparison is purely structural + greedy on local corpus.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from greedy import build_trie, greedy_count  # noqa: E402


def load(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)["verified"]


def script_bucket(s: str) -> str:
    """Coarse script classification for a token."""
    if not s:
        return "empty"
    if all(ord(c) < 128 for c in s):
        return "ascii"
    buckets: Counter[str] = Counter()
    for c in s:
        o = ord(c)
        if o < 128:
            buckets["ascii"] += 1
        elif 0x0400 <= o <= 0x04FF:
            buckets["cyrillic"] += 1
        elif 0x0370 <= o <= 0x03FF:
            buckets["greek"] += 1
        elif 0x0590 <= o <= 0x05FF:
            buckets["hebrew"] += 1
        elif 0x0600 <= o <= 0x06FF:
            buckets["arabic"] += 1
        elif 0x3040 <= o <= 0x309F:
            buckets["hiragana"] += 1
        elif 0x30A0 <= o <= 0x30FF:
            buckets["katakana"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            buckets["cjk"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            buckets["hangul"] += 1
        elif unicodedata.category(c).startswith("L"):
            buckets["latin_ext"] += 1
        else:
            buckets["other"] += 1
    return buckets.most_common(1)[0][0]


def len_hist(tokens: list[str]) -> Counter[int]:
    c: Counter[int] = Counter()
    for t in tokens:
        n = len(t)
        c[n if n < 10 else 10] += 1
    return c


def fmt(s: str) -> str:
    r = repr(s)
    return r if len(r) <= 40 else r[:37] + "..."


def bar(n: int, total: int, width: int = 30) -> str:
    if total == 0:
        return ""
    w = int(round(width * n / total))
    return "█" * w + "·" * (width - w)


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    v46 = load(ROOT / "vocab_46.json")
    v47 = load(ROOT / "vocab_47.json")
    s46, s47 = set(v46), set(v47)

    section("1. Vocabulary sizes and overlap")
    inter = s46 & s47
    only46 = s46 - s47
    only47 = s47 - s46
    print(f"  |4.6 verified (Gupta)|           = {len(s46):>7,}")
    print(f"  |4.7 verified (this project)|    = {len(s47):>7,}")
    print(f"  |intersection|                   = {len(inter):>7,}  "
          f"({len(inter)/len(s46):.1%} of 4.6, {len(inter)/len(s47):.1%} of 4.7)")
    print(f"  |4.6-only (dropped or unprobed)| = {len(only46):>7,}")
    print(f"  |4.7-only (novel in 4.7)|        = {len(only47):>7,}")
    jaccard = len(inter) / len(s46 | s47)
    print(f"  Jaccard(4.6, 4.7)                = {jaccard:.4f}")

    section("2. Length distribution (chars)")
    h46, h47 = len_hist(v46), len_hist(v47)
    print(f"  {'len':>4}  {'4.6':>7}  {'4.7':>7}  {'bar 4.6':<30}  {'bar 4.7':<30}")
    for L in sorted(set(h46) | set(h47)):
        label = f"{L}+" if L == 10 else str(L)
        print(f"  {label:>4}  {h46[L]:>7,}  {h47[L]:>7,}  "
              f"{bar(h46[L], max(h46.values())):<30}  "
              f"{bar(h47[L], max(h47.values())):<30}")

    section("3. Script / character-class distribution")
    sc46, sc47 = Counter(), Counter()
    for t in v46:
        sc46[script_bucket(t)] += 1
    for t in v47:
        sc47[script_bucket(t)] += 1
    keys = sorted(set(sc46) | set(sc47),
                  key=lambda k: -(sc46[k] + sc47[k]))
    print(f"  {'bucket':<12}  {'4.6':>7}  {'%':>5}  {'4.7':>7}  {'%':>5}")
    for k in keys:
        print(f"  {k:<12}  {sc46[k]:>7,}  "
              f"{sc46[k]/len(v46):>5.1%}  "
              f"{sc47[k]:>7,}  "
              f"{sc47[k]/len(v47):>5.1%}")

    section("4. Space-prefix and casing")
    sp46 = sum(1 for t in v46 if t.startswith(" "))
    sp47 = sum(1 for t in v47 if t.startswith(" "))
    cap46 = sum(1 for t in v46 if t.strip() and t.strip()[0].isupper())
    cap47 = sum(1 for t in v47 if t.strip() and t.strip()[0].isupper())
    print(f"  space-prefixed    4.6: {sp46:>5,} ({sp46/len(v46):>5.1%})  "
          f"4.7: {sp47:>5,} ({sp47/len(v47):>5.1%})")
    print(f"  cap-initial       4.6: {cap46:>5,} ({cap46/len(v46):>5.1%})  "
          f"4.7: {cap47:>5,} ({cap47/len(v47):>5.1%})")

    section("5. Sample tokens unique to each side")
    def sample(tokens: set[str], *, script: str, min_len=3, max_len=12, n=12):
        out = [t for t in sorted(tokens, key=lambda x: (len(x), x))
               if min_len <= len(t) <= max_len
               and script_bucket(t) == script
               and not any(ord(c) < 0x20 and c not in "\t\n\r" for c in t)]
        return out[:n]

    print("  4.7-only ASCII (novel merges introduced in 4.7):")
    for t in sample(only47, script="ascii", n=15):
        print(f"    {fmt(t)}")
    print("\n  4.6-only ASCII (present in Gupta, not re-verified for 4.7):")
    for t in sample(only46, script="ascii", n=15):
        print(f"    {fmt(t)}")
    print("\n  4.7-only non-ASCII (by script, 5 each):")
    for sc in ["cyrillic", "cjk", "katakana", "hiragana", "arabic", "hebrew", "greek"]:
        pool = sample(only47, script=sc, min_len=1, max_len=8, n=5)
        if pool:
            print(f"    [{sc}] " + "  ".join(fmt(t) for t in pool))
    print("\n  4.6-only non-ASCII (by script, 5 each):")
    for sc in ["cyrillic", "cjk", "katakana", "hiragana", "arabic", "hebrew", "greek"]:
        pool = sample(only46, script=sc, min_len=1, max_len=8, n=5)
        if pool:
            print(f"    [{sc}] " + "  ".join(fmt(t) for t in pool))

    section("6. Greedy longest-match head-to-head on corpus_data/")
    t46 = build_trie(v46)
    t47 = build_trie(v47)
    corpus_dir = ROOT / "corpus_data"
    files = sorted(corpus_dir.iterdir()) if corpus_dir.exists() else []
    if not files:
        print("  (no corpus_data/ found)")
        return

    print(f"  {'file':<22} {'bytes':>8}  {'4.6 toks':>9}  {'4.7 toks':>9}  "
          f"{'47/46':>7}  {'best':>5}")
    tot46 = tot47 = tot_bytes = 0
    for p in files:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"  {p.name:<22} skip ({e})")
            continue
        c46 = greedy_count(text, t46)
        c47 = greedy_count(text, t47)
        b = len(text.encode("utf-8"))
        tot46 += c46
        tot47 += c47
        tot_bytes += b
        ratio = c47 / c46 if c46 else 0.0
        winner = "4.7" if c47 < c46 else ("4.6" if c46 < c47 else "tie")
        print(f"  {p.name:<22} {b:>8,}  {c46:>9,}  {c47:>9,}  "
              f"{ratio:>7.3f}  {winner:>5}")
    print(f"  {'TOTAL':<22} {tot_bytes:>8,}  {tot46:>9,}  {tot47:>9,}  "
          f"{tot47/tot46:>7.3f}")
    print()
    print(f"  ratio = 4.7-greedy / 4.6-greedy  (values >1: 4.7-vocab loses)")
    print(f"  Bytes/token — 4.6-vocab: {tot_bytes/tot46:.2f}   "
          f"4.7-vocab (partial): {tot_bytes/tot47:.2f}")
    print()
    print("  CAVEAT: this compares *vocab coverage*, not the real 4.6 vs 4.7")
    print("  tokenizers. 4.6 wins because our 4.7 reconstruction is partial")
    print("  (~13K verified of an estimated ~25-30K). The true 4.7 tokenizer")
    print("  beats 4.6 on the same text per Anthropic's release notes.")

    section("7. Delta vs 4.6, by corpus length (novel-token yield)")
    # For each corpus file, what fraction of 4.7-greedy tokens are *new*
    # in 4.7 (i.e. in only47)?  Gives a sense of where 4.7 merges matter.
    from greedy import greedy_tokenize
    print(f"  {'file':<22} {'4.7 toks':>9}  {'4.7-only':>9}  {'%novel':>7}")
    for p in files:
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        toks = greedy_tokenize(text, t47)
        novel = sum(1 for t in toks if t in only47)
        pct = novel / len(toks) if toks else 0.0
        print(f"  {p.name:<22} {len(toks):>9,}  {novel:>9,}  {pct:>7.1%}")


if __name__ == "__main__":
    main()
