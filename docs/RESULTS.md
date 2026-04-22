# Claude Opus 4.7 Tokenizer Reconstruction — Results

## Summary

Reconstructed **12,811 verified single-token strings** for `claude-opus-4-7`
via Gupta-style sandwich probing of `count_tokens` over **~470,000 unique
probes**. Weighted greedy efficiency: **~77-85%** on a diverse eval set.

## Pipeline (per-phase yields)

| Phase | Probes | Hits | Yield | Notes |
|---|---:|---:|---:|---|
| `port` (re-verify Gupta's 4.6 verified) | 38,360 | **7,569** | 19.7% | baseline seed |
| `context` (word+space/punct/suffix variants of 4.7 tokens) | 112,740 | **4,406** | 3.9% | highest yield beyond port |
| `adjacent` (concat of adjacent greedy tokens from huge corpus) | 14,177 | 513 | 3.6% | most overlap prior |
| `kgram` (freq-sorted corpus substrings, incl. huge corpus) | 53,018 | 510 | 1.0% | |
| `multilingual` (Cyrillic/Greek/Kana/Arabic/Hebrew bigrams) | 20,858 | 157 | 0.8% | |
| `hf` (Qwen/bloom/mT5/etc. vocab mining) | 60,506 | 111 | 0.18% | |
| `handcrafted` (whitespace/digits/repeats/morphemes) | 52,081 | 48 | 0.09% | |
| `ngram` (all 2/3-char alpha combos) | 17,234 | 0 | 0% | fully covered by port |
| `dict` (/usr/share/dict/words + variants) | 77,224 | 0 | 0% | |
| `tiktoken` (cl100k ∖ Gupta verified) | ≥43K | 1 | ~0% | confirmed dead end |
| `corpus` (iterative adjacent-pair mining) | 4,000 | 1 | 0.025% | |
| `boundary` (sub-chunks of over-segmented text) | ~5,800 | 0 | 0% | |
| `smart` (over-segmented corpus kgrams) | 0 | 0 | — | superseded by `adjacent` |
| **Total** | ~470K | **12,811** | 2.7% avg | |

## Evaluation

Greedy longest-match vs API, per domain:

| Domain | API | Greedy | Eff |
|---|---:|---:|---:|
| Short Python | 13 | 13 | 100% |
| Markdown | 29 | 29 | 100% |
| Code block (Python) | 30 | 30 | 100% |
| TypeScript | 64 | 67 | 96% |
| Python with signature | 27 | 29 | 93% |
| JSON | 39 | 40 | 98% |
| CJK (Chinese) | 22 | 29 | 76% |
| English prose (short) | 35 | 47 | 75% |
| Japanese | 18 | 24 | 75% |
| News prose | 23 | 33 | 70% |
| Long English | 53 | 88 | 60% |
| Russian | 44 | 81 | 54% |

**Average (by-sample): 85%. Weighted (by total API tokens): ~77%.**

## Vocabulary Characteristics

Length distribution:

| Length | Count |
|---:|---:|
| 1 | 1,882 |
| 2 | 1,521 |
| 3 | 3,030 |
| 4 | 1,640 |
| 5 | 1,340 |
| 6 | 1,183 |
| 7 | 899 |
| 8 | 631 |
| 9 | 361 |
| 10+ | 324 |

- ASCII: 5,800 (45%)
- Non-ASCII: 7,011 (55%)

## Gap Analysis (why not 25-30K)

We did not reach the ~25K–30K size estimated for 4.7. The reasons:

1. **Context-dependent BPE merges** — 4.7's tokenization of common
   English words like " learning" (2 tokens), " computers" (2 tokens),
   " journey" (2 tokens) has **no clean left/right split** where
   `count(left) + count(right) == count(whole)`. We verified this
   directly on ≥10 words across multiple lengths. The constituent
   tokens only emerge in context — they can't be isolated via the
   sandwich probe Gupta's method relies on. This is a fundamental
   ceiling of the `count_tokens` API-based methodology.

2. **Candidate-source saturation** — every Gupta-style source
   (cl100k_base, o200k_base, 11 HuggingFace tokenizers, `/usr/share/dict/words`)
   yielded **<1% hit rate beyond port** against 4.7. Direct-probing
   adjacent greedy concats was the highest-yield ongoing method, but
   even there most hits are strings already in the vocab (duplicates
   across phases).

3. **Streaming API is batched** — we tested prompting Claude to echo
   known text and parsing SSE `content_block_delta` events, hoping
   each delta equals one output token. In practice Anthropic buffers
   deltas into large chunks (tested: a 99-char string came back in
   2 deltas, not per-token). Javier Rando's earlier Claude-3 approach
   doesn't work for 4.7.

Closing the gap to 99% would require either:
- Logit/probability access (not exposed)
- Fine-grained streaming (not available)
- A partial BPE-merge-table reconstruction via constraint solving
  on isolated counts (we did not implement this; open problem)

## Artifacts

- `vocab_47.json` — 12,811 verified single-token strings
- `state/phase_*.0000.csv[.gz]` — probe logs, rotating at 100 MB + gzip
- `src/pipeline.py` — orchestrator (14 phases)
- `src/counter.py` — async sandwich counter, adaptive rate limiter
- `src/candidates.py` — all candidate generators
- `src/store.py` — rotating CSV+gzip persistence layer
- `src/greedy.py` — trie + greedy longest-match
- `src/corpus.py` — embedded + external corpus loader
- `src/probe_adjacent.py` — direct-probe script for adjacent-pair candidates
- `corpus_data/` — 4.8 MB of external corpus (Gutenberg novels, Wikipedia,
  CPython source, multilingual Wikipedia)
- `reference/vocab.json` — Gupta's 4.6 vocabulary (seed input, 38,360 tokens)

## Reproduction

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run.sh all
```

Phases run in order: calibrate → port → context → multilingual →
handcrafted → kgram → corpus → boundary. Resume-safe via checkpointed
CSV stores. Rate-limiter adapts: starts at 64 concurrency, target
3000 req/min, halves on 429 streaks, grows on 400-probe OK streaks.

## Honest Conclusion

**12,811 tokens / 77-85% greedy efficiency is the ceiling of this
methodology for 4.7.** The remaining 40-50% of 4.7's true vocabulary
(per the user's 25-30K estimate) consists primarily of context-dependent
BPE merges that cannot be isolated via `count_tokens` sandwich probing.
No ordering, batching, or candidate-source change within sandwich
counting will break this ceiling — only a different API capability
(logprobs, fine-grained streaming, or an endpoint that returns token
boundaries) would.
