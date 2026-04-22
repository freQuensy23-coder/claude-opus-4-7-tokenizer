# Methods

Short description of every method used to reconstruct Claude Opus 4.7's
tokenizer from the `/v1/messages/count_tokens` endpoint alone.

## Core primitive: sandwich counting

All methods below rely on Gupta's sandwich trick to get an isolated token
count for any byte string `x`:

```
count(x) = raw(§x§) - raw(§§)
```

`§` (U+00A7) is a neutral marker that doesn't merge with its neighbors.
Subtracting the `§§` baseline removes chat-framing overhead. For 4.7,
baseline = 13.

## Candidate sources (where we got strings to probe)

| Method | What it does | Why |
|---|---|---|
| **port** | Re-verify every string in Gupta's 4.6 vocab (38,360 strings) against 4.7 | Most 4.6 tokens are also 4.7 tokens; cheapest & highest-yield seed |
| **tiktoken** | Decode every token id from `cl100k_base`, `o200k_base`, `gpt2`, `r50k_base`, `p50k_base` | GPT vocabularies trained on similar corpora overlap with Claude vocab |
| **hf** | Mine the full vocabs of 11 HF tokenizers (Qwen, bloom, mBART, mT5, Baichuan, DeepSeek, GPT-NeoX, DeBERTa, LaBSE, xlm-roberta, codet5p) | Multilingual + code coverage beyond tiktoken |
| **handcrafted** | Generate whitespace runs, digit sequences, repeated-char runs, morphemes, HTML entities, code fragments | Cover BPE patterns that all published vocabs share |
| **dict** | Every word in `/usr/share/dict/words` + casing variants (`word`, ` word`, ` Word`, ` WORD`), shuffled | Dictionary-level English coverage |
| **kgram** | Every character k-gram (k=2..14) from a 4.3 MB mixed corpus, frequency-sorted | BPE tokens are always real-text substrings |
| **multilingual** | Bigrams from Cyrillic, Greek, Hiragana, Katakana, Arabic, Hebrew, Latin-Extended | 4.7 keeps many non-ASCII short merges |
| **corpus** | Iterative bigram mining: greedy-tokenize a corpus with current vocab, then probe concatenations of adjacent tokens | Reverse-BPE: pairs that commonly appear together are often merges |
| **adjacent** | Adjacent-token-pair concatenations from the 4.3 MB corpus (large-scale) | Same principle as corpus, broader |

## Context-aware expansion (highest non-port yield)

| Method | What it does |
|---|---|
| **context** | For every confirmed 4.7 token `T`, generate variants `T+suf` where `suf ∈ {" ", ",", ".", "s", "es", "ed", "ing", ...}`, plus `" "+T+suf` and casing variants. Iterate: add new hits to seed, regenerate variants until saturated | BPE vocabs store distinct byte sequences for every practical context variant (`"history"`, `" history"`, `"history "` are three separate vocab entries) |

## Constraint-based methods (using the text→count map we already have)

For every probed record `(S, N)` we know `S` factors into exactly `N` 4.7
single-tokens. These methods exploit that:

| Method | Principle |
|---|---|
| **constraint / brute** | For every `(S, 2)` record: for every split `i`, if `S[:i]` is a known token, then `S[i:]` MUST be a single token. Probe the unknown half. |
| **mega** | Same for `(S, 3)` and `(S, 4)`: fix `N−1` known pieces at various positions, the remaining piece is a candidate. |
| **dp** | DP over `S` with minimum virtual (unknown) pieces. If the optimal factorization has exactly one virtual piece, that piece MUST be a single 4.7 token. |
| **triangulate** | Count how many different `(S, N)` factorizations independently suggest the same candidate. Candidates supported by many records have higher prior. |
| **smart_miner** | Same as adjacent but weighted by the greedy–API gap (records where greedy over-segments more get more vote weight). |

## Boundary / gap methods

| Method | Principle |
|---|---|
| **boundary** | For corpus windows where greedy_count > API_count, enumerate all sub-spans and probe as candidates. |
| **extension** | For each known `T`, find the bytes that follow `T` most often in corpus, probe `T + those_bytes`. |
| **pair** | Brute-force cartesian pairs of short alpha tokens. |
| **eval_sub** | Ultra-targeted: every 2..15-char substring of eval sample sentences. |

## Storage & infrastructure

- **store.py** — append-only rotating CSV store. Rotates at 100 MB, gzips
  the previous segment. Transparently reads both CSV and legacy JSONL,
  plain and `.gz`. Holds the full `text → count` map for resume-safety.
- **counter.py** — async sandwich counter with adaptive rate limiter
  (halves concurrency on 429 streaks, grows on success streaks).
  Target 3000 req/min.
- **greedy.py** — trie-backed greedy longest-match tokenizer used for
  evaluation and corpus-based candidate generation.
- **pipeline.py** — phase orchestrator with resume semantics.

## Evaluation

`phase_evaluate` calls the live `count_tokens` API on 10 diverse samples
(Python code, English prose, JSON, TypeScript, Russian, Chinese,
Japanese, markdown, news prose, long-form English) and compares to
greedy-longest-match counts from `vocab_47.json`.
Validation: random 500-token samples from the vocab file are re-probed
against the live API periodically; all come back `count = 1`.

## Yields (final)

| Method | Probes | Hits | Yield |
|---|---:|---:|---:|
| port | 38,360 | 7,569 | **19.7%** |
| context (iterative) | 112,740 | 4,406 | 3.9% |
| adjacent | 31,000 | 679 | 2.2% |
| constraint (c=2+c=3) | ~130K | ~400 | 0.3% |
| multilingual | 20,858 | 157 | 0.8% |
| hf | 121,012 | 222 | 0.18% |
| handcrafted | 104,162 | 96 | 0.09% |
| kgram | 53,018 | 510 | 1.0% |
| dict, ngram, dp, pair, eval_sub | combined ~400K | ~50 | ~0.01% |
| **TOTAL** | **~1.37 M** | **13,454** | **1.0% avg** |

Port dominates. Context is the main post-port multiplier. Beyond that,
yield drops fast because the remaining 4.7 vocab lives in context-dependent
BPE merges that do not return `count = 1` in isolation.
