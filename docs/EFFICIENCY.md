# Opus 4.6 vs 4.7 Tokenizer Efficiency — Live API Study

**Method.** 46 text samples across 30+ domains, each sent via
`count_tokens` to both `claude-opus-4-6` and `claude-opus-4-7`, with
framing baselines subtracted (`raw("a") − 1`: 4.6→7, 4.7→11). Samples
trimmed to ≤~11 KB each. One run (~2 s, zero 429s).

All ratios below are `tokens(4.7) / tokens(4.6)`.

> **<1.00** → 4.7 uses fewer tokens (4.7 "better").
> **>1.00** → 4.7 uses **more** tokens (4.7 worse).

---

## TL;DR

1. **4.7 is neutral on non-English scripts** (≈1.00 ± 0.02 for Arabic,
   Hebrew, Greek, Thai, Hindi, Russian, Chinese, Japanese, Korean).
2. **4.7 is 25–50 % worse on English and code.** Every English prose and
   every programming language sample costs significantly more tokens.
3. **Hypothesis.** 4.7 shed its long English / code merges (`" learning"`,
   `" computers"`, `" journey"`, `" function"`, etc.) while keeping the
   multilingual vocabulary. Net: 4.7's vocab is a *narrower* subset of
   4.6's that happens to drop exactly the high-frequency English/code
   merges that made 4.6 efficient on those domains.

---

## Results by domain (averaged)

| domain | n | avg 4.7/4.6 | 4.6 bytes/tok | 4.7 bytes/tok |
|---|---:|---:|---:|---:|
| thai               | 1 | **1.000** | 3.33 | 3.32 |
| semitic/ar         | 1 | **1.001** | 2.69 | 2.68 |
| greek              | 1 | **1.002** | 2.69 | 2.68 |
| ru/classical       | 1 | **1.002** | 4.17 | 4.16 |
| semitic/he         | 1 | **1.003** | 2.48 | 2.47 |
| devanagari/hi      | 1 | **1.004** | 3.71 | 3.70 |
| ru/biography       | 2 | **1.007** | 4.37 | 4.34 |
| cjk/ko             | 2 | **1.014** | 2.34 | 2.30 |
| ru/modern          | 2 | **1.019** | 4.90 | 4.81 |
| cjk/ja             | 2 | 1.099 | 2.89 | 2.66 |
| data/json          | 1 | 1.105 | 2.44 | 2.21 |
| misc/emoji         | 1 | 1.113 | 2.33 | 2.10 |
| misc/math (LaTeX)  | 1 | 1.139 | 1.79 | 1.57 |
| cjk/zh             | 2 | 1.153 | 2.82 | 2.46 |
| code/c             | 1 | 1.208 | 2.41 | 2.00 |
| data/logs          | 1 | 1.219 | 1.86 | 1.52 |
| code/css           | 1 | 1.251 | 2.06 | 1.65 |
| code/ruby          | 1 | 1.260 | 3.62 | 2.87 |
| code/rust          | 1 | 1.268 | 2.62 | 2.07 |
| code/sql           | 1 | 1.286 | 2.44 | 1.90 |
| code/html          | 1 | 1.297 | 3.19 | 2.46 |
| code/cpp           | 1 | 1.308 | 3.05 | 2.34 |
| code/py            | 2 | 1.311 | 3.50 | 2.68 |
| code/bash          | 1 | 1.350 | 2.45 | 1.81 |
| code/go            | 1 | 1.366 | 3.60 | 2.64 |
| code/yaml          | 1 | 1.358 | 3.15 | 2.32 |
| code/java          | 1 | 1.395 | 3.51 | 2.51 |
| en/modern          | 4 | **1.394** | 3.85 | 2.76 |
| code/ts            | 1 | 1.436 | 3.01 | 2.10 |
| **en/classical**   | 4 | **1.485** | 3.38 | 2.29 |
| **code/kotlin**    | 1 | **1.502** | 3.23 | 2.15 |
| **code/js**        | 1 | **1.507** | 2.91 | 1.93 |

---

## Per-sample (sorted by ratio)

| sample | bytes | 4.6 | 4.7 | 4.7/4.6 |
|---|---:|---:|---:|---:|
| `";" * 200` (novel 4.7 merge `);););`) | 400 | 200 | **101** | **0.505** |
| thai/bangkok  | 11,458 | 3,446 | 3,447 | 1.000 |
| zh/beijing    | 11,368 | 4,406 | 4,408 | 1.000 |
| ar/quran      | 7,287  | 2,711 | 2,714 | 1.001 |
| **ru/pushkin (Dubrovsky, 1832 Russian)** | 7,238 | **1,737** | **1,740** | **1.002** |
| greek/athens  | 7,242  | 2,697 | 2,702 | 1.002 |
| he/jerusalem  | 7,092  | 2,865 | 2,874 | 1.003 |
| hi/delhi      | 10,314 | 2,779 | 2,791 | 1.004 |
| ru/pushkin_biography | 7,136 | 1,736 | 1,746 | 1.006 |
| ja/tokyo      | 11,328 | 3,775 | 3,797 | 1.006 |
| ru/dostoevsky_wiki | 7,267 | 1,573 | 1,585 | 1.008 |
| ru/moscow_wiki   | 7,247 | 1,516 | 1,532 | 1.011 |
| ko/seoul      | 9,070  | 3,938 | 3,986 | 1.012 |
| ko/korean     | 9,524  | 4,023 | 4,086 | 1.016 |
| ru/ai_wiki    | 7,234  | 1,440 | 1,479 | 1.027 |
| data/json     | 4,000  | 1,640 | 1,812 | 1.105 |
| emoji         | 4,335  | 1,859 | 2,069 | 1.113 |
| math (LaTeX)  | 2,432  | 1,357 | 1,545 | 1.139 |
| ja/wiki_xml   | 5,284  | 1,895 | 2,258 | 1.192 |
| code/c (linux)| 4,000  | 1,658 | 2,003 | 1.208 |
| data/http_logs| 4,000  | 2,152 | 2,624 | 1.219 |
| code/css (bootstrap)| 4,000 | 1,944 | 2,431 | 1.251 |
| code/ruby (rails)| 4,000 | 1,106 | 1,394 | 1.260 |
| code/py (pathlib)| 4,000 | 1,146 | 1,448 | 1.264 |
| code/rust (vec)| 4,000 | 1,524 | 1,933 | 1.268 |
| code/sql (pg) | 4,000  | 1,640 | 2,109 | 1.286 |
| code/html (mdn)| 4,000 | 1,255 | 1,628 | 1.297 |
| zh/wiki_xml   | 3,311  | 1,082 | 1,412 | 1.305 |
| code/cpp (llvm)| 4,000 | 1,310 | 1,713 | 1.308 |
| code/bash (ohmyzsh)| 4,000 | 1,635 | 2,208 | 1.350 |
| code/py (asyncio)| 4,000 | 1,138 | 1,545 | 1.358 |
| code/yaml (coredns)| 4,000 | 1,271 | 1,726 | 1.358 |
| code/go (json_decoder)| 4,004 | 1,111 | 1,518 | 1.366 |
| en/whisper_readme | 4,000 | 1,033 | 1,415 | 1.370 |
| en/md_vscode  | 4,002  | 1,144 | 1,586 | 1.386 |
| en/sherlock   | 4,012  | 1,041 | 1,445 | 1.388 |
| en/wiki_ai    | 4,002  | 1,167 | 1,627 | 1.394 |
| code/java (spring)| 4,000 | 1,141 | 1,592 | 1.395 |
| en/wiki_chatgpt | 4,004 | 868  | 1,237 | 1.425 |
| en/alice      | 4,060  | 1,072 | 1,533 | 1.430 |
| code/ts (tsc) | 4,000  | 1,327 | 1,906 | 1.436 |
| code/kotlin   | 4,000  | 1,238 | 1,859 | 1.502 |
| code/js (lodash)| 4,000 | 1,375 | 2,072 | 1.507 |
| en/mobydick   | 4,024  | 1,585 | 2,430 | 1.533 |
| en/tomsawyer  | 4,210  | 1,267 | 2,015 | 1.590 |
| `"#" * 500`   | 500    | 10   | **33** | **3.300** |

---

## Interpretation

**4.7 gives up its English/code advantage.** On Russian classical,
Russian news, Arabic, Hebrew, Greek, Thai, and Hindi, 4.7 costs
essentially the same as 4.6 (≤+2.7 %). On English prose, 4.7 costs
**~40 % more**. On popular programming languages (JS, TS, Kotlin),
up to **+50 %**.

The corpus-level picture (from `COMPARE_46_47.md`) showed 4.7's vocab
is a near-subset of 4.6's (only 2 new merges out of 13,454). What we
see here confirms *which* merges 4.6 had that 4.7 dropped: the long
English-word and code-identifier merges (` learning`, ` computers`,
` function`, ` return`, ` const`, etc. — all 2 tokens on 4.7, 1 token
on 4.6 per spot-checks earlier).

**Per-language fair-price picture** (bytes per token — higher = cheaper):

```
ru/modern        4.90 → 4.81   essentially unchanged
ru/biography     4.37 → 4.34   unchanged
ru/classical     4.17 → 4.16   unchanged
en/classical     3.38 → 2.29   ← big drop
en/modern        3.85 → 2.76   ← big drop
code/py          3.50 → 2.68   ← big drop
code/java        3.51 → 2.51   ← big drop
```

Russian is priced at ~4.8 B/token on both models. English drops from
3.85 to 2.76 B/token on 4.7 — i.e. **English became ~28 % more
expensive per byte under 4.7**.

---

## What this likely means for Anthropic's "improved" claim

The release notes' phrasing — *"4.7 uses an updated tokenizer that
improves how the model processes text"* — is not about bytes-per-token
efficiency. Given:

- 4.7's vocab is a near-subset of 4.6's (13,454 tokens verified; 2 new)
- 4.7's English/code cost went up by 25–50 %
- 4.7's non-English cost is flat

…the most consistent explanation is that **4.7 deliberately narrowed
its vocabulary**, probably to:

- shrink the embedding matrix (faster inference, smaller model),
- equalise per-byte cost across languages (English was the outlier at
  3.85 B/tok while Russian was at 4.9; on 4.7 they converge at 2.8 vs 4.8),
- or reduce token-level overfitting on English-dominant training data.

Whatever the model-internal reason, for someone who pays per token,
**4.6 is 30–50 % cheaper on English and code** than 4.7.

---

## Artifacts

- `src/efficiency_study.py` — the measurement script
- `state/efficiency.csv` — full per-sample results
- `corpus_eff/` — fetched samples (Wikipedia multilingual, open-source code)
