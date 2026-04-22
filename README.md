# claude-opus-4-7-tokenizer

Reconstruction of the **Claude Opus 4.7** BPE tokenizer vocabulary via the
public `count_tokens` API.

Extends Rohan Gupta's sandwich-probing method
([rohangpta/ctoc](https://github.com/rohangpta/ctoc)) to 4.7.

## Headline numbers

| Metric | Value |
|---|---:|
| Verified 4.7 single-tokens | **14,036** |
| Total API probes | ~1.37 M |
| Weighted greedy efficiency vs live API | **~85 %** |
| Live-API validation on random 500-token sample | **100 %** |
| Reference: Gupta's 4.6 vocab (this repo also ships it) | 38,360 |

Per-domain greedy efficiency (greedy-token-count / API-token-count):

| Domain | Efficiency |
|---|---:|
| Python / TypeScript / JSON / markdown | 93 – 100 % |
| English prose (short) | ~75 % |
| Long English | ~82 % |
| CJK, Japanese | ~75 % |
| Russian | ~57 % |

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full per-phase yield breakdown
and [`docs/methods.md`](docs/methods.md) for the catalog of every method used.

## 4.6 vs 4.7 — what actually changed

Once the 4.7 vocabulary was reconstructed, we compared it directly against
Gupta's 4.6 vocabulary and ran a live-API efficiency study across 46
samples in 30+ domains. Full tables in
[`docs/COMPARE_46_47.md`](docs/COMPARE_46_47.md),
[`docs/VALIDATE_NOVEL.md`](docs/VALIDATE_NOVEL.md), and
[`docs/EFFICIENCY.md`](docs/EFFICIENCY.md).

### 1. The 4.7 vocabulary is nearly a subset of 4.6's

Of **14,036** sandwich-reachable 4.7 tokens, we validated each against both
models' `count_tokens` endpoints. All but two are also single tokens on 4.6:

| 4.7 tokens | count |
|---|---:|
| in Gupta's 4.6 verified set | 7,569 |
| not in Gupta but **confirmed single on 4.6** via live API | ~6,465 |
| **genuinely novel to 4.7** (single on 4.7, NOT single on 4.6) | **2** |

The two net-new merges are `################` (16 × `#`) and `);););`
(`)` `;` `)` `;`). Every other verified 4.7 token was already a 4.6 token —
Gupta's cross-tokenizer mining simply never probed most of them.

### 2. Live-API efficiency, 4.7 vs 4.6 (ratio = tokens(4.7) / tokens(4.6))

|| avg ratio ||
|---|---:|---|
| thai / arabic / greek / hebrew / hindi | **1.00–1.00** | no change |
| russian (classical, biography, modern) | **1.00–1.02** | no change |
| korean | 1.01 | no change |
| japanese | 1.10 | +10 % |
| chinese | 1.15 | +15 % |
| emoji / math / JSON / logs | 1.11–1.22 | |
| code (C, CSS, Ruby, Python, Rust, SQL, HTML, C++, Bash, YAML, Go, Java, TS, JS, Kotlin) | **1.21–1.51** | up to +51 % |
| **english prose (classical + modern)** | **1.39–1.49** | **+40–49 %** |
| `# × 500` | 3.30 | worst outlier |

4.7 is **essentially tied with 4.6 on every non-English non-Latin script**,
and **25–50 % worse on English and code**.

### 3. The mechanism: base English merges dropped; casing markers kept

4.6 compressed common English words as single whole-word tokens
(`"learning"`, `"function"`, `"algorithm"` = 1 token each). 4.7 dropped
most of those merges. Live probes:

| word | 4.6 lower | 4.7 lower | 4.6 UPPER | 4.7 UPPER |
|---|---:|---:|---:|---:|
| `hello` | 1 | 2 | 2 | 4 |
| `function` | 1 | 1 | 2 | 6 |
| `algorithm` | 1 | **4** | 2 | **7** |
| `understanding` | 1 | **3** | 2 | **9** |
| `NULL_POINTER_EXCEPTION` | 7 | **16** | — | — |

The **"+1 per word" all-caps penalty is identical on both models** — 4.6
and 4.7 each add exactly one token when an N-word lowercase phrase is
converted to all-caps, regardless of N. That's the signature of an
Anthropic-specific Caps Lock / morphology marker (previously documented
by [Sander Land](https://tokencontributions.substack.com/p/whole-words-and-claude-tokenization)),
and it survives intact in 4.7. What went away is the *base* English whole-
word vocabulary that the marker was paired with.

### 4. Why? (speculation)

Anthropic's release-note wording — *"4.7 uses an updated tokenizer that
improves how the model processes text"* — is not about bytes-per-token
efficiency. Candidate reasons the vocab was narrowed:

- **Smaller embedding matrix.** Fewer whole-word English merges → smaller
  vocab → smaller input/output embedding tables → faster to train and serve.
- **Cross-lingual fairness.** 4.6's English was the outlier at 3.85 B/tok
  while Russian was at 4.90 B/tok. On 4.7 they converge at 2.76 vs 4.81
  (the disparity roughly halves).
- **Instruction-following regularisation.** Anthropic's migration guide
  calls out *"more literal instruction following"* as a 4.7 property.
  Fewer memorised whole-word tokens forces compositional subword
  tokenisation, which plausibly reduces overfitting to specific phrasings.
- [Artificial Analysis data (via HN)](https://news.ycombinator.com/item?id=47816960)
  shows 4.7 producing **~38 % fewer output tokens** than 4.6 on the same
  benchmark suite. Input fertility up, output fertility down.

## Running

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run.sh all                 # full pipeline, resume-safe
./run.sh evaluate            # compare vocab_47.json vs live API on 10 samples
./run.sh port|context|...    # single phase
```

Dependencies managed via [uv](https://github.com/astral-sh/uv); `pyproject.toml`
declares `httpx`, `tiktoken`, `transformers`, `sentencepiece`, `protobuf`,
`tqdm`. All state is written to `state/phase_*.csv[.gz]` so interrupting and
resuming is safe.

## Method in 60 seconds

1. **Sandwich primitive** — `count(x) = raw("§x§") – raw("§§")`. `§` (U+00A7)
   doesn't merge with its neighbors, so subtracting the `§§` baseline yields
   the isolated token count of `x`. Baseline on 4.7 = 13.
2. **Port** — every string in Gupta's 4.6 vocab re-probed against 4.7; the
   survivors (7,569 of 38,360) form the seed.
3. **Context expansion** — for every confirmed 4.7 token `T`, probe
   `T+suffix`, `"_"+T+suffix` etc. Each distinct byte sequence is a distinct
   BPE vocab entry — e.g. `history`, `_history`, `history_`, `_history_`
   are four separate tokens.
4. **Candidate mining** — tiktoken + 11 HF tokenizers, word-anchored k-grams
   from a 4.3 MB mixed corpus, adjacent-pair concatenations, multilingual
   bigrams, constraint satisfaction over 1.37 M existing `(string, count)`
   records.

Full method catalog in [docs/methods.md](docs/methods.md).

## What this can **not** do

4.7 uses many **context-dependent BPE merges**: common words like
`" quick"`, `" learning"`, `" journey"` return `count ≥ 2` in *isolation*
but are encoded as single merges when adjacent to specific bytes. These
tokens are structurally invisible to any `count_tokens`-only methodology
— proven experimentally by probing every substring of every eval sentence
and getting zero new hits.

Estimated true 4.7 vocab size: ~25–30 K. This repo recovers the
~13–14 K sandwich-reachable subset; the remainder requires API
capabilities Anthropic doesn't expose (logprobs, fine-grained streaming
deltas, or a tokenize endpoint).

## Repo layout

```
├── vocab_47.json            ⭐ 14,036 verified 4.7 single-tokens
├── vocab_46.json               38,360 tokens (Gupta's 4.6 reference)
├── run.sh                      main entry
├── src/                        pipeline.py, counter.py, candidates.py,
│                               corpus.py, greedy.py, store.py, probe_*.py,
│                               compare_46_47.py, validate_novel.py
├── reference/                  Gupta's ctoc.cc, vocab.json, REPORT.md
├── corpus_data/                4.3 MB external corpus (Gutenberg +
│                               Wikipedia + CPython + multilingual)
├── state/                      probe checkpoints, one CSV per phase
│                               (rotating at 100 MB, gzipped)
├── data/outputs/               auxiliary analyses
│   └── opus_46_new_tokens.json
├── corpus_eff/                 fetched multilingual Wikipedia + open-source
│                               code samples used by the efficiency study
├── docs/
│   ├── methods.md              all methods with yields
│   ├── RESULTS.md              final numbers
│   ├── COMPARE_46_47.md        4.6 vs 4.7 comparison
│   ├── VALIDATE_NOVEL.md       live-API validation of 4.7-novel tokens
│   └── EFFICIENCY.md           46-sample, 30-domain 4.6-vs-4.7 efficiency study
├── scripts/                    status.sh, run_all_after.sh
└── logs/                       one log per pipeline invocation
```

## Reproducing

```bash
git clone https://github.com/freQuensy23-coder/claude-opus-4-7-tokenizer
cd claude-opus-4-7-tokenizer
export ANTHROPIC_API_KEY=sk-ant-...
./run.sh evaluate          # verify published vocab against live API
./run.sh all               # re-run the full pipeline (~3–6 h with
                           #   adaptive rate limiting; resume-safe)
```

## Acknowledgments

- Rohan Gupta for [ctoc](https://github.com/rohangpta/ctoc) and the
  `§§` sandwich technique.
- Javier Rando for the streaming-based probing approach (Claude 3).
- Sander Land for [whole-word Claude tokenization observations](https://tokencontributions.substack.com/p/whole-words-and-claude-tokenization).

## License

MIT — see [LICENSE](LICENSE).
