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
├── docs/
│   ├── methods.md              all methods with yields
│   ├── RESULTS.md              final numbers
│   ├── COMPARE_46_47.md        4.6 vs 4.7 comparison
│   └── VALIDATE_NOVEL.md       live-API validation of 4.7-novel tokens
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
