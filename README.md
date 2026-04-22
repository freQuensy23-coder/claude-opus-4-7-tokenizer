# claude-tokenizer

Reconstruct the Claude Opus 4.7 tokenizer vocabulary from the public
`count_tokens` API via Gupta-style sandwich probing.

**Result: 13,454 verified single-tokens, ~85% greedy efficiency on live API.**

## Running

```bash
export ANTHROPIC_API_KEY=sk-ant-...
./run.sh all                 # full pipeline (resume-safe)
./run.sh evaluate            # evaluate current vocab_47.json vs live API
./run.sh port|context|...    # run a specific phase
```

## Project layout

```
claude-tokenizer/
├── README.md                    # this file
├── pyproject.toml / uv.lock     # deps (managed with uv)
├── run.sh                       # main entry point
├── vocab_47.json                # ⭐ main artifact: 13,454 verified 4.7 tokens
├── vocab_46.json                # reference: Gupta's 4.6 vocab (38,360 tokens)
│
├── src/                         # all Python source
│   ├── pipeline.py              #   phase orchestrator
│   ├── counter.py               #   async sandwich counter + adaptive rate limiter
│   ├── candidates.py            #   candidate generators (tiktoken, HF, context, ngram, ...)
│   ├── corpus.py                #   embedded + external corpus loader
│   ├── greedy.py                #   trie + greedy longest-match tokenizer
│   ├── store.py                 #   rotating CSV+gzip persistence
│   ├── compare_46_47.py         #   4.6 vs 4.7 head-to-head analysis
│   ├── validate_novel.py        #   validate 4.7-novel tokens against both models
│   └── probe_*.py               #   one-off probe scripts
│
├── reference/                   # input seed (read-only)
│   ├── vocab.json               #   Gupta's 4.6 verified vocab (38,360)
│   ├── ctoc.cc / gen_vocab.py   #   Gupta's original C++ / Python tooling
│   └── REPORT.md                #   Gupta's technical report
│
├── corpus_data/                 # external corpora for mining (~4.3 MB)
│   ├── alice.txt, mobydick.txt, sherlock.txt, tomsawyer.txt   # Gutenberg
│   ├── wiki_ai.xml, wiki_py.xml, wiki_ru.xml, wiki_ja.xml, … # Wikipedia
│   └── code_asyncio.py, code_linux.c, …                      # source code
│
├── state/                       # probe checkpoints (resume-safe)
│   └── phase_*.NNNN.csv[.gz]    #   per-phase text→count records,
│                                #   rotating at 100 MB + gzip
│
├── data/
│   └── outputs/                 # auxiliary JSON outputs
│       └── opus_46_new_tokens.json   # tokens 4.7 has that 4.6 doesn't
│
├── docs/                        # all long-form documentation
│   ├── methods.md               #   short description of every method used
│   ├── RESULTS.md               #   final numbers + per-phase yields
│   ├── COMPARE_46_47.md         #   4.6 vs 4.7 tokenizer comparison
│   └── VALIDATE_NOVEL.md        #   validation of 4.7-novel tokens
│
├── scripts/                     # helper shell scripts
│   ├── run_all_after.sh         #   chained runner
│   └── status.sh                #   progress snapshot
│
└── logs/                        # pipeline run logs (log per invocation)
```

## Pipeline phases (in order)

1. **calibrate** — confirm §§ baseline and spot-check known tokens
2. **port** — re-verify Gupta's 4.6 verified set against 4.7 (seed)
3. **context** — trailing-space/punct/morpheme variants of confirmed tokens
4. **multilingual** — bigrams from Cyrillic / Greek / Kana / Arabic / Hebrew
5. **handcrafted** — whitespace/digits/repeated-chars/Unicode probes
6. **kgram** — frequency-sorted word-anchored k-grams from corpus
7. **corpus** — iterative adjacent-pair mining from real text
8. **boundary** — sub-chunk search on over-segmented windows
9. **evaluate** — greedy vs API on diverse samples

See [docs/methods.md](docs/methods.md) for the full catalog of every method
(including the constraint-based / DP / triangulation miners used post-pipeline).

## Artifacts

- `vocab_47.json` — `{"verified": [...13,454...], "model": "claude-opus-4-7", "baseline": 13, "count": 13454}`
- `vocab_46.json` — Gupta's 4.6 vocab re-packaged (38,360 tokens)
- `state/phase_*.csv[.gz]` — full `text → count` map for all 1.37M probes
  (rotating CSV, gzipped when a segment exceeds 100 MB)
