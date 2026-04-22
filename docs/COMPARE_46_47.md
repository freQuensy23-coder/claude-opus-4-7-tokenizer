# Claude Opus 4.6 vs 4.7 Tokenizer Comparison (Gupta's method)

Both vocabularies were recovered by **Gupta's `§§` sandwich probing** of the
`count_tokens` API.

| File | Source | Verified tokens |
|---|---|---:|
| `vocab_46.json` | Rohan Gupta, [rohangpta/ctoc](https://github.com/rohangpta/ctoc) `vocab.json@main` | 38,360 |
| `vocab_47.json` | this project (partial; see `RESULTS.md`) | 13,193 |

`vocab_46.json` is byte-identical to `reference/vocab.json` (sha256 prefix
`345cc6e7…`); no newer upstream exists.

---

## 1. Set overlap

|  | count | % of 4.6 | % of 4.7 |
|---|---:|---:|---:|
| `|4.6 ∩ 4.7|` | 7,569 | 19.7% | **57.4%** |
| `|4.6 \ 4.7|` (in Gupta, absent from our 4.7) | 30,791 | 80.3% | — |
| `|4.7 \ 4.6|` (**novel to 4.7**) | **5,624** | — | 42.6% |
| Jaccard | 0.172 | | |

**Interpretation.** Of the 13,193 tokens we verified for 4.7, 57% were carried
over from 4.6 and 43% are novel. But only 20% of Gupta's 4.6 set survived
re-verification against 4.7's API — i.e. **~80% of 4.6 tokens no longer
sandwich as single tokens in 4.7**. This matches the `port` phase yield in
`RESULTS.md` (7,569 / 38,360 = 19.7%).

---

## 2. Length distribution

```
  len      4.6      4.7
    1    1,888    1,916    ← singletons preserved
    2    2,179    1,817
    3    4,214    3,074
    4    3,865    1,663
    5    4,836    1,348
    6    5,030    1,190
    7    4,719      900
    8    3,957      632
    9    2,972      361
  10+    4,700      292
```

4.6's distribution peaks at length 5–6 (typical BPE); 4.7 is **top-heavy on
length ≤ 3** because our reconstruction couldn't isolate the long
context-dependent merges (`" learning"`, `" computers"`, `" journey"`, etc.)
— they survive only in context. This is the methodological ceiling noted in
`RESULTS.md §Gap Analysis`.

---

## 3. Script distribution

| bucket | 4.6 | | 4.7 | | delta |
|---|---:|---:|---:|---:|---|
| ascii      | 35,650 | 92.9% | 9,330 | 70.7% | ASCII-heavy in Gupta |
| cjk        | 1,135  | 3.0%  | 1,135 | **8.6%** | 3× share in 4.7 |
| cyrillic   | 422    | 1.1%  | 1,024 | **7.8%** | 7× share; **+142%** absolute |
| latin_ext  | 135    | 0.4%  | 263   | 2.0%  | +95% absolute |
| hiragana   | 106    | 0.3%  | 197   | 1.5%  | +86% absolute |
| arabic     | 41     | 0.1%  | 67    | 0.5%  | +63% |
| greek      | 33     | 0.1%  | 62    | 0.5%  | +88% |
| hangul     | 379    | 1.0%  | 379   | 2.9%  | 1:1, share up |
| hebrew     | 18     | 0.0%  | 28    | 0.2%  | +56% |

**4.7 is substantially richer in multilingual merges** — even though our
verified set is a third the size of Gupta's, it contains **more** Cyrillic,
Latin-extended, Hiragana, Arabic, Greek, and Hebrew tokens in absolute terms.
This aligns with Anthropic's note that "4.7 uses an updated tokenizer that
improves how the model processes text" — the improvement is weighted toward
non-English.

---

## 4. Space-prefix & casing

|  | 4.6 | 4.7 |
|---|---:|---:|
| space-prefixed | 19,148 (49.9%) | 4,881 (37.0%) |
| cap-initial    | 9,705 (25.3%)  | 1,133 (8.6%)  |

The cap-initial drop is again a coverage artifact: most capitalised English
words in our 4.7 set sandwich-fail and stay unverified.

---

## 5. Sample tokens

**Novel in 4.7** (4.7-only), short ASCII examples:
- `' " '`, `' & '`, `' * '`, `' + '`, `' . '`, `' : '`, `' < '`, `' > '`, `' I '`
- Trailing-`\r` variants: `')\r'`, `',\r'`, `':\r'`, `';\r'`, `'>\r'`

→ 4.7 added many **trailing-space punctuation** and **CR-suffixed**
delimiter merges.

**Novel in 4.7**, non-ASCII:
- Cyrillic letter + space:  `'І '  'А '  'В '  'З '  'И '`
- Hiragana letter + space:  `'い '  'う '  'え '  'お '  'か '`
- Arabic letter + space:    `'، '  'د '  'س '  'م '  'و '`
- Hebrew letter + space:    `'א '  'י '  'או '  'את '`
- Greek letter + space:     `'; '  '· '  'α '  'β '  'η '`

→ 4.7 introduced a consistent family of **single-letter-plus-space merges
across many scripts** — word-boundary merges that 4.6 didn't have.

**Gone from 4.7** (present in Gupta 4.6, not re-verified):
- `' ""', ' "#', ' "$', ' "%', ' "&'` — Gupta-era triple-merges of
  ` "` + punct. In 4.7 these appear to split differently.
- Cyrillic stems: `'ѝ', 'Дани', 'дани', 'рата', ' СССР'` — long Cyrillic
  word-fragments replaced by the shorter single-letter merges above.

---

## 6. Greedy head-to-head on `corpus_data/` (3.98 MB)

Greedy longest-match tokenisation using each vocab's trie (the algorithm in
`src/greedy.py`, same as Gupta's ctoc):

| file | bytes | 4.6 toks | 4.7 toks | 4.7/4.6 |
|---|---:|---:|---:|---:|
| alice.txt          |   151,191 |   41,783 |   62,553 | 1.497 |
| code_asyncio.py    |    81,634 |   20,392 |   31,101 | 1.525 |
| code_linux.c       |    85,102 |   34,586 |   50,009 | 1.446 |
| code_pathlib.py    |    54,240 |   15,021 |   21,834 | 1.454 |
| mobydick.txt       | 1,234,609 |  349,409 |  549,823 | 1.574 |
| readme_whisper.md  |     8,246 |    2,340 |    3,605 | 1.541 |
| sherlock.txt       |   595,198 |  157,075 |  247,299 | 1.574 |
| tomsawyer.txt      |   425,088 |  118,482 |  182,716 | 1.542 |
| wiki_ai.xml        |   276,043 |   90,016 |  147,547 | 1.639 |
| wiki_ja.xml        |   228,009 |  101,025 |  120,292 | **1.191** |
| wiki_py.xml        |   149,044 |   49,553 |   75,972 | 1.533 |
| wiki_ru.xml        |   691,130 |  307,438 |  330,219 | **1.074** |
| wiki_zh.xml        |     3,563 |    1,178 |    1,859 | 1.578 |
| **TOTAL** |   3,983,097 | **1,288,298** | **1,824,829** | **1.416** |

Bytes/token: 4.6 = **3.09**, 4.7-partial = **2.18**.

> ⚠️ **Caveat.** This measures *vocab coverage*, not the real 4.6-vs-4.7
> tokenizers. 4.6 "wins" because our 4.7 reconstruction is partial (13K of
> an estimated 25–30K). The actual 4.7 tokenizer — inside the live API —
> beats 4.6 on the same text.
>
> Evidence that the relative gap tracks coverage rather than intrinsic
> quality: the 4.7/4.6 ratio is **worst on English prose** (1.53–1.64) and
> **best on Russian / Japanese** (1.07, 1.19) — the corpora where our
> 4.7 verified set has the highest Cyrillic / Hiragana coverage.

---

## 7. Share of 4.7-greedy tokens that are novel (in `only47`)

| file | 4.7 toks | 4.7-only | %novel |
|---|---:|---:|---:|
| sherlock.txt       | 247,299 |  90,676 | **36.7%** |
| alice.txt          |  62,553 |  22,018 | 35.2% |
| tomsawyer.txt      | 182,716 |  61,920 | 33.9% |
| mobydick.txt       | 549,823 | 175,888 | 32.0% |
| readme_whisper.md  |   3,605 |     824 | 22.9% |
| wiki_ai.xml        | 147,547 |  21,669 | 14.7% |
| code_pathlib.py    |  21,834 |   3,185 | 14.6% |
| wiki_py.xml        |  75,972 |  10,826 | 14.2% |
| code_asyncio.py    |  31,101 |   4,266 | 13.7% |
| code_linux.c       |  50,009 |   5,650 | 11.3% |
| wiki_ru.xml        | 330,219 |  31,900 |  9.7% |
| wiki_zh.xml        |   1,859 |      94 |  5.1% |
| wiki_ja.xml        | 120,292 |   3,496 |  2.9% |

**Interpretation.** On long English prose, **one in three greedy 4.7 tokens
was not verified by Gupta for 4.6** — these are where 4.7's new merges
materially change the segmentation. On Japanese and Chinese the novel-token
fraction is ≤5%, because our 4.7 CJK additions almost all already existed in
4.6 (CJK absolute count is identical: 1,135 vs 1,135; it's the share, not
the rows, that shifted).

---

## Reproduce

```bash
uv run python src/compare_46_47.py > compare_output.txt
```

Artifacts:

- `vocab_46.json` — canonical 4.6 vocab (38,360, Gupta)
- `vocab_47.json` — partial 4.7 vocab (13,193, this project)
- `src/compare_46_47.py` — the comparison script
