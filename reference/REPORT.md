# Reverse-Engineering an Offline Estimator for Anthropic's Token Counter

## Technical Report

### 1. Motivation

Anthropic provides a `count_tokens` API endpoint that returns exact token counts for any input text, but no public tokenizer for Claude 3+. This means developers building applications on Claude — particularly coding agents that need to manage context windows — have no way to estimate token counts without making an API call for every piece of text.

We set out to reverse-engineer Claude's tokenizer vocabulary by probing the `count_tokens` endpoint, with the goal of building an offline estimator that achieves 95%+ accuracy on real-world text.

### 2. The count_tokens API

The API accepts a message and returns the total input token count including message framing overhead:

```python
response = client.messages.count_tokens(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": text}]
)
raw_count = response.input_tokens  # includes framing tokens
```

A single-character message like `"a"` returns 8, meaning there are 7 framing tokens added by the message structure. For a candidate string to be a single token: `raw_count - baseline == 1`.

The API has no per-call charge but requires a positive account balance. Latency is ~150ms per call, giving a practical throughput of ~5-7 checks per second.

### 3. Core Insight: The Boundary Property

BPE token counts are **not monotonic** with string length. Adding a character can decrease the count:

- `"hel"` = 2 tokens, `"hell"` = 1 token (a longer merge becomes available)

This means binary search for token boundaries doesn't work. However, we can exploit a different property:

> **Position i is a token boundary in string s if and only if:**
> `count(s[:i]) + count(s[i:]) == count(s)`

If no token spans the split point, the prefix and suffix tokenize independently and their counts sum to the total. This gives us O(n) boundary detection for any string.

### 4. The Baseline Bug: A Tale of Three Bugs

The seemingly simple formula `token_count = raw_count - baseline` hides three subtle bugs that took significant effort to discover and fix.

#### Bug 1: Variable baseline by character type

The API framing overhead differs based on the **first character** of the message:

| First character | Raw count for single-token text | Effective baseline |
|----------------|-------------------------------|-------------------|
| Letter (a-z, A-Z) | 8 | 7 |
| Digit (0-9) | 9 | 8 |
| CJK / most Unicode | 9+ | 8+ |

Using a single baseline of 7 (calibrated on `"a"`) caused every digit-starting token to appear as 2 tokens, and every CJK character to appear as 2+ tokens.

**Impact:** 1,006 digit-starting tokens (mostly 3-digit numbers like `916`, `030`) were incorrectly rejected during extraction. All Unicode token verification was systematically wrong.

#### Bug 2: Trailing newline stripping

The API silently strips trailing newlines from message content:

```
raw("a")    = 8
raw("a\n")  = 8    # same!
raw("a\n\n") = 8   # same!
```

This made every token ending in `\n` or `\r\n` appear to be a single token when it was actually multiple tokens. For example, `" ))\n"` measured as 1 token via the raw method, but is actually 2 tokens (`" ))"` + `"\n"`).

**Impact:** 683 false-positive tokens ending in newlines were added to the vocabulary.

#### Bug 3: Control character stripping

Characters in the range 0x00-0x1F (except `\n`, `\r`, `\t`) are stripped by the API, returning 0 effective tokens. The old baseline turned these into negative or zero counts that were misinterpreted.

**Impact:** 311 false-positive control character tokens.

#### The Fix: Section Sign Sandwich

We discovered that wrapping text between `§` (U+00A7) markers normalizes the baseline across all character types:

```python
def count_tokens(text: str) -> int:
    sandwich_baseline = count_tokens_raw("§§")  # cached
    sandwiched = count_tokens_raw("§" + text + "§")
    return sandwiched - sandwich_baseline
```

Why `§` works:
- It has a standard baseline (doesn't shift framing overhead)
- It doesn't merge with adjacent characters (no BPE merges across the boundary)
- It prevents trailing newline stripping (the closing `§` is always the last character)

Verification: tested against 1,000 randomly sampled tokens — 99.8-100% accuracy.

### 5. Vocabulary Extraction Methodology

We extracted vocabulary through multiple phases, each targeting a different source of candidates.

#### Phase 1: Cross-reference published tokenizers (~29,000 tokens)

The highest-yield strategy was checking tokens from existing BPE tokenizers against Claude's API. Tokens from one BPE tokenizer are likely to appear in another, since they're all trained on similar internet text.

| Tokenizer | Vocab size | Hit rate | New tokens found |
|-----------|-----------|----------|-----------------|
| tiktoken cl100k_base (GPT-4) | 100K | ~46% | ~29,000 |
| tiktoken o200k_base (GPT-4o) | 200K | ~15% | ~4,000 |
| tiktoken gpt2 | 50K | ~8% | ~500 |

**Technique:** For each token in the source tokenizer, decode the token ID to a UTF-8 string via `tok.decode([token_id])`, filter out strings containing `\ufffd` (incomplete byte sequences), then check `count_tokens(candidate) == 1`.

**Critical detail:** GPT-2-style tokenizers use byte-level BPE with a byte-to-Unicode mapping. You must use `tok.decode([id])`, not the raw vocabulary keys, to get actual strings.

#### Phase 1b: HuggingFace multilingual tokenizers (~3,800 tokens)

We cross-referenced 11 HuggingFace tokenizers, prioritizing multilingual and CJK-heavy models:

| Model | Vocab size | Unicode candidates | Hits |
|-------|-----------|-------------------|------|
| THUDM/glm-4-9b-chat | 151K | 48K | ~400 |
| baichuan-inc/Baichuan2-7B-Base | 126K | 84K | ~350 |
| Qwen/Qwen2.5-0.5B | 152K | 56K | ~300 |
| bigscience/bloom | 251K | 141K | ~250 |
| facebook/mbart-large-50 | 250K | 157K | ~200 |
| google/mt5-base | 250K | 124K | ~150 |
| + 5 more models | | | |

Sorting candidates by cross-tokenizer frequency (tokens appearing in more tokenizers checked first) gave a 74% hit rate on the first 1,000 candidates, declining to ~2% by 20K candidates.

#### Phase 1c: Common English words and programming identifiers (~500 tokens)

Generated candidates from word frequency lists and programming keywords, checking both raw and space-prefixed variants (`"function"`, `" function"`, `"Function"`, `" Function"`).

#### Phase 2: Digit token recovery (~1,006 tokens)

After fixing the baseline bug, we re-checked all 1,657 digit-starting candidates that had been previously rejected. 1,006 were actually single tokens — almost all 3-digit numbers (e.g., `"916"`, `"030"`, `"271"`). The old baseline of 7 made these appear as 2 tokens when they're actually 1.

#### Phase 3: Repeated-character probing (~139 tokens)

BPE merges create single tokens for specific lengths of repeated characters. These follow non-monotonic patterns:

```
"=" * 1  → 1 token     "=" * 5  → 2 tokens    "=" * 9  → 1 token
"=" * 2  → 1 token     "=" * 6  → 1 token     "=" * 10 → 2 tokens
"=" * 3  → 1 token     "=" * 7  → 2 tokens    ...
"=" * 4  → 1 token     "=" * 8  → 1 token     "=" * 64 → 1 token
```

We systematically probed 14 characters (`= - * # / _ ~ . + > < | \ !`) at lengths 1-99, finding single tokens up to length 64. The `-` character has the most merge points (21 distinct single-token lengths).

#### Phase 4: Whitespace tokens (manual probing)

Using the sandwich method with different markers, we verified:
- Space sequences: 1-16 consecutive spaces are each a single token
- Tabs: 1-4 consecutive tabs are each a single token
- Newlines: `\n`, `\n\n`, `\r\n` are single tokens

These whitespace tokens had a massive impact on code tokenization accuracy, since every indented line in Python uses 4-16 leading spaces.

### 6. Results

#### Final vocabulary

| Category | Count |
|----------|-------|
| ASCII tokens | 33,339 |
| Unicode tokens | 3,156 |
| **Total verified** | **36,495** |
| Candidates checked | 276,640 |

Token length distribution peaks at 5-7 characters, with a long tail up to 64 (repeated-character tokens):

```
Length 1:  1,873    Length 6:  4,965    Length 11: 1,219
Length 2:  1,550    Length 7:  4,667    Length 12:   683
Length 3:  3,464    Length 8:  3,916    Length 13:   368
Length 4:  3,742    Length 9:  2,940    ...
Length 5:  4,753    Length 10: 2,039    Length 64:     6
```

50% of tokens are space-prefixed (18,327), reflecting how BPE tokenizers represent word boundaries.

#### Offline estimator accuracy

We built a greedy longest-match tokenizer using the extracted vocabulary. No merge table is needed — just always pick the longest vocabulary match at each position.

| Corpus | Greedy efficiency | Notes |
|--------|------------------|-------|
| Python source code (9 files) | **96.1%** | This project's source |
| Mixed code + docs (9 files) | **95.1%** | External project (telos) |
| English prose (5 samples) | **99.2%** | General text |

"Efficiency" = `API_count / greedy_count × 100%`. Values below 100% mean the greedy tokenizer over-segments (uses more tokens than the real tokenizer). Values above 100% are rare and mean the greedy tokenizer found shorter segmentations than BPE.

#### DP optimal tokenization adds minimal improvement

We also implemented dynamic programming optimal tokenization — finding the globally minimum-token segmentation given the vocabulary:

| File | API | Greedy | DP optimal |
|------|-----|--------|-----------|
| tokenizer.py (2,331 tokens) | 2,331 | 2,409 (96.8%) | 2,406 (96.9%) |
| extract_vocab.py (10,024 tokens) | 10,024 | 10,338 (97.0%) | 10,296 (97.4%) |

DP only improves by ~0.3-0.4% over greedy. This proves the remaining gap is **missing vocabulary**, not a suboptimal tokenization algorithm. Greedy longest-match is effectively optimal given the vocabulary we have.

### 7. Key Technical Insights

#### BPE is greedy-compatible (mostly)

A surprising finding: greedy longest-match tokenization achieves 95-96% of BPE's output without knowing the merge order. This works because BPE's merge rules tend to produce tokens that are also the longest matches at each position. The cases where they diverge (maybe 4-5% of tokens) involve context-dependent merges where BPE's bottom-up approach finds different boundaries than left-to-right greedy.

#### The remaining 4% gap is specific subwords

The cases where our estimator disagrees with the API are consistently:
1. **ALL-CAPS words** like `STRATEGY` (2 BPE tokens) — our greedy tokenizer doesn't know the correct subword split, so it falls back to character-by-character
2. **Uncommon word fragments** that BPE merges in context but aren't in our vocabulary as standalone tokens
3. **Long repeated characters** where BPE picks different length combinations than our greedy approach

#### Cross-tokenizer mining is the dominant strategy

Of the 36,495 tokens we found, roughly 34,000 (~93%) came from cross-referencing other published tokenizers. Direct probing strategies (boundary-finding, Unicode scanning, BFS extension) had orders-of-magnitude lower yield per API call. The insight: BPE tokenizers trained on similar data converge on similar vocabularies.

#### The API has undocumented behaviors

Three undocumented behaviors of the `count_tokens` endpoint significantly affected our methodology:
1. **Variable framing overhead** by first character type
2. **Trailing newline stripping**
3. **Control character stripping** (0x00-0x1F except \t, \n, \r)

These aren't bugs in the API — they likely reflect preprocessing in Claude's message handling. But they create subtle measurement errors that compound across thousands of checks.

### 8. Architecture

```
ctoc.cc               Single-file C++17 CLI (~500 lines)
  main()                Argparse, orchestrate, print table
  Trie                  Vocabulary stored as a trie for O(n) tokenization
  count_tokens()        Greedy longest-match walk over trie
  discover_files()      std::filesystem recursive directory traversal
  detect_language()     Extension -> language name mapping
  print_summary()       cloc-style formatted output
  print_by_file()       Per-file formatted output

vocab.json            Vocabulary: {"verified": [...], "checked": [...]}

MODULE.bazel          Bazel build config (hermetic_cc_toolchain / zig c++)
BUILD.bazel           cc_binary target
.bazelrc              C++17 flags, cross-compilation platform configs
```

### 9. Limitations and Future Work

**Current limitations:**
- Coverage on non-English text is lower (3,156 Unicode tokens vs ~33K ASCII)
- ALL-CAPS and uncommon subwords cause over-segmentation
- No merge order means we can't perfectly reconstruct BPE's context-dependent splits
- The API could change its tokenizer at any time, invalidating the vocabulary

**Potential improvements:**
- **More Unicode extraction:** Only ~28K of 392K Unicode candidates have been checked. Continuing could add 1-2K more tokens.
- **Subword boundary-finding:** For commonly over-tokenized words, use `find_boundaries_linear()` to discover the correct BPE subword splits.
- **Merge order estimation:** With enough boundary data, it may be possible to partially reconstruct BPE's merge table, improving accuracy beyond 96%.
- **Streaming API extraction:** Javier Rando demonstrated extracting tokens via the streaming API's chunk boundaries — a complementary approach that could validate and extend our results.

### 10. Prior Work

- **Sander Land** (2024): Probed Claude's tokenizer via billing metadata and later the `count_tokens` endpoint. Found ~22K base tokens and documented Claude's preference for whole English words as single tokens.
- **Javier Rando** (2024): Used the streaming API's token-by-token output to extract vocabulary — a fundamentally different approach from our count-based probing.
- **Published tokenizers:** Anthropic published `@anthropic-ai/tokenizer` (npm) and `Xenova/claude-tokenizer` (HuggingFace) for Claude 1/2, but Claude 3+ uses a different, unpublished tokenizer.

### Appendix: Reproduction

```bash
# Install dependencies
pip install anthropic tiktoken transformers sentencepiece protobuf

# Set API key
export ANTHROPIC_API_KEY=sk-...

# Run extraction (takes ~6 hours with API calls)
python extract_vocab.py

# Or use the pre-extracted vocabulary
python -c "
from tokenizer import greedy_tokenize
import json
vocab = set(json.load(open('vocab.json'))['verified'])
text = 'Hello, world!'
tokens = greedy_tokenize(text, vocab)
print(f'{len(tokens)} tokens: {tokens}')
"
```
