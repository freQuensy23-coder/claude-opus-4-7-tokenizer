# Validation of 4.7-novel tokens via live count_tokens

Re-probed all **5,885 tokens** in `vocab_47.json \ vocab_46.json` via the sandwich
protocol against *both* models' `count_tokens` endpoints.

## 1. Per-model sandwich count distribution

| count on model | 4.7 | 4.6 |
|---|---:|---:|
| `>=2 (got 2)` | 0 | 2 |
| `single` | 5,885 | 5,883 |

## 2. Joint classification

- **Single on 4.7, NOT single on 4.6 (genuinely novel)**: 2 (0.0%)
- **Single on 4.7 AND on 4.6 (Gupta missed in 4.6)**: 5,883 (100.0%)
- **Not single on 4.7 (false positive in our vocab)**: 0 (0.0%)
- **Error on either pass**: 0 (0.0%)

## 3. Examples

### Truly novel in 4.7 (single on 4.7, >1 on 4.6)

- `'################'` — on 4.6: 2 tokens
- `'););'` — on 4.6: 2 tokens

### Shared: single on both (Gupta missed in 4.6 probing)

- `'               \r'`
- `'\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0...`
- `'características'`
- `'International '`
- `' application '`
- `' association '`
- `' description '`
- `' development '`
- `' established '`
- `' information '`
- `' października'`
- `'           \r'`
- `' California '`
- `' References '`
- `' University '`
- `' background '`
- `' california '`
- `' components '`
- `' department '`
- `' government '`
- `' households '`
- `' management '`
- `' population '`
- `' production '`
- `' references '`
- `' television '`
- `' university '`
- `'application '`
- `'association '`
- `'description '`

### False positives (claimed single, actually not)

