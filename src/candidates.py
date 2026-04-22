"""Candidate generators: tiktoken vocabularies, HuggingFace multilingual
tokenizers, and hand-crafted probe sets.

Each generator yields strings; the caller dedupes and filters.
"""
from __future__ import annotations

import string
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


def tiktoken_candidates(encoding_name: str) -> Iterator[str]:
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    n = enc.n_vocab
    for tid in range(n):
        try:
            s = enc.decode([tid])
        except Exception:
            continue
        if not s or "\ufffd" in s:
            continue
        yield s


HF_MODELS = [
    # Chosen for:
    #  - open access (no HF auth or license gate)
    #  - large / multilingual vocab
    #  - diversity (CJK, Indic, code, multilingual, modern frontier)
    "Qwen/Qwen2.5-0.5B",
    "bigscience/bloom",
    "facebook/mbart-large-50",
    "google/mt5-base",
    "xlm-roberta-base",
    "baichuan-inc/Baichuan2-7B-Base",
    "deepseek-ai/DeepSeek-V2-Lite",
    "Salesforce/codet5p-220m",
    "EleutherAI/gpt-neox-20b",
    "microsoft/deberta-v3-base",
    "sentence-transformers/LaBSE",
]


def hf_candidates(model: str) -> Iterator[str]:
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
    except Exception as e:
        print(f"[hf] skip {model}: {e}", flush=True)
        return
    vocab = tok.get_vocab()
    # Decode each id; use convert_ids_to_tokens as fallback.
    for tid in vocab.values():
        try:
            s = tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not s or "\ufffd" in s:
            continue
        yield s


def rank_by_cross_frequency(sources: dict[str, Iterable[str]]) -> list[str]:
    """Return candidates sorted by number of sources they appear in (desc),
    then length desc. Input: {source_name: iterable of strings}."""
    counts: Counter[str] = Counter()
    for _, it in sources.items():
        for s in set(it):
            counts[s] += 1
    return [s for s, _ in sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))]


ASCII_PRINTABLE = string.printable


def repeated_chars(chars: str = "=-*#/_~.+><|\\!_ \t", max_len: int = 99) -> Iterator[str]:
    for ch in chars:
        for n in range(1, max_len + 1):
            yield ch * n


def whitespace_probes() -> Iterator[str]:
    for n in range(1, 33):
        yield " " * n
    for n in range(1, 9):
        yield "\t" * n
    for a in (" ", "\t"):
        for b in ("\n", "\r\n"):
            yield a + b
            yield b + a
    for n in range(1, 5):
        yield "\n" * n
    yield "\r\n\r\n"


def digit_probes() -> Iterator[str]:
    # All 1-3 digit sequences (incl. leading zeros).
    for n in range(10):
        yield str(n)
    for a in range(10):
        for b in range(10):
            yield f"{a}{b}"
    for a in range(10):
        for b in range(10):
            for c in range(10):
                yield f"{a}{b}{c}"
    # 4-digit years.
    for y in range(1800, 2100):
        yield str(y)
    # Common 4-digit.
    for n in (1000, 2000, 5000, 10000, 100000):
        yield str(n)


def identifier_probes() -> Iterator[str]:
    # Common English words + variants.
    words = [
        "the", "and", "for", "with", "that", "this", "from", "have", "are", "was", "were",
        "been", "not", "all", "can", "has", "but", "you", "your", "our", "their", "they",
        "would", "should", "could", "will", "about", "what", "when", "where", "which",
        "who", "why", "how", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "first", "last", "next", "some", "any", "other", "many", "much",
        "more", "most", "less", "least", "time", "year", "day", "week", "month", "people",
        "world", "country", "city", "home", "work", "school", "state", "part", "life",
        "hand", "place", "case", "fact", "group", "problem", "system", "program", "question",
        "right", "wrong", "number", "point", "water", "money", "story", "example", "person",
        "woman", "man", "child", "kind", "end", "word", "line", "idea", "family", "head",
    ]
    programming = [
        "function", "return", "class", "struct", "import", "export", "const", "let", "var",
        "def", "async", "await", "yield", "lambda", "None", "True", "False", "null",
        "undefined", "self", "this", "new", "delete", "static", "public", "private",
        "protected", "void", "int", "string", "String", "bool", "float", "double", "char",
        "print", "println", "log", "error", "warn", "info", "debug", "assert", "throw",
        "catch", "try", "finally", "else", "elif", "switch", "case", "break", "continue",
        "while", "for", "in", "of", "do", "if", "then", "from", "as", "with", "raise",
        "pass", "global", "nonlocal", "is", "not", "and", "or", "xor", "True", "False",
        "list", "dict", "set", "tuple", "array", "Array", "List", "Dict", "Set", "Map",
        "key", "value", "index", "item", "count", "length", "size", "len", "append", "push",
        "pop", "shift", "unshift", "slice", "splice", "filter", "map", "reduce", "find",
        "sort", "reverse", "join", "split", "replace", "match", "test", "exec",
        "token", "tokens", "vocab", "model", "API", "api", "response", "request", "client",
        "server", "user", "admin", "config", "settings", "env", "path", "file", "dir",
        "util", "helper", "handler", "manager", "service", "controller", "router", "middleware",
    ]
    for w in words + programming:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize(), w.upper(), " " + w.upper()):
            yield variant


def unicode_probes() -> Iterator[str]:
    # Single printable Unicode code points from common blocks.
    ranges = [
        (0x00A0, 0x00FF),   # Latin-1 Supplement
        (0x0100, 0x017F),   # Latin Extended-A
        (0x0180, 0x024F),   # Latin Extended-B
        (0x0370, 0x03FF),   # Greek
        (0x0400, 0x04FF),   # Cyrillic
        (0x0500, 0x052F),   # Cyrillic Supplement
        (0x0590, 0x05FF),   # Hebrew
        (0x0600, 0x06FF),   # Arabic
        (0x0900, 0x097F),   # Devanagari
        (0x0980, 0x09FF),   # Bengali
        (0x0E00, 0x0E7F),   # Thai
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
        (0x4E00, 0x4FFF),   # CJK Unified Ideographs (first 8K)
        (0x5000, 0x5FFF),
        (0x6000, 0x6FFF),
        (0x7000, 0x7FFF),
        (0x8000, 0x8FFF),
        (0x9000, 0x9FFF),
        (0xAC00, 0xAD00),   # Hangul Syllables (slice)
        (0x2000, 0x206F),   # General Punctuation
        (0x2190, 0x21FF),   # Arrows
        (0x2200, 0x22FF),   # Math Operators
        (0x2500, 0x257F),   # Box Drawing
        (0x2600, 0x26FF),   # Misc Symbols
        (0x2700, 0x27BF),   # Dingbats
        (0x1F300, 0x1F5FF), # Misc Symbols & Pictographs
        (0x1F600, 0x1F64F), # Emoticons
        (0x1F680, 0x1F6FF), # Transport
    ]
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            yield ch
            yield " " + ch


def punct_combinations() -> Iterator[str]:
    puncts = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    for p in puncts:
        yield p
        yield " " + p
        yield p + " "
        yield p + p
        yield p + p + p
    # Common digraphs.
    for a in puncts:
        for b in puncts:
            if a != b:
                yield a + b


def code_snippet_probes() -> Iterator[str]:
    # Indentation + opening tokens that are common.
    for indent in ("    ", "        ", "            ", "                ", "\t", "\t\t"):
        for op in ("def ", "class ", "return ", "import ", "from ", "if ", "else:", "for ", "while ",
                   "function ", "const ", "let ", "var ", "public ", "private ", "static ",
                   "} else ", "} else if ", "} catch ", "} finally ", "// ", "# ", "/* ", "*/", "*/\n",
                   ");\n", "()\n", "{\n", "}\n", ";\n", ",\n"):
            yield indent + op
    for op in ("() {", "() =>", ": int", ": str", ": float", ": bool", "-> None", "-> str", "-> int",
               "== None", "!= None", "is None", "is not None", ">>> ", ">>>", "... ", "<<< ",
               "true,", "false,", "null,", "None,", "True,", "False,"):
        yield op


def english_morphemes() -> Iterator[str]:
    """Common English morphemes, affixes, and function-word fragments that
    BPE tokenizers commonly merge but aren't in any base tokenizer vocab
    under the exact form Claude might use."""
    morphemes = [
        "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "ful",
        "less", "able", "ible", "ous", "ive", "ize", "ise", "ify", "al", "ic",
        "ical", "ism", "ist", "ity", "ty", "ure", "age", "ship", "hood", "dom",
        "ward", "wise", "like", "some", "free", "proof", "worthy", "er's", "'s",
        "n't", "'re", "'ve", "'ll", "'d", "'m",
        "un", "re", "pre", "post", "non", "anti", "sub", "super", "inter",
        "trans", "multi", "semi", "mid", "over", "under", "out", "up", "down",
        "de", "dis", "mis", "co", "con", "com", "ex", "in", "im", "il", "ir",
    ]
    for m in morphemes:
        yield m
        yield " " + m
        yield m + " "
        yield m + "."
        yield m + ","
    # Word-starts that are common roots.
    roots = [
        "self", "make", "take", "give", "know", "think", "look", "come", "find",
        "tell", "ask", "work", "seem", "feel", "try", "leave", "call", "write",
        "read", "create", "build", "send", "run", "stop", "start", "stay",
        "gen", "sys", "file", "data", "info", "user", "app", "web", "net",
        "dev", "env", "conf", "init", "test", "log", "err", "tmp", "doc",
    ]
    for r in roots:
        yield r
        yield " " + r
        yield r.capitalize()
        yield " " + r.capitalize()


def html_url_probes() -> Iterator[str]:
    fragments = [
        "<div", "<span", "<p>", "</p>", "<a ", "</a>", "<br>", "<br/>",
        "<html", "<head", "<body", "<meta", "<link", "<script", "</script>",
        "</div>", "</span>", "<!--", "-->",
        "https://", "http://", "www.", ".com", ".org", ".io", ".net", ".gov",
        "://", "?q=", "&amp;", "&lt;", "&gt;", "&quot;", "&#39;",
    ]
    yield from fragments


def all_handcrafted() -> Iterator[str]:
    yield from whitespace_probes()
    yield from repeated_chars()
    yield from digit_probes()
    yield from identifier_probes()
    yield from english_morphemes()
    yield from punct_combinations()
    yield from code_snippet_probes()
    yield from html_url_probes()
    yield from unicode_probes()


def load_gupta_vocab(path: Path) -> tuple[list[str], list[str]]:
    import json
    with path.open() as f:
        d = json.load(f)
    return d["verified"], d.get("checked", [])


def kgrams_from_text(text: str, k_min: int = 2, k_max: int = 12,
                     min_freq: int = 2) -> list[str]:
    """Extract all substrings of length k_min..k_max from text, sorted by
    frequency (desc) then length (desc).

    This is the core candidate source: real-text substrings are the most
    likely BPE merges. Random or alphabetical n-grams yield near zero.
    """
    from collections import Counter
    cnt: Counter[str] = Counter()
    L = len(text)
    for k in range(k_min, k_max + 1):
        for i in range(L - k + 1):
            cnt[text[i:i + k]] += 1
    return [kg for kg, f in cnt.most_common() if f >= min_freq]


def multilingual_bigrams() -> Iterator[str]:
    """2/3-char combinations from major non-ASCII scripts.

    4.7 keeps many short multilingual merges (4.3% hit rate empirically
    on Russian 2-chars). Enumerate common script alphabets exhaustively.
    """
    scripts = {
        "russian": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "russian_upper": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
        "greek": "αβγδεζηθικλμνξοπρστυφχψω",
        "greek_upper": "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
        # Japanese hiragana + katakana
        "hiragana": "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
        "katakana": "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
        # Arabic isolated letters (partial)
        "arabic": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
        # Hebrew
        "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
        # Latin Extended (accented)
        "latin_ext": "áàâäãåéèêëíìîïóòôöõúùûüýÿçñæøœ",
        "latin_ext_upper": "ÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÝŸÇÑÆØŒ",
    }
    for name, alpha in scripts.items():
        # bigrams
        for a in alpha:
            for b in alpha:
                yield a + b
                yield " " + a + b
        # Also single char + space
        for a in alpha:
            yield a + " "
            yield " " + a + " "


def context_variants_of(vocab: set[str]) -> Iterator[str]:
    """Generate high-yield context variants: trailing-space and punctuation
    variants that are empirically single tokens in 4.7. Works for both
    ASCII words and non-ASCII (CJK, Cyrillic) tokens.
    """
    # Order matters for early yield: trailing space first.
    ascii_variants_after = [" ", ",", ".", ";", ":", "?", "!", "s", "es", "ed",
                             "ing", "ly", "n", "r", "-"]
    nonascii_variants_after = [" ", ",", ".", "。", "，", "、", "：", "；"]
    variants_before = ["", " "]
    for t in vocab:
        bare = t.strip(" \t")
        if not bare:
            continue
        is_ascii = bare.isascii()
        if is_ascii:
            if not all(c.isalpha() for c in bare):
                continue
            if len(bare) < 3 or len(bare) > 12:
                continue
            lower = bare.lower()
            cap = bare[0].upper() + bare[1:].lower()
            bases = (lower, cap)
            suffixes = ascii_variants_after
        else:
            # Non-ASCII: treat as-is; don't case-fold.
            if len(bare) < 1 or len(bare) > 8:
                continue
            bases = (bare,)
            suffixes = nonascii_variants_after
        for base in bases:
            for pre in variants_before:
                yield pre + base
                for post in suffixes:
                    yield pre + base + post


def smart_kgrams(text: str, vocab: set[str], k_min: int = 3, k_max: int = 12,
                 min_freq: int = 3, top_n: int | None = None) -> list[str]:
    """Targeted k-gram mining: yield ONLY substrings where our current
    vocab over-segments them (greedy_count >= 2). Frequency-sorted.

    This is the smart miner: it only probes strings that are CANDIDATES
    for missing merges, not random k-grams. Hit rate empirically 5-15%
    (higher than uncurated k-grams' 1-2%).
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from greedy import build_trie, greedy_count
    from collections import Counter

    trie = build_trie(sorted(vocab, key=len, reverse=True))
    cnt: Counter[str] = Counter()
    L = len(text)
    # Cache: for each position, the greedy match length starting there
    # to quickly decide if a substring crosses greedy boundaries.
    data = text.encode("utf-8")
    # Scan positions at word boundaries + all positions for exhaustive coverage
    for k in range(k_min, k_max + 1):
        for i in range(L - k + 1):
            s = text[i:i + k]
            # Only keep candidates that our current greedy over-segments
            if greedy_count(s, trie) >= 2:
                cnt[s] += 1
    ranked = [kg for kg, f in cnt.most_common() if f >= min_freq]
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def word_anchored_kgrams(text: str, k_min: int = 2, k_max: int = 14,
                         min_freq: int = 2) -> list[str]:
    """Extract k-grams that START at word boundaries (after whitespace/punct).

    BPE subwords in BPE-tokenized text overwhelmingly start at word
    boundaries. Anchoring at those positions gives much higher hit rate.
    Also yields space-prefixed variants (" word") which 4.7 heavily uses.
    """
    from collections import Counter
    cnt: Counter[str] = Counter()
    L = len(text)
    boundary = set(" \t\n\r.,;:!?\"'()[]{}<>/|\\-_=+*&^%$#@~`")
    at_boundary = [True] * (L + 1)
    for i, ch in enumerate(text):
        if ch in boundary:
            at_boundary[i + 1] = True
    for k in range(k_min, k_max + 1):
        for i in range(L - k + 1):
            if not at_boundary[i]:
                continue
            sub = text[i:i + k]
            cnt[sub] += 1
            # Also include the space-prefixed version explicitly; this
            # doubles the candidate space but captures ' word' tokens.
            if i > 0 and text[i - 1] == " ":
                cnt[" " + sub] += 1
    return [kg for kg, f in cnt.most_common() if f >= min_freq]


def ngram_probes() -> Iterator[str]:
    """Systematic n-gram probing: all 2-char, common 3-char, common 4-char
    combinations with and without leading space.

    4.7 keeps many short character n-grams as single tokens (e.g. 'au',
    'en', 'er', 'ing' is 2 tokens but ' tion' might be 1). This fills the
    gap that English subword merges leave.
    """
    import string
    lower = string.ascii_lowercase
    upper = string.ascii_uppercase

    # All 2-char lowercase + uppercase pairs
    for a in lower:
        for b in lower:
            yield a + b
            yield " " + a + b
    for a in upper:
        for b in lower:
            yield a + b
            yield " " + a + b

    # All 3-char lowercase (17576 of them). Heavy but high-yield.
    for a in lower:
        for b in lower:
            for c in lower:
                yield a + b + c
    # Space-prefixed 3-chars starting with vowel or common consonants.
    for a in "aeioubcdfghlmnprst":
        for b in lower:
            for c in lower:
                yield " " + a + b + c

    # Common 4-char suffixes/prefixes from real English.
    common_4gram = [
        "tion", "ment", "ness", "ing ", "ion ", "able", "ible", "ment",
        "ship", "ness", "less", "ful ", "hood", "dom ", "ward",
        "pre", "post", "over", "under", "sub", "super",
        "com", "con", "pro", "per", "trans", "inter", "anti", "semi",
        "the ", " the", "and ", " and", "for ", " for", "with", "from",
        "that", "this", "they", "them", "have", "been", "will", "would",
        "about", "after", "other", "their", "would", "could", "should",
    ]
    for g in common_4gram:
        yield g


def dict_words(word_file: str = "/usr/share/dict/words",
               min_len: int = 2, max_len: int = 16) -> Iterator[str]:
    """Iterate dictionary words with common BPE-style variants.

    Yields each word as: `word`, ` word`, ` Word`. 4.7 is observed to keep
    many title-cased proper nouns (" April", " American") as single tokens.
    """
    from pathlib import Path as _P
    p = _P(word_file)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            w = line.strip()
            if not w or len(w) < min_len or len(w) > max_len:
                continue
            if not w.isascii():
                continue
            lower = w.lower()
            cap = w[0].upper() + w[1:].lower() if w else w
            yield lower
            yield " " + lower
            yield " " + cap
            if w.isalpha() and len(w) <= 8:
                yield " " + w.upper()
