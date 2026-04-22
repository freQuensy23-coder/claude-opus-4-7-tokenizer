#!/usr/bin/env python3
"""Generate vocab_data.cc and vocab_data.h from vocab.json.

Reads the "verified" array from the vocab JSON and emits a C++ source file
containing the token strings as a compile-time array, so the binary is
self-contained and doesn't need the JSON at runtime.
"""

import json
import sys


def c_escape(s: str) -> str:
    """Escape a Python string for embedding as a C string literal.

    Uses octal escapes for non-ASCII bytes to avoid the C++ problem where
    \\xNN followed by a hex digit gets parsed as a longer hex literal.
    """
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '\\':
            out.append('\\\\')
        elif ch == '"':
            out.append('\\"')
        elif ch == '\n':
            out.append('\\n')
        elif ch == '\r':
            out.append('\\r')
        elif ch == '\t':
            out.append('\\t')
        elif ch == '\0':
            out.append('\\0')
        elif ch == '\a':
            out.append('\\a')
        elif ch == '\b':
            out.append('\\b')
        elif ch == '\f':
            out.append('\\f')
        elif ch == '\v':
            out.append('\\v')
        elif 0x20 <= o <= 0x7E:
            out.append(ch)
        else:
            # Emit raw UTF-8 bytes as octal escapes (max 3 digits, so no
            # ambiguity with following characters unlike hex escapes).
            for b in ch.encode('utf-8'):
                out.append(f'\\{b:03o}')
    return ''.join(out)


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <vocab.json> <out.cc> <out.h>", file=sys.stderr)
        sys.exit(1)

    vocab_path, cc_path, h_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokens = data['verified']

    # Write header
    with open(h_path, 'w', encoding='utf-8') as f:
        f.write('#ifndef VOCAB_DATA_H_\n')
        f.write('#define VOCAB_DATA_H_\n\n')
        f.write('#include <cstddef>\n\n')
        f.write(f'extern const char* const VOCAB_TOKENS[{len(tokens)}];\n')
        f.write(f'extern const size_t VOCAB_COUNT;\n\n')
        f.write('#endif  // VOCAB_DATA_H_\n')

    # Write source
    with open(cc_path, 'w', encoding='utf-8') as f:
        f.write('#include "vocab_data.h"\n\n')
        f.write(f'const char* const VOCAB_TOKENS[{len(tokens)}] = {{\n')
        for i, token in enumerate(tokens):
            escaped = c_escape(token)
            f.write(f'    "{escaped}",\n')
        f.write('};\n\n')
        f.write(f'const size_t VOCAB_COUNT = {len(tokens)};\n')

    print(f"Generated {cc_path} and {h_path} with {len(tokens)} tokens", file=sys.stderr)


if __name__ == '__main__':
    main()
