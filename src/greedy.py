"""Greedy longest-match tokenizer + trie, matching Gupta's ctoc approach."""
from __future__ import annotations

import json
from pathlib import Path


class Trie:
    __slots__ = ("root",)

    def __init__(self):
        self.root: dict = {}

    def add(self, s: str):
        node = self.root
        for b in s.encode("utf-8"):
            node = node.setdefault(b, {})
        node[-1] = True  # terminal

    def longest_match(self, data: bytes, start: int) -> int:
        """Return end index (exclusive) of longest token starting at `start`.
        Returns start+1 if no multi-byte match (byte fallback)."""
        node = self.root
        best = start + 1
        i = start
        while i < len(data):
            nxt = node.get(data[i])
            if nxt is None:
                break
            node = nxt
            i += 1
            if -1 in node:
                best = i
        return best


def build_trie(vocab: list[str]) -> Trie:
    t = Trie()
    for s in vocab:
        if s:
            t.add(s)
    return t


def greedy_tokenize(text: str, trie: Trie) -> list[str]:
    data = text.encode("utf-8")
    out: list[str] = []
    i = 0
    while i < len(data):
        j = trie.longest_match(data, i)
        out.append(data[i:j].decode("utf-8", errors="replace"))
        i = j
    return out


def greedy_count(text: str, trie: Trie) -> int:
    data = text.encode("utf-8")
    i = 0
    n = 0
    while i < len(data):
        i = trie.longest_match(data, i)
        n += 1
    return n


def load_vocab(path: Path) -> list[str]:
    with path.open() as f:
        d = json.load(f)
    return d["verified"]
