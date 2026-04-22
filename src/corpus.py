"""Diverse multi-domain corpus for bigram/merge discovery.

~100KB of mixed text: English prose (fiction + non-fiction), code (Python,
JS, TS, Go, Rust, C), JSON/YAML/HTML/markdown, short multilingual samples.

Used by the corpus-mining phase to discover 4.7-specific merges that don't
appear in any public BPE tokenizer.
"""
from __future__ import annotations

from pathlib import Path


ENGLISH_PROSE = """
In the beginning God created the heaven and the earth. And the earth was without form, and void; and darkness was upon the face of the deep. And the Spirit of God moved upon the face of the waters. And God said, Let there be light: and there was light. And God saw the light, that it was good: and God divided the light from the darkness.

It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of light, it was the season of darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way.

The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! The five boxing wizards jump quickly. Sphinx of black quartz, judge my vow. The jay, pig, fox, zebra and my wolves quack!

Call me Ishmael. Some years ago, never mind how long precisely, having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation.

The sun was shining on the sea, shining with all his might; he did his very best to make the billows smooth and bright. And this was odd, because it was the middle of the night. A bird in the hand is worth two in the bush. Actions speak louder than words. Better late than never. Birds of a feather flock together.

Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed. Deep learning uses neural networks with multiple layers to progressively extract higher level features from the raw input. The technology has numerous applications in computer vision, natural language processing, speech recognition, and autonomous systems.

Consider the following scenario: a distributed system must handle failures gracefully, maintain consistency under concurrent updates, and scale to handle variable load. Engineers typically address these requirements through techniques such as replication, consensus protocols like Raft or Paxos, sharding, and careful design of APIs that minimize coupling between services.

Climate change represents one of the most significant challenges facing humanity. Rising global temperatures have led to more frequent extreme weather events, melting polar ice caps, rising sea levels, and disruptions to ecosystems worldwide. Addressing this crisis requires coordinated international action, transition to renewable energy, and fundamental changes in how we produce and consume goods.

According to the latest research published in Nature, scientists have discovered a new method for detecting dark matter. The technique uses quantum sensors to measure subtle gravitational effects that cannot be explained by visible matter alone. This breakthrough could revolutionize our understanding of the universe.
"""

CODE_SAMPLES = '''
def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number using iteration."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict = {}
        self.order: list = []

    def get(self, key):
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)


async def fetch_user(user_id: int) -> dict | None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"/api/users/{user_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()


// JavaScript / TypeScript
interface User {
    id: number;
    name: string;
    email: string;
    roles: string[];
    createdAt: Date;
}

export function formatUser(user: User): string {
    return `${user.name} <${user.email}>`;
}

const users = await fetchUsers({ limit: 100, offset: 0 });
users.filter(u => u.roles.includes('admin')).forEach(u => console.log(u));

// Go
package main

import (
    "fmt"
    "net/http"
)

func handleIndex(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, %s!", r.URL.Path[1:])
}

func main() {
    http.HandleFunc("/", handleIndex)
    http.ListenAndServe(":8080", nil)
}

// Rust
use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, i32> = HashMap::new();
    scores.insert(String::from("Alice"), 95);
    scores.insert(String::from("Bob"), 87);

    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
}

// C
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int compare(const void *a, const void *b) {
    return *(int *)a - *(int *)b;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nums>\\n", argv[0]);
        return 1;
    }
    return 0;
}

# SQL
SELECT u.id, u.name, COUNT(o.id) as order_count, SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 100;
'''

JSON_YAML_HTML = '''
{
    "name": "claude-tokenizer",
    "version": "1.0.0",
    "description": "A fast offline tokenizer for Claude Opus",
    "main": "dist/index.js",
    "scripts": {
        "build": "tsc",
        "test": "jest",
        "start": "node dist/index.js"
    },
    "dependencies": {
        "axios": "^1.6.0",
        "express": "^4.18.2",
        "lodash": "^4.17.21"
    },
    "devDependencies": {
        "@types/node": "^20.10.0",
        "typescript": "^5.3.0",
        "jest": "^29.7.0"
    },
    "repository": {
        "type": "git",
        "url": "https://github.com/example/project.git"
    }
}

# config.yaml
server:
  host: 0.0.0.0
  port: 8080
  timeout: 30s

database:
  driver: postgres
  host: db.example.com
  port: 5432
  name: myapp
  user: appuser
  pool:
    max_connections: 20
    idle_timeout: 300s

logging:
  level: info
  format: json
  outputs:
    - type: stdout
    - type: file
      path: /var/log/app.log
      max_size: 100MB
      max_backups: 10

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="/styles.css">
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </nav>
    </header>
    <main>
        <h1>Welcome</h1>
        <p>This is a <strong>sample</strong> paragraph with <em>formatting</em>.</p>
        <ul>
            <li>First item</li>
            <li>Second item</li>
            <li>Third item</li>
        </ul>
    </main>
    <footer>&copy; 2026 Example Corp.</footer>
</body>
</html>
'''

MARKDOWN = '''
# Getting Started with Python

Python is a **high-level**, _interpreted_ programming language known for its readability and versatility. This guide will walk you through the basics.

## Installation

You can install Python from [python.org](https://www.python.org/). On macOS, you can also use Homebrew:

```bash
brew install python@3.12
python3 --version
```

On Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-pip
```

## Hello World

Create a file `hello.py`:

```python
print("Hello, world!")
```

Run it with `python3 hello.py`.

## Variables and Types

Python supports several built-in types:

- `int`: integer numbers (e.g., `42`, `-7`)
- `float`: floating-point numbers (e.g., `3.14`, `1e-10`)
- `str`: strings (e.g., `"hello"`, `'world'`)
- `bool`: booleans (`True` or `False`)
- `list`: ordered mutable collections
- `dict`: key-value mappings
- `set`: unordered unique collections
- `tuple`: immutable ordered collections

## Control Flow

Use `if`, `elif`, and `else`:

```python
x = 10
if x > 5:
    print("large")
elif x > 0:
    print("small")
else:
    print("non-positive")
```

For loops iterate over iterables:

```python
for i in range(5):
    print(i)

for name in ["alice", "bob", "charlie"]:
    print(f"Hello, {name}!")
```

## Next Steps

- [ ] Read the [official tutorial](https://docs.python.org/3/tutorial/)
- [ ] Practice on [LeetCode](https://leetcode.com/)
- [ ] Build a small project
'''

MULTILINGUAL = '''
Français: Je pense, donc je suis. La liberté des uns s'arrête là où commence celle des autres. Ce qui se conçoit bien s'énonce clairement. Il n'y a que les imbéciles qui ne changent pas d'avis.

Español: En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor.

Deutsch: Alle Menschen sind frei und gleich an Würde und Rechten geboren. Sie sind mit Vernunft und Gewissen begabt und sollen einander im Geist der Brüderlichkeit begegnen.

Italiano: Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, ché la diritta via era smarrita. Ahi quanto a dir qual era è cosa dura esta selva selvaggia e aspra e forte che nel pensier rinova la paura.

Português: A dificuldade é o que estimula o engenho. A maior glória em viver não reside em nunca cair, mas em nos levantarmos cada vez que caímos.

Русский: Все счастливые семьи похожи друг на друга, каждая несчастливая семья несчастлива по-своему. Широка страна моя родная. Война и мир. Доброе утро, как дела?

中文: 學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？天行健，君子以自強不息。三人行必有我師焉。

日本語: 吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。雪国のトンネルを抜けると夜の底が白くなった。

한국어: 모든 국민은 법 앞에 평등하다. 나는 사람이 그립다. 인생은 짧고 예술은 길다. 가난하다고 해서 외로움을 모르겠는가.

العربية: إن الحياة لتمنحنا من الدروس أكثر مما نتصور، والوقت خير معلم. العلم في الصغر كالنقش على الحجر.

हिन्दी: सत्यमेव जयते। अहिंसा परमो धर्मः। विद्या ददाति विनयम्। अतिथि देवो भव। कर्म एव धर्म।

ภาษาไทย: ความรักชาติ ศาสนา พระมหากษัตริย์ เป็นหลักชัยของชนชาวไทย. สวัสดีครับ ขอบคุณมาก.
'''


def default_corpus() -> str:
    return "\n\n".join([ENGLISH_PROSE, CODE_SAMPLES, JSON_YAML_HTML, MARKDOWN, MULTILINGUAL])


def extended_corpus() -> str:
    """Multi-repeat + variations so we expose more bigram contexts."""
    parts = [default_corpus()]
    root = Path(__file__).resolve().parent.parent
    for extra in [
        root / "reference" / "REPORT.md",
        root / "reference" / "ctoc.cc",
        root / "reference" / "gen_vocab.py",
        root / "src" / "pipeline.py",
        root / "src" / "candidates.py",
        root / "src" / "counter.py",
        root / "src" / "corpus.py",
    ]:
        if extra.exists():
            parts.append(extra.read_text())
    # System word list if available (provides real word diversity).
    words = Path("/usr/share/dict/words")
    if words.exists():
        # Sample ~20K words mid-file to avoid alphabetical bias.
        lines = words.read_text().splitlines()
        import random
        rng = random.Random(42)
        sampled = rng.sample(lines, min(20_000, len(lines)))
        parts.append(" ".join(sampled))
    return "\n\n".join(parts)


def huge_corpus() -> str:
    """All external corpus files from corpus_data/ + extended_corpus.

    Used for targeted merge discovery. Several megabytes of diverse text
    (Gutenberg novels, Wikipedia XML, CPython source, multilingual).
    """
    parts = [extended_corpus()]
    root = Path(__file__).resolve().parent.parent
    cd = root / "corpus_data"
    if cd.exists():
        for p in sorted(cd.glob("*")):
            try:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return "\n\n".join(parts)
