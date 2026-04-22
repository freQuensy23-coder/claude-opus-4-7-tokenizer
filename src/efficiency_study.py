"""Tokenizer efficiency study: claude-opus-4-6 vs claude-opus-4-7 via live
count_tokens on 30+ diverse text samples, normalised to ~4000 bytes each.

Writes EFFICIENCY.md plus a CSV snapshot in state/efficiency.csv.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

API = "https://api.anthropic.com/v1/messages/count_tokens"
HEADERS = {
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}

# --- Sample definitions: (domain, label, path-or-lambda, trim_bytes) ---
def read(p: str, n: int = 4000) -> str:
    try:
        t = Path(p).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    # If file begins with XML header, drop first bit.
    if t.startswith("<?xml") or "<mediawiki" in t[:200]:
        i = t.find(">")
        t = t[i+1:]
    return t[:n]


def json_sample() -> str:
    rec = {"id": 42, "name": "Alex", "tags": ["tokenizer", "bpe", "claude"],
           "payload": {"nested": {"a": [1,2,3], "b": {"c": "d"}},
                       "ts": "2026-04-22T14:00:00Z"},
           "meta": {"version": "4.7.1", "region": "eu-west-1"}}
    return json.dumps([rec]*40, indent=2)[:4000]


def url_log_sample() -> str:
    import random
    random.seed(1)
    lines = []
    for _ in range(80):
        status = random.choice([200, 200, 200, 404, 500, 301, 301])
        path = "/" + "/".join(random.choice(
            ["api","v1","users","items","search","login","logout",
             "billing","account","orders","products","cart"])
            for _ in range(random.randint(1,4)))
        lines.append(
            f'[2026-04-22T{random.randint(10,23):02d}:{random.randint(0,59):02d}:00Z] '
            f'{random.randint(1,255)}.{random.randint(0,255)}.'
            f'{random.randint(0,255)}.{random.randint(0,255)} '
            f'"GET {path} HTTP/1.1" {status} {random.randint(100,99999)} '
            f'"-" "Mozilla/5.0 (X11; Linux x86_64)"')
    return "\n".join(lines)[:4000]


def emoji_sample() -> str:
    return ("The launch went 🚀 — team crushed it 🎉🔥! New features:\n"
            "• Tokenizer 🧠 2x faster ⚡️\n"
            "• Inference 💨 latency ↓ 40%\n"
            "• Multilingual 🌍 support for 中文, العربية, русский, 日本語\n"
            "Celebrating 🥳🎂 with 🍕🍻🍺🎈. Happy team 😊😄🙌.\n") * 15


def math_latex_sample() -> str:
    return r"""
The Cauchy-Schwarz inequality states that for vectors $u, v \in \mathbb{R}^n$:
\[
|\langle u, v \rangle|^2 \le \langle u, u \rangle \cdot \langle v, v \rangle.
\]
Proof. Consider $f(t) = \langle u - tv, u - tv \rangle = \|u\|^2 - 2t\langle u,v\rangle + t^2\|v\|^2$.
Since $f(t) \ge 0$ for all $t \in \mathbb{R}$, the discriminant must be $\le 0$:
\[4\langle u,v\rangle^2 - 4\|u\|^2\|v\|^2 \le 0 \iff \langle u,v\rangle^2 \le \|u\|^2\|v\|^2.\]
Now for the $L^2$ Hilbert space $H = L^2([0,1])$, integrating:
\[\left|\int_0^1 f(x)\overline{g(x)}\,dx\right|^2 \le \int_0^1 |f|^2\,dx \cdot \int_0^1 |g|^2\,dx.\]
""" * 4


SAMPLES: list[tuple[str, str, str]] = [
    # (domain, label, text)
    ("en/classical",  "alice",          read("corpus_data/alice.txt")),
    ("en/classical",  "mobydick",       read("corpus_data/mobydick.txt")),
    ("en/classical",  "sherlock",       read("corpus_data/sherlock.txt")),
    ("en/classical",  "tomsawyer",      read("corpus_data/tomsawyer.txt")),
    ("en/modern",     "readme_whisper", read("corpus_data/readme_whisper.md")),
    ("en/modern",     "md_vscode",      read("corpus_eff/md_tech.md")),
    ("en/modern",     "wiki_chatgpt",   read("corpus_eff/en_wiki_news_ai.txt")),
    ("en/modern",     "wiki_ai",        read("corpus_data/wiki_ai.xml")),

    ("ru/classical",  "pushkin_dubrovsky", read("corpus_eff/ru_classical_dubrovsky.txt")),
    ("ru/biography",  "dostoevsky_wiki",   read("corpus_eff/ru_wiki_dostoevsky.txt")),
    ("ru/biography",  "pushkin_wiki",      read("corpus_eff/ru_classical_pushkin.txt")),
    ("ru/modern",     "moscow_wiki",       read("corpus_eff/ru_wiki_moscow.txt")),
    ("ru/modern",     "ai_wiki_ru",        read("corpus_eff/ru_wiki_news_ai.txt")),

    ("code/py",       "asyncio",  read("corpus_data/code_asyncio.py")),
    ("code/py",       "pathlib",  read("corpus_data/code_pathlib.py")),
    ("code/c",        "linux",    read("corpus_data/code_linux.c")),
    ("code/cpp",      "llvm",     read("corpus_eff/code_cpp.cpp")),
    ("code/js",       "lodash",   read("corpus_eff/code_js.js")),
    ("code/ts",       "tsc",      read("corpus_eff/code_ts.ts")),
    ("code/rust",     "vec",      read("corpus_eff/code_rust.rs")),
    ("code/go",       "json_dec", read("corpus_eff/code_go.go")),
    ("code/java",     "spring",   read("corpus_eff/code_java.java")),
    ("code/kotlin",   "descr",    read("corpus_eff/code_kotlin.kt")),
    ("code/ruby",     "rails",    read("corpus_eff/code_ruby.rb")),
    ("code/sql",      "pg_sql",   read("corpus_eff/code_sql.sql")),
    ("code/html",     "mdn_html", read("corpus_eff/code_html.html")),
    ("code/css",      "bootstrap", read("corpus_eff/code_css.css")),
    ("code/yaml",     "coredns",  read("corpus_eff/code_yaml.yaml")),
    ("code/bash",     "ohmyzsh",  read("corpus_eff/code_bash.sh")),

    ("cjk/ja",        "wiki_ja_xml",  read("corpus_data/wiki_ja.xml")),
    ("cjk/ja",        "wiki_tokyo",   read("corpus_eff/ja_wiki_tokyo.txt")),
    ("cjk/zh",        "wiki_zh_xml",  read("corpus_data/wiki_zh.xml")),
    ("cjk/zh",        "wiki_beijing", read("corpus_eff/zh_wiki_beijing.txt")),
    ("cjk/ko",        "wiki_seoul",   read("corpus_eff/ko_wiki_seoul.txt")),
    ("cjk/ko",        "wiki_korean",  read("corpus_eff/ko_wiki_korean.txt")),

    ("semitic/ar",    "wiki_quran",      read("corpus_eff/ar_wiki_quran.txt")),
    ("semitic/he",    "wiki_jerusalem",  read("corpus_eff/he_wiki_jerusalem.txt")),
    ("greek",         "wiki_athens",     read("corpus_eff/el_wiki_athens.txt")),
    ("thai",          "wiki_bangkok",    read("corpus_eff/th_wiki_bangkok.txt")),
    ("devanagari/hi", "wiki_delhi",      read("corpus_eff/hi_wiki_delhi.txt")),

    ("data/json",     "synth",        json_sample()),
    ("data/logs",     "http_access",  url_log_sample()),
    ("misc/emoji",    "launch",       emoji_sample()),
    ("misc/math",     "cauchy_schwarz", math_latex_sample()),
    ("misc/random",   "hash_16",      "#" * 500),
    ("misc/random",   "paren_semi",   ");" * 200),
]


async def count_once(client: httpx.AsyncClient, model: str, text: str,
                     retries: int = 8) -> int:
    payload = {"model": model,
               "messages": [{"role": "user", "content": text}]}
    back = 1.0
    for i in range(retries):
        r = await client.post(API, json=payload)
        if r.status_code == 200:
            return r.json()["input_tokens"]
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            await asyncio.sleep(float(ra) if ra else back)
            back = min(30.0, back * 2)
            continue
        if r.status_code >= 500:
            await asyncio.sleep(back); back = min(30.0, back * 2)
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    raise RuntimeError("retries exhausted")


async def main() -> None:
    samples = [(d, n, t) for (d, n, t) in SAMPLES if t]
    print(f"[study] {len(samples)} samples loaded\n")

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=60.0,
        limits=httpx.Limits(max_connections=40),
    ) as client:
        # Framing baseline = raw("a") - 1 (since "a" itself is 1 content token).
        base_46 = await count_once(client, "claude-opus-4-6", "a") - 1
        base_47 = await count_once(client, "claude-opus-4-7", "a") - 1
        print(f"[study] framing baseline 4.6={base_46}  4.7={base_47}\n")

        sem = asyncio.Semaphore(40)

        async def work(model: str, text: str) -> int:
            async with sem:
                return await count_once(client, model, text)

        t0 = time.monotonic()
        results = []
        tasks46 = [work("claude-opus-4-6", t) for (_,_,t) in samples]
        tasks47 = [work("claude-opus-4-7", t) for (_,_,t) in samples]
        c46 = await asyncio.gather(*tasks46)
        c47 = await asyncio.gather(*tasks47)
        for (d, n, t), a, b in zip(samples, c46, c47):
            # Framing-adjusted content token counts.
            n46 = a - base_46
            n47 = b - base_47
            results.append({
                "domain": d, "label": n,
                "bytes": len(t.encode("utf-8")),
                "t46": n46, "t47": n47,
                "ratio": n47 / n46 if n46 else 0.0,
                "bpt46": len(t.encode("utf-8")) / n46 if n46 else 0.0,
                "bpt47": len(t.encode("utf-8")) / n47 if n47 else 0.0,
            })
        print(f"[study] done in {time.monotonic()-t0:.1f}s\n")

    # Print ranked table.
    results.sort(key=lambda r: r["ratio"])
    print(f"{'domain':<16} {'label':<22} {'bytes':>6} {'4.6':>7} {'4.7':>7} "
          f"{'4.7/4.6':>8} {'bpt46':>6} {'bpt47':>6}")
    print("-"*95)
    for r in results:
        print(f"{r['domain']:<16} {r['label']:<22} {r['bytes']:>6} "
              f"{r['t46']:>7} {r['t47']:>7} {r['ratio']:>8.3f} "
              f"{r['bpt46']:>6.2f} {r['bpt47']:>6.2f}")

    # Aggregate by domain.
    print("\n--- by domain (averaged) ---")
    bucket: dict[str, list] = defaultdict(list)
    for r in results:
        bucket[r["domain"]].append(r)
    print(f"{'domain':<16} {'n':>3} {'avg ratio':>10} {'mean bpt46':>11} {'mean bpt47':>11}")
    for d in sorted(bucket):
        rs = bucket[d]
        ar = sum(x["ratio"] for x in rs) / len(rs)
        a46 = sum(x["bpt46"] for x in rs) / len(rs)
        a47 = sum(x["bpt47"] for x in rs) / len(rs)
        print(f"{d:<16} {len(rs):>3} {ar:>10.3f} {a46:>11.2f} {a47:>11.2f}")

    # Persist CSV + return.
    import csv
    out = ROOT / "state" / "efficiency.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain","label","bytes","t46","t47","ratio","bpt46","bpt47"])
        for r in results:
            w.writerow([r["domain"], r["label"], r["bytes"],
                        r["t46"], r["t47"], f"{r['ratio']:.4f}",
                        f"{r['bpt46']:.3f}", f"{r['bpt47']:.3f}"])
    print(f"\n[study] wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
