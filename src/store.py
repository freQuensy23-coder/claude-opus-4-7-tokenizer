"""Compact append-only store for probe results.

Format: CSV with columns `t,c,err` (string, int-or-empty, str-or-empty).
Rotates: when the active `.csv` exceeds `MAX_BYTES` (default 100 MB) it's
gzipped to `.csv.gz` and a new active file is started. Reading transparently
handles both plain `.csv` and gzipped `.csv.gz` (including legacy `.jsonl`
and `.jsonl.gz` for backwards compatibility with earlier runs).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Iterator

MAX_BYTES = 100 * 1024 * 1024


class Store:
    """Append-only writer with size-triggered rotation + gzip."""

    def __init__(self, prefix: Path, max_bytes: int = MAX_BYTES):
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._f = None
        self._writer = None
        self._size = 0
        self._seg = 0
        self._advance_to_writable()

    def _part_plain(self, i: int) -> Path:
        return self.prefix.with_name(f"{self.prefix.name}.{i:04d}.csv")

    def _part_gz(self, i: int) -> Path:
        return self.prefix.with_name(f"{self.prefix.name}.{i:04d}.csv.gz")

    def _advance_to_writable(self):
        # Find lowest `i` where neither plain nor gz exists; that's the new segment.
        i = 0
        while self._part_gz(i).exists():
            i += 1
        # If a plain file exists at this `i`, resume it if small, else rotate.
        p = self._part_plain(i)
        if p.exists() and p.stat().st_size >= self.max_bytes:
            self._compress(p)
            i += 1
            p = self._part_plain(i)
        self._seg = i
        self._size = p.stat().st_size if p.exists() else 0
        write_header = not p.exists() or p.stat().st_size == 0
        self._f = p.open("a", newline="", buffering=1)
        self._writer = csv.writer(self._f)
        if write_header:
            self._writer.writerow(["t", "c", "err"])
            self._size = self._f.tell()

    def _compress(self, plain: Path):
        gz = plain.with_suffix(plain.suffix + ".gz")
        with plain.open("rb") as src, gzip.open(gz, "wb", compresslevel=6) as dst:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                dst.write(chunk)
        plain.unlink()

    def _maybe_rotate(self):
        if self._size < self.max_bytes:
            return
        self._f.close()
        self._compress(self._part_plain(self._seg))
        self._seg += 1
        p = self._part_plain(self._seg)
        self._f = p.open("a", newline="", buffering=1)
        self._writer = csv.writer(self._f)
        self._writer.writerow(["t", "c", "err"])
        self._size = self._f.tell()

    def write(self, text: str, count: int | None = None, err: str | None = None):
        buf = io.StringIO()
        csv.writer(buf).writerow([text, "" if count is None else count,
                                  "" if err is None else err])
        line = buf.getvalue()
        self._f.write(line)
        self._size += len(line.encode("utf-8"))
        self._maybe_rotate()

    def flush(self):
        if self._f:
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _iter_file(path: Path) -> Iterator[dict]:
    """Yield dicts {'t': str, 'c': int|None, 'err': str|None} from one file.
    Supports: .csv, .csv.gz, .jsonl, .jsonl.gz
    """
    name = path.name
    if name.endswith(".csv.gz"):
        opener = lambda: gzip.open(path, "rt", encoding="utf-8", newline="")
        is_csv = True
    elif name.endswith(".csv"):
        opener = lambda: path.open("r", encoding="utf-8", newline="")
        is_csv = True
    elif name.endswith(".jsonl.gz"):
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
        is_csv = False
    elif name.endswith(".jsonl"):
        opener = lambda: path.open("r", encoding="utf-8")
        is_csv = False
    else:
        return

    with opener() as f:
        if is_csv:
            rdr = csv.reader(f)
            try:
                header = next(rdr)
            except StopIteration:
                return
            for row in rdr:
                if len(row) < 1:
                    continue
                t = row[0]
                c = row[1] if len(row) > 1 else ""
                e = row[2] if len(row) > 2 else ""
                yield {
                    "t": t,
                    "c": int(c) if c != "" else None,
                    "err": e if e != "" else None,
                }
        else:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                yield {"t": j["t"], "c": j.get("c"), "err": j.get("err")}


def iter_records(prefix: Path) -> Iterator[dict]:
    """Iterate all records under prefix (across parts and legacy files)."""
    prefix = Path(prefix)
    parent = prefix.parent
    name = prefix.name
    # Match both new CSV-parts and legacy JSONL.
    patterns = [
        f"{name}.*.csv.gz",
        f"{name}.*.csv",
        f"{name}.jsonl.gz",
        f"{name}.jsonl",
    ]
    seen_files: set[str] = set()
    for pat in patterns:
        for p in sorted(parent.glob(pat)):
            if p.name in seen_files:
                continue
            seen_files.add(p.name)
            yield from _iter_file(p)


def load_checked(prefix: Path) -> set[str]:
    """All `t` keys ever written (hits, misses, errors)."""
    return {r["t"] for r in iter_records(prefix)}


def load_hits(prefix: Path) -> set[str]:
    """Only records where count == 1."""
    return {r["t"] for r in iter_records(prefix) if r.get("c") == 1}


def migrate_jsonl_to_csv(prefix: Path, gzip_old: bool = True):
    """Convert a prefix's legacy JSONL file to CSV part 0000."""
    prefix = Path(prefix)
    jsonl = prefix.with_name(f"{prefix.name}.jsonl")
    if not jsonl.exists() or jsonl.stat().st_size == 0:
        return
    csv_out = prefix.with_name(f"{prefix.name}.0000.csv")
    if csv_out.exists() or prefix.with_name(f"{prefix.name}.0000.csv.gz").exists():
        return
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "c", "err"])
        for rec in _iter_file(jsonl):
            w.writerow([rec["t"], "" if rec["c"] is None else rec["c"],
                        "" if rec["err"] is None else rec["err"]])
    if gzip_old:
        with jsonl.open("rb") as src, gzip.open(str(jsonl) + ".gz", "wb",
                                                 compresslevel=6) as dst:
            while True:
                chunk = src.read(1 << 16)
                if not chunk:
                    break
                dst.write(chunk)
        jsonl.unlink()


def total_size(prefix: Path) -> int:
    prefix = Path(prefix)
    parent = prefix.parent
    name = prefix.name
    total = 0
    for pat in [f"{name}.*.csv.gz", f"{name}.*.csv",
                f"{name}.jsonl.gz", f"{name}.jsonl"]:
        for p in parent.glob(pat):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total
