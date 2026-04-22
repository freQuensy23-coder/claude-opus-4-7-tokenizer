#!/usr/bin/env bash
# Quick snapshot of pipeline progress.
cd "$(dirname "$0")/.."
python3 - <<'PY'
import sys; sys.path.insert(0, "src")
from store import iter_records
from pathlib import Path
state = Path("state")
total_hits = set()
print(f"{'phase':<30} {'probed':>8} {'hits':>6} {'rate%':>6}")
print("-" * 52)
prefixes = sorted({p.name.split('.', 1)[0] for p in state.glob("phase_*")})
for stem in prefixes:
    if 'archived' in stem: continue
    hits = 0; total = 0
    for r in iter_records(state / stem):
        total += 1
        if r.get("c") == 1:
            hits += 1
            total_hits.add(r["t"])
    rate = 100 * hits / max(1, total)
    print(f"{stem:<30} {total:>8} {hits:>6} {rate:>5.2f}%")
print("-" * 52)
print(f"{'UNIQUE hits (all phases):':<30} {len(total_hits):>8}")
PY
echo ""
echo "--- running processes ---"
pgrep -af "pipeline.py|run_all_after|run.sh" 2>/dev/null | grep -v pgrep || echo "no pipeline running"
echo ""
echo "--- recent log lines ---"
ls -t /private/tmp/claude-501/-Users-a-mametyev-PycharmProjects-claude-tokenizer/*/tasks/*.output 2>/dev/null | head -1 | xargs -I{} tail -6 {}
