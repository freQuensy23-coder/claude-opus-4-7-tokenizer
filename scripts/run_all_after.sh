#!/usr/bin/env bash
# Wait for a PID (or all python pipelines) to finish, then run remaining phases sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."

# Wait for prior pipeline invocations to exit.
while pgrep -f "src/pipeline.py" >/dev/null 2>&1; do
    sleep 5
done
echo "prior pipeline finished, launching remaining phases"

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "ERROR: no ANTHROPIC_API_KEY" >&2; exit 1
fi

for ph in tiktoken handcrafted hf boundary; do
    echo ""
    echo "============ launching $ph ============"
    ./run.sh "$ph" || { echo "phase $ph failed"; exit 1; }
done

echo ""
echo "============ assemble ============"
uv run python -u -c "
import sys; sys.path.insert(0, 'src')
from pipeline import phase_assemble
phase_assemble()
"

echo ""
echo "============ evaluate ============"
./run.sh evaluate
