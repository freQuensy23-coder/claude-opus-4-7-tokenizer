#!/usr/bin/env bash
# Launch the 4.7 tokenizer reconstruction pipeline.
# Usage: ./run.sh [phase]       (phase defaults to 'all')
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set" >&2
    exit 1
fi

PHASE="${1:-all}"
RPM="${RPM:-3000}"
MAX_CONC="${MAX_CONC:-300}"
MODEL="${MODEL:-claude-opus-4-7}"

mkdir -p state logs

LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S)_${PHASE}.log"
echo "Logging to $LOG"
echo "Phase: $PHASE  RPM: $RPM  MaxConcurrency: $MAX_CONC  Model: $MODEL"

uv run python -u src/pipeline.py \
    --phase "$PHASE" \
    --rpm "$RPM" \
    --max-concurrency "$MAX_CONC" \
    --model "$MODEL" 2>&1 | tee "$LOG"
