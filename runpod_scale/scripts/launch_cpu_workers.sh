#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
OUTPUT_ROOT="${GIADA_OUTPUT_ROOT:-/workspace/giada-data/s1}"
PLAN="${GIADA_PLAN:-$OUTPUT_ROOT/plan.json}"
WORKER_COUNT="${GIADA_WORKER_COUNT:-1}"

mkdir -p "$OUTPUT_ROOT/logs"
cd "$GIADA_ROOT"
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  python -m src.giada_runpod.cli worker \
    --plan "$PLAN" \
    --output "$OUTPUT_ROOT" \
    --elm-repo "$GIADA_ROOT" \
    --teacher-repo "$TEACHER_ROOT" \
    --worker-index "$worker" \
    --worker-count "$WORKER_COUNT" \
    --worker-seed "$((7000001 + worker))" \
    >"$OUTPUT_ROOT/logs/worker-$worker.log" 2>&1 &
done
wait
