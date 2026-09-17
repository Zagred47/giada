#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
WORKER_COUNT="${GIADA_WORKER_COUNT:-8}"
OUTPUT_ROOT="${GIADA_NEURONIO_CONTEXT_ROOT:-/workspace/giada-data/surrogate-validity-neuronio-context-v1}"
CONFIG="$GIADA_ROOT/runpod_scale/configs/surrogate_validity_eval_neuronio_contextual.yml"

cd "$GIADA_ROOT"
mkdir -p "$OUTPUT_ROOT/status" "$OUTPUT_ROOT/logs"
if [[ ! -f "$OUTPUT_ROOT/plan.json" ]]; then
  "$PYTHON_BIN" -m src.giada_runpod.cli plan --config "$CONFIG" --output "$OUTPUT_ROOT"
fi
expected="$($PYTHON_BIN -c "import json; print(len(json.load(open('$OUTPUT_ROOT/plan.json'))['shards']))")"
echo "[GIADA validity contextual] starting $expected shards on $WORKER_COUNT workers"
GIADA_OUTPUT_ROOT="$OUTPUT_ROOT" GIADA_PLAN="$OUTPUT_ROOT/plan.json" \
GIADA_ROOT="$GIADA_ROOT" GIADA_TEACHER_ROOT="$TEACHER_ROOT" \
GIADA_PYTHON="$PYTHON_BIN" GIADA_WORKER_COUNT="$WORKER_COUNT" \
  bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" &
launcher_pid=$!
while kill -0 "$launcher_pid" 2>/dev/null; do
  completed="$(find "$OUTPUT_ROOT/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
  failed="$(find "$OUTPUT_ROOT/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
  echo "[GIADA validity contextual] shards $completed/$expected; failed $failed"
  sleep 20
done
wait "$launcher_pid"
"$PYTHON_BIN" -m src.giada_runpod.cli validate \
  --plan "$OUTPUT_ROOT/plan.json" --output "$OUTPUT_ROOT"
"$PYTHON_BIN" -m src.giada_runpod.cli verify-output-spikes \
  --corpus "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/output_spike_verification.json"
"$PYTHON_BIN" -m src.giada_runpod.cli seal-neuronio-contextual-eval \
  --corpus "$OUTPUT_ROOT"
echo "[GIADA validity contextual] sealed: $OUTPUT_ROOT"
