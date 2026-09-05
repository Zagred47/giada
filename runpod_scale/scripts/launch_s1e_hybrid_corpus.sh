#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
WORKER_COUNT="${GIADA_WORKER_COUNT:-8}"
S1E_ROOT="${GIADA_S1E_ROOT:-/workspace/giada-data/s1e-hybrid-production-v1}"
BACKGROUND_ROOT="$S1E_ROOT/background"
TARGETED_ROOT="$S1E_ROOT/targeted"
COMPOSITE_ROOT="$S1E_ROOT/composite"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GIADA Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

cd "$GIADA_ROOT"
mkdir -p "$S1E_ROOT/logs"

plan_if_missing() {
  local config="$1"
  local output="$2"
  if [[ ! -f "$output/plan.json" ]]; then
    "$PYTHON_BIN" -m src.giada_runpod.cli plan --config "$config" --output "$output"
  fi
}

run_component() {
  local label="$1"
  local output="$2"
  local expected
  local launcher_pid
  mkdir -p "$output/status"
  expected="$($PYTHON_BIN -c "import json; print(len(json.load(open('$output/plan.json'))['shards']))")"
  echo "[GIADA RunPod][S1e] starting $label"
  GIADA_OUTPUT_ROOT="$output" \
  GIADA_PLAN="$output/plan.json" \
  GIADA_ROOT="$GIADA_ROOT" \
  GIADA_TEACHER_ROOT="$TEACHER_ROOT" \
  GIADA_PYTHON="$PYTHON_BIN" \
  GIADA_WORKER_COUNT="$WORKER_COUNT" \
    bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" &
  launcher_pid=$!
  while kill -0 "$launcher_pid" 2>/dev/null; do
    local completed
    local failed
    completed="$(find "$output/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
    failed="$(find "$output/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
    echo "[GIADA RunPod][S1e][$label] shards $completed/$expected; failed markers $failed"
    sleep 30
  done
  wait "$launcher_pid"
  "$PYTHON_BIN" -m src.giada_runpod.cli validate \
    --plan "$output/plan.json" \
    --output "$output"
  echo "[GIADA RunPod][S1e] completed and validated $label"
}

plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s1e_hybrid_background.yml" "$BACKGROUND_ROOT"
plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s1e_hybrid_targeted.yml" "$TARGETED_ROOT"

run_component "long stochastic background" "$BACKGROUND_ROOT"
run_component "confirmed targeted episodes" "$TARGETED_ROOT"

"$PYTHON_BIN" -m src.giada_runpod.cli compose-s1e \
  --background "$BACKGROUND_ROOT" \
  --targeted "$TARGETED_ROOT" \
  --output "$COMPOSITE_ROOT"

echo "[GIADA RunPod][S1e] production corpus complete: $COMPOSITE_ROOT"
