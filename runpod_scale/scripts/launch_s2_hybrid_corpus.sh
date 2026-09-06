#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
WORKER_COUNT="${GIADA_WORKER_COUNT:-8}"
S2_ROOT="${GIADA_S2_ROOT:-/workspace/giada-data/s2-hybrid-production-v1}"
BACKGROUND_ROOT="$S2_ROOT/background"
TARGETED_ROOT="$S2_ROOT/targeted"
COMPOSITE_ROOT="$S2_ROOT/composite"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GIADA Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

cd "$GIADA_ROOT"
mkdir -p "$S2_ROOT/logs"

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
  local expected launcher_pid completed failed
  mkdir -p "$output/status"
  expected="$($PYTHON_BIN -c "import json; print(len(json.load(open('$output/plan.json'))['shards']))")"
  echo "[GIADA RunPod][S2] starting $label"
  GIADA_OUTPUT_ROOT="$output" \
  GIADA_PLAN="$output/plan.json" \
  GIADA_ROOT="$GIADA_ROOT" \
  GIADA_TEACHER_ROOT="$TEACHER_ROOT" \
  GIADA_PYTHON="$PYTHON_BIN" \
  GIADA_WORKER_COUNT="$WORKER_COUNT" \
    bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" &
  launcher_pid=$!
  while kill -0 "$launcher_pid" 2>/dev/null; do
    completed="$(find "$output/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
    failed="$(find "$output/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
    echo "[GIADA RunPod][S2][$label] shards $completed/$expected; failed markers $failed"
    sleep 30
  done
  if ! wait "$launcher_pid"; then
    echo "[GIADA RunPod][S2][$label] worker launcher failed; recent worker errors:" >&2
    for worker_log in "$output"/logs/worker-*.log; do
      [[ -f "$worker_log" ]] || continue
      echo "--- $(basename "$worker_log") ---" >&2
      tail -n 12 "$worker_log" >&2
    done
    return 1
  fi
  completed="$(find "$output/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
  failed="$(find "$output/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
  if [[ "$completed" -ne "$expected" || "$failed" -ne 0 ]]; then
    echo "[GIADA RunPod][S2][$label] generation incomplete: completed=$completed/$expected failed=$failed" >&2
    for worker_log in "$output"/logs/worker-*.log; do
      [[ -f "$worker_log" ]] || continue
      echo "--- $(basename "$worker_log") ---" >&2
      tail -n 12 "$worker_log" >&2
    done
    return 1
  fi
  "$PYTHON_BIN" -m src.giada_runpod.cli validate \
    --plan "$output/plan.json" \
    --output "$output"
  echo "[GIADA RunPod][S2] completed and validated $label"
}

plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s2_hybrid_background.yml" "$BACKGROUND_ROOT"
plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s2_hybrid_targeted.yml" "$TARGETED_ROOT"

run_component "long stochastic background" "$BACKGROUND_ROOT"
run_component "confirmed targeted episodes" "$TARGETED_ROOT"

"$PYTHON_BIN" -m src.giada_runpod.cli compose-s2 \
  --background "$BACKGROUND_ROOT" \
  --targeted "$TARGETED_ROOT" \
  --output "$COMPOSITE_ROOT"

echo "[GIADA RunPod][S2] production corpus complete: $COMPOSITE_ROOT"
