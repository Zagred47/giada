#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
WORKER_COUNT="${GIADA_WORKER_COUNT:-8}"
TEST_ROOT="${GIADA_S3_FRESH_TEST_ROOT:-/workspace/giada-data/s3-fresh-teacher-test-v1}"
BACKGROUND_ROOT="$TEST_ROOT/background"
TARGETED_ROOT="$TEST_ROOT/targeted"
COMPOSITE_ROOT="$TEST_ROOT/composite"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GIADA Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi

cd "$GIADA_ROOT"
mkdir -p "$TEST_ROOT/logs"

plan_if_missing() {
  local config="$1" output="$2"
  if [[ ! -f "$output/plan.json" ]]; then
    "$PYTHON_BIN" -m src.giada_runpod.cli plan --config "$config" --output "$output"
  fi
}

run_component() {
  local label="$1" output="$2" expected launcher_pid completed failed
  mkdir -p "$output/status"
  expected="$($PYTHON_BIN -c "import json; print(len(json.load(open('$output/plan.json'))['shards']))")"
  echo "[GIADA RunPod][S3 fresh test] starting $label"
  GIADA_OUTPUT_ROOT="$output" GIADA_PLAN="$output/plan.json" \
  GIADA_ROOT="$GIADA_ROOT" GIADA_TEACHER_ROOT="$TEACHER_ROOT" \
  GIADA_PYTHON="$PYTHON_BIN" GIADA_WORKER_COUNT="$WORKER_COUNT" \
    bash "$GIADA_ROOT/runpod_scale/scripts/launch_cpu_workers.sh" &
  launcher_pid=$!
  while kill -0 "$launcher_pid" 2>/dev/null; do
    completed="$(find "$output/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
    failed="$(find "$output/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
    echo "[GIADA RunPod][S3 fresh test][$label] shards $completed/$expected; failed $failed"
    sleep 30
  done
  wait "$launcher_pid"
  completed="$(find "$output/status" -maxdepth 1 -name '*.done.json' 2>/dev/null | wc -l)"
  failed="$(find "$output/status" -maxdepth 1 -name '*.failed.json' 2>/dev/null | wc -l)"
  if [[ "$completed" -ne "$expected" || "$failed" -ne 0 ]]; then
    echo "[GIADA RunPod][S3 fresh test][$label] incomplete: $completed/$expected; failed $failed" >&2
    return 1
  fi
  "$PYTHON_BIN" -m src.giada_runpod.cli validate --plan "$output/plan.json" --output "$output"
}

plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s3_fresh_test_background.yml" "$BACKGROUND_ROOT"
plan_if_missing "$GIADA_ROOT/runpod_scale/configs/s3_fresh_test_targeted.yml" "$TARGETED_ROOT"
run_component "fresh stochastic background" "$BACKGROUND_ROOT"
run_component "fresh targeted episodes" "$TARGETED_ROOT"

"$PYTHON_BIN" -m src.giada_runpod.cli compose-s3-fresh-test \
  --background "$BACKGROUND_ROOT" --targeted "$TARGETED_ROOT" --output "$COMPOSITE_ROOT"
"$PYTHON_BIN" -m src.giada_runpod.cli fingerprint-corpus \
  --corpus "$COMPOSITE_ROOT" --output "$COMPOSITE_ROOT/corpus_fingerprint.json"
echo "[GIADA RunPod][S3 fresh test] sealed corpus complete: $COMPOSITE_ROOT"
