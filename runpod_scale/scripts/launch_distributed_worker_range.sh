#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
OUTPUT_ROOT="${GIADA_OUTPUT_ROOT:?GIADA_OUTPUT_ROOT is required}"
DISTRIBUTED_PLAN="${GIADA_DISTRIBUTED_PLAN:-$OUTPUT_ROOT/distributed_plan}"
GLOBAL_WORKER_COUNT="${GIADA_GLOBAL_WORKER_COUNT:?GIADA_GLOBAL_WORKER_COUNT is required}"
WORKER_START="${GIADA_WORKER_START:?GIADA_WORKER_START is required}"
LOCAL_WORKER_COUNT="${GIADA_LOCAL_WORKER_COUNT:?GIADA_LOCAL_WORKER_COUNT is required}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
POD_LABEL="${GIADA_POD_LABEL:-$(hostname)}"
RECOVER_STALE_CLAIMS="${GIADA_RECOVER_STALE_CLAIMS:-0}"

for value in "$GLOBAL_WORKER_COUNT" "$WORKER_START" "$LOCAL_WORKER_COUNT"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "worker counts and offsets must be non-negative integers" >&2
    exit 2
  fi
done
if ((GLOBAL_WORKER_COUNT <= 0 || LOCAL_WORKER_COUNT <= 0)); then
  echo "global and local worker counts must be positive" >&2
  exit 2
fi
if ((WORKER_START + LOCAL_WORKER_COUNT > GLOBAL_WORKER_COUNT)); then
  echo "local worker range exceeds immutable global worker count" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GIADA Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$DISTRIBUTED_PLAN/manifest.json" ]]; then
  echo "distributed manifest not found at $DISTRIBUTED_PLAN/manifest.json" >&2
  exit 1
fi
if [[ "$RECOVER_STALE_CLAIMS" != "0" && "$RECOVER_STALE_CLAIMS" != "1" ]]; then
  echo "GIADA_RECOVER_STALE_CLAIMS must be 0 or 1" >&2
  exit 2
fi

git config --global --add safe.directory "$GIADA_ROOT"
git config --global --add safe.directory "$TEACHER_ROOT"
git -C "$GIADA_ROOT" rev-parse --verify HEAD >/dev/null
git -C "$TEACHER_ROOT" rev-parse --verify HEAD >/dev/null

LOG_ROOT="$OUTPUT_ROOT/logs/pods/$POD_LABEL/launcher-$$"
mkdir -p "$LOG_ROOT"
cd "$GIADA_ROOT"
pids=()
workers=()
for ((offset=0; offset<LOCAL_WORKER_COUNT; offset++)); do
  worker=$((WORKER_START + offset))
  recovery=()
  if [[ "$RECOVER_STALE_CLAIMS" == "1" ]]; then
    recovery+=(--recover-stale-claims)
  fi
  env \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "$PYTHON_BIN" -m src.giada_runpod.cli worker \
    --distributed-plan "$DISTRIBUTED_PLAN" \
    --output "$OUTPUT_ROOT" \
    --elm-repo "$GIADA_ROOT" \
    --teacher-repo "$TEACHER_ROOT" \
    --worker-index "$worker" \
    --worker-count "$GLOBAL_WORKER_COUNT" \
    --worker-seed "$((7000001 + worker))" \
    "${recovery[@]}" \
    >"$LOG_ROOT/worker-$(printf '%05d' "$worker").log" 2>&1 &
  pids+=("$!")
  workers+=("$worker")
done

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[GIADA RunPod][distributed] global worker ${workers[$index]} failed" >&2
    failures=$((failures + 1))
  fi
done
if ((failures > 0)); then
  echo "[GIADA RunPod][distributed] $failures/$LOCAL_WORKER_COUNT local workers failed; inspect $LOG_ROOT" >&2
  exit 1
fi
echo "[GIADA RunPod][distributed] global worker range $WORKER_START..$((WORKER_START + LOCAL_WORKER_COUNT - 1)) complete"
