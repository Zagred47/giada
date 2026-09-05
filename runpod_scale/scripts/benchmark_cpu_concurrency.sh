#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
TEACHER_ROOT="${GIADA_TEACHER_ROOT:-/workspace/neuron_as_deep_net}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
WORKER_COUNT="${GIADA_BENCHMARK_WORKERS:-4}"
DURATION_MS="${GIADA_BENCHMARK_DURATION_MS:-3000}"
BASELINE_RATE="${GIADA_BASELINE_RATE:-68.8896756259176}"
OUTPUT_ROOT="${GIADA_BENCHMARK_OUTPUT:-/workspace/giada-data/cpu-concurrency-${WORKER_COUNT}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "GIADA Python environment not found at $PYTHON_BIN" >&2
  exit 1
fi
if [[ -e "$OUTPUT_ROOT/concurrency_report.json" ]]; then
  echo "Completed concurrency report already exists: $OUTPUT_ROOT/concurrency_report.json" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cd "$GIADA_ROOT"
started_epoch="$($PYTHON_BIN -c 'import time; print(time.time())')"
pids=()
for ((worker=0; worker<WORKER_COUNT; worker++)); do
  worker_root="$OUTPUT_ROOT/worker-$worker"
  mkdir -p "$worker_root"
  env MPLBACKEND=Agg "$PYTHON_BIN" -m src.giada_runpod.cli benchmark \
    --config runpod_scale/configs/s1_soma.yml \
    --output "$worker_root" \
    --elm-repo "$GIADA_ROOT" \
    --teacher-repo "$TEACHER_ROOT" \
    --worker-seed "$((7200001 + worker))" \
    --duration-ms "$DURATION_MS" \
    >"$worker_root/benchmark.log" 2>&1 &
  pids+=("$!")
done

while true; do
  running=0
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      running=$((running + 1))
    fi
  done
  completed=$((WORKER_COUNT - running))
  echo "[GIADA RunPod][CPU concurrency ${WORKER_COUNT}] completed ${completed}/${WORKER_COUNT}; running ${running}"
  [[ "$running" -eq 0 ]] && break
  sleep 15
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
finished_epoch="$($PYTHON_BIN -c 'import time; print(time.time())')"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$WORKER_COUNT" "$DURATION_MS" "$BASELINE_RATE" "$started_epoch" "$finished_epoch" "$failures" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
workers = int(sys.argv[2])
duration = int(sys.argv[3])
baseline = float(sys.argv[4])
wall = max(float(sys.argv[6]) - float(sys.argv[5]), 1e-9)
failures = int(sys.argv[7])
reports = []
for worker in range(workers):
    path = root / f"worker-{worker}" / "benchmark_report.json"
    if path.is_file():
        reports.append(json.loads(path.read_text()))
cold_start_rate = len(reports) * duration / wall
worker_generation_rates = [
    float(row["benchmark"]["transitions_per_second"])
    for row in reports
]
steady_state_rate = sum(worker_generation_rates)
peak_rss = [
    row.get("resource_usage", {}).get("peak_process_rss_mib")
    for row in reports
]
peak_rss = [float(value) for value in peak_rss if value is not None]
report = {
    "schema_version": "giada-runpod-cpu-concurrency-v2",
    "valid": failures == 0 and len(reports) == workers,
    "worker_count": workers,
    "duration_ms_per_worker": duration,
    "completed_worker_count": len(reports),
    "failure_count": failures,
    "cold_start_probe_wall_seconds": wall,
    "cold_start_probe_transitions_per_second": cold_start_rate,
    "cold_start_probe_parallel_efficiency_fraction": cold_start_rate / max(workers * baseline, 1e-9),
    "worker_generation_rates": worker_generation_rates,
    "aggregate_transitions_per_second": steady_state_rate,
    "single_worker_reference_transitions_per_second": baseline,
    "parallel_efficiency_fraction": steady_state_rate / max(workers * baseline, 1e-9),
    "projected_s1_wall_minutes": 600000.0 / max(steady_state_rate, 1e-9) / 60.0,
    "maximum_worker_peak_rss_mib": max(peak_rss, default=None),
    "sum_worker_peak_rss_mib": sum(peak_rss) if peak_rss else None,
}
temporary = root / "concurrency_report.json.tmp"
temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
temporary.replace(root / "concurrency_report.json")
print(json.dumps(report, indent=2))
if not report["valid"]:
    raise SystemExit(2)
PY
