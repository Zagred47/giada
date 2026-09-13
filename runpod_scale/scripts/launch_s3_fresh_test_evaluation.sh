#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
PYTHON_BIN="${GIADA_GPU_PYTHON:-python}"
CORPUS_ROOT="${GIADA_S3_FRESH_TEST_CORPUS:-/workspace/giada-data/s3-fresh-teacher-test-v1/composite}"
SOURCE_ROOT="${GIADA_S3_MATCHED_SOURCE:-/workspace/giada-results/s3-matched-exposure-v1}"
OUTPUT_ROOT="${GIADA_S3_FRESH_TEST_RESULTS:-/workspace/giada-results/s3-fresh-teacher-test-v1}"

cd "$GIADA_ROOT"
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})"
"$PYTHON_BIN" -m pytest tests/test_giada_runpod_scale.py -q
"$PYTHON_BIN" -m src.giada_runpod.cli evaluate-s3-fresh-test \
  --config "$GIADA_ROOT/runpod_scale/configs/s3_fresh_test_evaluation.yml" \
  --corpus "$CORPUS_ROOT" \
  --source "$SOURCE_ROOT" \
  --output "$OUTPUT_ROOT" \
  --elm-repo "$GIADA_ROOT"

echo "[GIADA RunPod][S3 fresh test] evaluation complete: $OUTPUT_ROOT/final_report.json"
