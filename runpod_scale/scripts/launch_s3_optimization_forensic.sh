#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
PYTHON_BIN="${GIADA_GPU_PYTHON:-python}"
CORPUS_ROOT="${GIADA_S3_CORPUS:-/workspace/giada-data/s3-hybrid-production-v1/composite}"
SOURCE_ROOT="${GIADA_S3_SOURCE_RESULTS:-/workspace/giada-results/s3-matched-v1}"
RESULT_ROOT="${GIADA_S3_FORENSIC_RESULTS:-/workspace/giada-results/s3-optimization-forensic-v1}"
BASE_CONFIG="$GIADA_ROOT/runpod_scale/configs/s3_matched_training.yml"
FORENSIC_CONFIG="$GIADA_ROOT/runpod_scale/configs/s3_late_optimization_forensic.yml"

cd "$GIADA_ROOT"

"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})"
"$PYTHON_BIN" -m pytest tests/test_giada_runpod_scale.py -q

if [[ -f "$RESULT_ROOT/final_report.json" ]]; then
  echo "Completed S3 forensic already exists: $RESULT_ROOT/final_report.json" >&2
  exit 1
fi

"$PYTHON_BIN" -m src.giada_runpod.cli forensic-s3 \
  --base-config "$BASE_CONFIG" \
  --config "$FORENSIC_CONFIG" \
  --corpus "$CORPUS_ROOT" \
  --source "$SOURCE_ROOT" \
  --output "$RESULT_ROOT" \
  --elm-repo "$GIADA_ROOT"

echo "[GIADA RunPod][S3 forensic] complete: $RESULT_ROOT/final_report.json"
