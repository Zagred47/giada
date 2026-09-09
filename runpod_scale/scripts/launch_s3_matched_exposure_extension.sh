#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
PYTHON_BIN="${GIADA_GPU_PYTHON:-python}"
CORPUS_ROOT="${GIADA_S3_CORPUS:-/workspace/giada-data/s3-hybrid-production-v1/composite}"
SOURCE_ROOT="${GIADA_S3_FORENSIC_SOURCE:-/workspace/giada-results/s3-optimization-forensic-v2}"
RESULT_ROOT="${GIADA_S3_MATCHED_EXPOSURE_RESULTS:-/workspace/giada-results/s3-matched-exposure-v1}"
BASE_CONFIG="$GIADA_ROOT/runpod_scale/configs/s3_matched_training.yml"
EXTENSION_CONFIG="$GIADA_ROOT/runpod_scale/configs/s3_matched_exposure_extension.yml"

cd "$GIADA_ROOT"

"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})"
"$PYTHON_BIN" -m pytest tests/test_giada_runpod_scale.py -q

if [[ -f "$RESULT_ROOT/final_report.json" ]]; then
  echo "Completed matched-exposure result already exists: $RESULT_ROOT/final_report.json" >&2
  exit 1
fi

"$PYTHON_BIN" -m src.giada_runpod.cli extend-s3-matched-exposure \
  --base-config "$BASE_CONFIG" \
  --config "$EXTENSION_CONFIG" \
  --corpus "$CORPUS_ROOT" \
  --source "$SOURCE_ROOT" \
  --output "$RESULT_ROOT" \
  --elm-repo "$GIADA_ROOT"

echo "[GIADA RunPod][S3 matched exposure] complete: $RESULT_ROOT/final_report.json"
