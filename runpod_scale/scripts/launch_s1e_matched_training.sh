#!/usr/bin/env bash
set -euo pipefail

GIADA_ROOT="${GIADA_ROOT:-/workspace/giada}"
PYTHON_BIN="${GIADA_GPU_PYTHON:-python}"
CORPUS_ROOT="${GIADA_S1E_CORPUS:-/workspace/giada-data/s1e-hybrid-production-v1/composite}"
RESULT_ROOT="${GIADA_S1E_RESULTS:-/workspace/giada-results/s1e-matched-v1}"
CONFIG_PATH="$GIADA_ROOT/runpod_scale/configs/s1e_matched_training.yml"

cd "$GIADA_ROOT"

"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)})"
"$PYTHON_BIN" -m pytest tests/test_giada_runpod_scale.py -q

if [[ -f "$RESULT_ROOT/final_report.json" ]]; then
  echo "Completed S1e result already exists: $RESULT_ROOT/final_report.json" >&2
  exit 1
fi

"$PYTHON_BIN" -m src.giada_runpod.cli train \
  --config "$CONFIG_PATH" \
  --corpus "$CORPUS_ROOT" \
  --output "$RESULT_ROOT" \
  --elm-repo "$GIADA_ROOT"

echo "[GIADA RunPod][S1e matched] complete: $RESULT_ROOT/final_report.json"
