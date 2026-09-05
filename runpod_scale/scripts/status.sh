#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${GIADA_OUTPUT_ROOT:-/workspace/giada-data/s1}"
PLAN="${GIADA_PLAN:-$OUTPUT_ROOT/plan.json}"
python - "$PLAN" "$OUTPUT_ROOT" <<'PY'
import json, pathlib, sys, time
plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
shards = plan['shards']
done = list((root / 'status').glob('*.done.json')) if (root / 'status').is_dir() else []
completed = sum(json.loads(path.read_text())['transition_count'] for path in done)
target = plan['config']['target_transitions']
print({
    'shards': f'{len(done)}/{len(shards)}',
    'transitions': f'{completed:,}/{target:,}',
    'percent': round(100 * completed / target, 2),
    'failed_markers': len(list((root / 'status').glob('*.failed.json'))) if (root / 'status').is_dir() else 0,
})
PY
