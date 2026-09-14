#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${GIADA_OUTPUT_ROOT:?GIADA_OUTPUT_ROOT is required}"
DISTRIBUTED_PLAN="${GIADA_DISTRIBUTED_PLAN:-$OUTPUT_ROOT/distributed_plan}"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"

"$PYTHON_BIN" - "$DISTRIBUTED_PLAN" "$OUTPUT_ROOT" <<'PY'
import json
import pathlib
import sys

plan_root = pathlib.Path(sys.argv[1])
output_root = pathlib.Path(sys.argv[2])
if not output_root.is_dir():
    raise SystemExit(f"Generation output does not exist (not started): {output_root}")
manifest = json.loads((plan_root / "manifest.json").read_text())
status = output_root / "status"
claims = output_root / "claims"
done = list(status.glob("*.done.json")) if status.is_dir() else []
failed = list(status.glob("*.failed.json")) if status.is_dir() else []
active_workers = list((claims / "workers").glob("*.claim.json")) if claims.is_dir() else []
active_shards = list((claims / "shards").glob("*.claim.json")) if claims.is_dir() else []
completed = sum(json.loads(path.read_text())["transition_count"] for path in done)
target = int(manifest["transition_count"])
print({
    "shards": f"{len(done):,}/{int(manifest['shard_count']):,}",
    "transitions": f"{completed:,}/{target:,}",
    "percent": round(100 * completed / target, 3),
    "failed_markers": len(failed),
    "claimed_workers": len(active_workers),
    "claimed_shards": len(active_shards),
    "global_worker_count": int(manifest["global_worker_count"]),
})
PY
