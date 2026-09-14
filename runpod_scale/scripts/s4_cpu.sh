#!/usr/bin/env bash
set -euo pipefail
# Identical command on every pod: hostname assignment is read from the registry.
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${GIADA_PYTHON:-/workspace/.giada-venv/bin/python}"
cd "$REPO"
exec "$PYTHON_BIN" -u -m src.giada_runpod.s4_production "$@"
