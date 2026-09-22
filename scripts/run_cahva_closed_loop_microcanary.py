"""Run the Task 7 native NEURON/CUDA experiment outside the notebook kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.giada_teacher import (  # noqa: E402
    ClosedLoopCaHVAConfig,
    ExtractedGateFormula,
    run_closed_loop_microcanary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mod", required=True, type=Path)
    parser.add_argument("--task5-root", required=True, type=Path)
    parser.add_argument("--mechanism-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print("[GIADA Task 7] loading formula", flush=True)
    formula = ExtractedGateFormula.from_mod(args.mod)
    print("[GIADA Task 7] starting native microcanary", flush=True)
    report = run_closed_loop_microcanary(
        formula,
        args.task5_root,
        args.mechanism_root,
        args.output_dir,
        ClosedLoopCaHVAConfig(),
        code_revision=args.revision,
    )
    print(f"[GIADA Task 7] completed; valid={report['valid']}", flush=True)


if __name__ == "__main__":
    main()
