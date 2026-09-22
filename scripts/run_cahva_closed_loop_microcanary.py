"""Run Task 7 or 7b native experiments outside the notebook kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.giada_teacher import (  # noqa: E402
    ActiveClosedLoopCaHVAConfig,
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
    parser.add_argument("--design", choices=("task7", "task7b"), default="task7")
    args = parser.parse_args()
    print(f"[GIADA {args.design}] loading formula", flush=True)
    formula = ExtractedGateFormula.from_mod(args.mod)
    print("[GIADA Task 7] starting native microcanary", flush=True)
    report = run_closed_loop_microcanary(
        formula,
        args.task5_root,
        args.mechanism_root,
        args.output_dir,
        ActiveClosedLoopCaHVAConfig() if args.design == "task7b" else ClosedLoopCaHVAConfig(),
        code_revision=args.revision,
    )
    print(f"[GIADA {args.design}] completed; valid={report['valid']}", flush=True)


if __name__ == "__main__":
    main()
