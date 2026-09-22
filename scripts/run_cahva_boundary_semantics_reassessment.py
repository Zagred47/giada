"""Run the Task 7c frozen forensic reassessment outside Kaggle's kernel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.giada_teacher.cahva_boundary_semantics_reassessment import (  # noqa: E402
    reassess_boundary_semantics,
)
from src.giada_teacher.double_oracle import ExtractedGateFormula  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mod", required=True, type=Path)
    parser.add_argument("--task5-source", required=True, type=Path)
    parser.add_argument("--task7b-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    formula = ExtractedGateFormula.from_mod(args.mod)
    report = reassess_boundary_semantics(
        formula, args.task5_source, args.task7b_source, args.output_dir,
        code_revision=args.revision,
    )
    print({
        "valid": report["valid"],
        "formula_voltage_max_abs_mv": report["corrected_formula_voltage_max_abs_mv"],
        "formula_gate_max_rmse": report["corrected_formula_gate_max_rmse"],
        "formula_current_max_abs_ma_cm2": report["corrected_formula_current_max_abs_ma_cm2"],
        "teacher_regenerated": report["teacher_regenerated"],
        "model_retrained": report["model_retrained"],
    }, flush=True)


if __name__ == "__main__":
    main()
