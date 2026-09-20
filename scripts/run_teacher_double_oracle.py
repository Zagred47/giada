from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.giada_teacher import (
    ExtractedGateFormula,
    NeuronIsolatedGateOracle,
    compile_nmodl,
    run_double_oracle,
    write_double_oracle_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GIADA Task 0.4 Ca_HVA double oracle")
    parser.add_argument("mod_file", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    formula = ExtractedGateFormula.from_mod(args.mod_file)
    mechanism_root = compile_nmodl(args.mod_file, args.build_dir)
    neuron_oracle = NeuronIsolatedGateOracle(mechanism_root)
    report = run_double_oracle(formula, neuron_oracle)
    write_double_oracle_report(report, args.json, args.markdown)
    print(json.dumps({key: report[key] for key in (
        "valid", "failure_count", "maximum_absolute_error", "grid"
    )}, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
