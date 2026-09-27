"""Run the Task 16 native replay in a clean process, including after a failure."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher", "dataset-root", "dataset-source", "task5", "task15c", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    from src.giada_teacher import ExtractedGateFormula
    from src.giada_teacher.roadmap_embedded_frozen_confirmation import run_frozen_embedded_confirmation

    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    formula = ExtractedGateFormula.from_mod(
        args.teacher / "L5PC_NEURON_simulation/mods/Ca_HVA.mod")
    report = run_frozen_embedded_confirmation(
        formula, repository, args.teacher, args.dataset_root, args.dataset_source,
        args.task5, args.task15c, args.output, code_revision=revision,
        progress=lambda percent, label: print(
            f"[GIADA Task 16][SHA-256 {label}] {percent}%", flush=True),
    )
    print(json.dumps({"valid": report["valid"], "diagnosis": report.get("diagnosis"),
                      "output": str(args.output), "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
