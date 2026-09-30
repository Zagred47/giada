"""Run Task 17b rate-attribution matrix in an isolated native subprocess."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "teacher", "task15c", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if not args.worker:
        from src.giada_teacher.native_process import supervise_native
        _, code = supervise_native(
            [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
             *sys.argv[1:], "--worker"], args.output)
        raise SystemExit(0 if code == 0 and (args.output / "final_report.json").is_file() else 1)
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    try:
        from src.giada_teacher.roadmap_task17b_rate_attribution import run_task17b
        report = run_task17b(
            args.repo, args.teacher,
            args.teacher / "L5PC_NEURON_simulation/mods/Ca_HVA.mod",
            args.task15c, args.output, code_revision=revision)
        print(json.dumps({key: report[key] for key in
                          ("valid", "decision", "gate_c_authorized",
                           "full_lut_worst_gate_error")}), flush=True)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure_report.json").write_text(json.dumps({
            "schema_version": "giada-task17b-failure-v1", "valid": False,
            "decision": "PRECONDITION_OR_RUNTIME_NO_GO", "gate_c_authorized": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc()[-12000:], "code_revision": revision,
        }, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
