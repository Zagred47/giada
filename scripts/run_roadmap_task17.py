"""Run Task 17 causal Ca_HVA microcanary in an isolated Kaggle subprocess."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--task15c", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    if not args.worker:
        from src.giada_teacher.native_process import supervise_native
        _, code = supervise_native(
            [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
             *sys.argv[1:], "--worker"], args.output,
        )
        raise SystemExit(0 if code == 0 and (args.output / "final_report.json").is_file() else 1)
    revision = "unknown"
    provenance = {}
    try:
        from src.giada_teacher.native_process import native_phase
        native_phase("worker_import")
        from src.giada_teacher.roadmap_causal_cahva_replacement import run_causal_replacement
        revision = subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip()
        provenance = {"git_head": revision, "source_sha256": {
            name: hashlib.sha256((args.repo / name).read_bytes()).hexdigest()
            for name in ("scripts/run_roadmap_task17.py",
                         "src/giada_teacher/roadmap_causal_cahva_replacement.py",
                         "src/giada_teacher/native_process.py")},
            "working_tree_dirty": bool(subprocess.check_output(
                ["git", "-C", str(args.repo), "status", "--porcelain"], text=True).strip())}
        report = run_causal_replacement(
            args.repo, args.teacher,
            args.teacher / "L5PC_NEURON_simulation/mods/Ca_HVA.mod",
            args.task15c, args.output, code_revision=revision,
        )
        report["execution_provenance"] = provenance
        (args.output / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({key: report[key] for key in (
            "valid", "decision", "gate_c_authorized", "formula_control_valid",
            "lut_absolute_valid", "paired_effect_valid", "episode_count",
        )}), flush=True)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {"schema_version": "giada-task17-failure-v1", "valid": False,
                   "gate_c_authorized": False, "decision": "PRECONDITION_OR_RUNTIME_NO_GO",
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc()[-12000:],
                   "code_revision": revision, "execution_provenance": provenance}
        (args.output / "failure_report.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(f"[GIADA Task 17] {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
