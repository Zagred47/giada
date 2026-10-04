"""Isolated native Task 17f runner, including a development-only pilot mode."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "teacher", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--development-smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(args.repo.resolve()))
    if not args.worker:
        from src.giada_teacher.native_process import supervise_native
        _, code = supervise_native(
            [sys.executable, "-X", "faulthandler", "-u", str(Path(__file__).resolve()),
             *sys.argv[1:], "--worker"], args.output)
        raise SystemExit(0 if code == 0 and (args.output / "final_report.json").is_file() else 1)
    revision = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    source_files = [Path(__file__).resolve(),
                    args.repo / "src/giada_teacher/roadmap_task17f_event_supported_confirmation.py",
                    args.repo / "src/giada_teacher/roadmap_causal_cahva_replacement.py",
                    args.repo / "experiments/task17f_event_supported_confirmation_preregistration.md"]
    provenance = {"code_revision": revision,
        "source_sha256": {str(p.relative_to(args.repo.resolve())): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (p.resolve() for p in source_files)},
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "-C", str(args.repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip())}
    try:
        from src.giada_teacher.roadmap_task17f_event_supported_confirmation import run_task17f
        result = run_task17f(args.repo, args.teacher,
            args.teacher / "L5PC_NEURON_simulation/mods/Ca_HVA.mod", args.output,
            code_revision=revision, pilot_only=args.pilot_only,
            development_smoke=args.development_smoke)
        print(json.dumps({k: result.get(k) for k in
                          ("valid", "decision", "gate_c_authorized", "task18_authorized", "gates")}), flush=True)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "failure_report.json").write_text(json.dumps({
            "valid": False, "decision": "PRECONDITION_OR_RUNTIME_NO_GO",
            "gate_c_authorized": False, "task18_authorized": False,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc()[-12000:], "code_revision": revision,
        }, indent=2), encoding="utf-8")
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "code_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
