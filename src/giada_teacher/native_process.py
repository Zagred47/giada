"""Parent-side supervision: a native abort cannot write its own failure report."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path


def supervise_native(command, output_dir, *, metadata=None):
    """Run a fresh-output worker, tee bounded lines and always retain diagnostics.

    The log/phase files are siblings so the worker can retain its fail-if-output-
    exists contract. Never overwrite an earlier experiment or diagnostic log.
    """
    output = Path(output_dir)
    log = output.with_name(output.name + ".process.log")
    phase = output.with_name(output.name + ".phase.json")
    for path in (output, log, phase):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite previous run: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1",
               GIADA_NATIVE_PHASE_PATH=str(phase))
    returncode = None
    launch_error = None
    with log.open("x", encoding="utf-8") as stream:
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors="replace", env=env) as process:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    # Avoid accidentally flooding the notebook with serialized arrays.
                    print(line[:1500], end="" if line.endswith("\n") else "\n", flush=True)
                returncode = process.wait()
        except OSError as error:
            launch_error = f"{type(error).__name__}: {error}"
            stream.write(launch_error + "\n")
    output.mkdir(parents=True, exist_ok=True)
    last_phase = json.loads(phase.read_text()) if phase.is_file() else None
    status = {"returncode": returncode, "last_phase": last_phase,
              "log": str(log), "launch_error": launch_error, **(metadata or {})}
    if returncode is not None and returncode < 0:
        try:
            status["signal"] = signal.Signals(-returncode).name
        except ValueError:
            status["signal"] = str(-returncode)
    (output / "process_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    final = output / "final_report.json"
    failure = output / "failure_report.json"
    if returncode != 0 or not final.is_file():
        previous = json.loads(failure.read_text()) if failure.is_file() else {}
        # Even a completed final report does not authorize success after abnormal exit.
        report = {"schema_version": "giada-task17-failure-v2", "valid": False,
                  "gate_c_authorized": False, "decision": "PRECONDITION_OR_RUNTIME_NO_GO",
                  "error_type": "NativeProcessFailure" if returncode else "MissingFinalReport",
                  "error": launch_error or f"Worker exit={returncode}; see process_status.json and process.log",
                  **previous, "process": status}
        failure.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        report = json.loads(final.read_text())
    # Include diagnostics in the normal downloadable output as well.
    import shutil
    shutil.copyfile(log, output / "process.log")
    if phase.is_file():
        shutil.copyfile(phase, output / "last_phase.json")
    return report, returncode


def native_phase(label, **details):
    """Persist the last reached phase *before* entering potentially fatal C code."""
    value = {"phase": label, **details}
    destination = os.environ.get("GIADA_NATIVE_PHASE_PATH")
    if destination:
        path = Path(destination)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        temporary.replace(path)
    print(f"[GIADA Task 17] {label}" + (f" {details}" if details else ""), flush=True)
