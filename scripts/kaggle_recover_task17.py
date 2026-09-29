"""Paste this entire file into ONE Kaggle cell in the existing Task17 notebook.

Then execute its ZIP cell. No kernel restart, output deletion, or cached module
reload is needed: simulation code runs in a new supervised Python process.
"""

from pathlib import Path
import json
import subprocess
import sys
import uuid

REPO = Path(globals().get("REPO", "/kaggle/working/giada_roadmap_task17/giada"))
TEACHER = Path(globals().get("TEACHER", REPO.parent / "neuron_as_deep_net"))
assert REPO.is_dir() and TEACHER.is_dir(), "Usa il notebook Task17 aggiornato se i checkout non esistono."
assert globals().get("TASK15C") is not None, "Esegui prima la cella che individua l'input Task15c."
subprocess.run(["git", "-C", str(REPO), "fetch", "origin", "codex/surrogate-validity-audit"], check=True)
subprocess.run(["git", "-C", str(REPO), "checkout", "--detach", "FETCH_HEAD"], check=True)
REVISION = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
assert (REPO / "src/giada_teacher/native_process.py").is_file(), "Revisione senza supervisore nativo."
OUTPUT = Path("/kaggle/working/artifacts") / f"giada_roadmap_task17_causal_replacement_{REVISION[:7]}_{uuid.uuid4().hex[:8]}"
print({"revision": REVISION, "output": str(OUTPUT)}, flush=True)
completed = subprocess.run([
    sys.executable, "-u", str(REPO / "scripts/run_roadmap_task17.py"),
    "--repo", str(REPO), "--teacher", str(TEACHER),
    "--task15c", str(TASK15C), "--output", str(OUTPUT),
], check=False)
result_path = OUTPUT / ("failure_report.json" if (OUTPUT / "failure_report.json").is_file() else "final_report.json")
if result_path.is_file():
    report = json.loads(result_path.read_text())
else:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"valid": False, "gate_c_authorized": False, "decision": "SUPERVISOR_INTERRUPTED",
              "error": f"Supervisore terminato con exit={completed.returncode}", "output_dir": str(OUTPUT)}
    (OUTPUT / "failure_report.json").write_text(json.dumps(report, indent=2))
if completed.returncode and report.get("valid"):
    report = {**report, "valid": False, "error": f"Exit anomalo: {completed.returncode}"}
from IPython.display import display
display({key: report.get(key) for key in (
    "valid", "decision", "formula_control_valid", "lut_absolute_valid", "paired_effect_valid",
    "episode_count", "gate_c_authorized", "error_type", "error", "process")})
print("Ora esegui la cella ZIP aggiornata: usa OUTPUT e include process.log/process_status.json.")
