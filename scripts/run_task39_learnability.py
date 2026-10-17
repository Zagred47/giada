"""Isolated Task39 worker; durable technical diagnostics and compact output."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.giada_teacher.task39_learnability import run

class Progress:
    def write(self, value): return sys.__stdout__.write(value.replace('[GIADA Task38]', '[GIADA Task39]'))
    def flush(self): return sys.__stdout__.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path); a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    sys.stdout = Progress()
    try:
        (a.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        result = run(ROOT, a.output)
        print('[GIADA Task39] '+result['decision'], flush=True)
    except BaseException as exc:
        (a.output/'failure_report.json').write_text(json.dumps(dict(valid=False, technical_failure=True,
            scientific_no_go=False, error=repr(exc), traceback=traceback.format_exc()), indent=2))
        traceback.print_exc(); sys.exit(1)
