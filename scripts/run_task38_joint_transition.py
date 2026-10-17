"""Isolated CUDA worker; every technical failure has a durable diagnostic."""
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
from src.giada_teacher.task38_joint_transition import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        (args.output/'revision.json').write_text(json.dumps({'revision': revision}))
        report = run(ROOT, args.output)
        print('[GIADA Task38] '+report['decision'], flush=True)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'failure_report.json').write_text(json.dumps(dict(valid=False, technical_failure=True,
            scientific_no_go=False, error=repr(exc), traceback=traceback.format_exc()), indent=2))
        traceback.print_exc()
        sys.exit(1)
