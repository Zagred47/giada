"""Isolated Task39e worker: preserve technical failures separately from NO-GO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher.task39e_tail_support_heads import run
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        (args.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        result=run(ROOT,args.output);print('[GIADA Task39e] '+result['decision'],flush=True)
    except BaseException as exc:
        (args.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();sys.exit(1)
