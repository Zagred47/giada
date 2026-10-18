"""Isolated Task39b launcher, fail with durable technical report."""
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
from src.giada_teacher.task39b_architecture_diagnosis import run

class Progress:
    def write(self,value):return sys.__stdout__.write(value.replace('[GIADA Task38]','[GIADA Task39b]'))
    def flush(self):return sys.__stdout__.flush()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);sys.stdout=Progress()
    try:
        (args.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        result=run(ROOT,args.output);print('[GIADA Task39b] '+result['decision'],flush=True)
    except BaseException as exc:
        (args.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,
            error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();sys.exit(1)
