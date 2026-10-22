"""Pinned notebook/console entrypoint for41g."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41g_preservation as g

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        g.p.save_json(args.output/'revision.json',dict(revision=revision,dirty_runtime=False))
        report=g.run(ROOT,args.output,smoke=args.smoke)
        g.p.save_json(args.output/'final_report.json',report)
        print('[GIADA Task41g] '+report['decision'],flush=True)
    except BaseException as exc:
        g.p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();raise

if __name__=='__main__':main()
