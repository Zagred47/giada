"""Pinned notebook and console entrypoint for Task41h."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41h_path_repair as h

def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        h.p.save_json(a.output/'revision.json',dict(revision=revision,dirty_runtime=False))
        report=h.run(ROOT,a.output,smoke=a.smoke);h.p.save_json(a.output/'final_report.json',report)
        print('[GIADA Task41h] '+report['decision'],flush=True)
    except BaseException as exc:
        h.p.save_json(a.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();raise
if __name__=='__main__':main()
