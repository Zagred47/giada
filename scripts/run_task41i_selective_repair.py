"""Pinned Task41i notebook/console worker; bounded diagnostic output."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41i_selective_repair as i

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        i.p.save_json(args.output/'revision.json',dict(revision=revision,dirty_runtime=False))
        report=i.run(ROOT,args.output,smoke=args.smoke);i.p.save_json(args.output/'final_report.json',report)
        print('[GIADA Task41i] '+report['decision'],flush=True)
    except BaseException as exc:
        i.p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
