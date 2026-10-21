"""Task41b isolated worker, checkpointed before selection and external recovery."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,traceback,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task41b_local_precision import run,p

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        p.save_json(args.output/'revision.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        result=run(ROOT,args.output,smoke=args.smoke);print('[GIADA Task41b] '+result['decision'],flush=True)
    except BaseException as exc:
        p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc();raise
