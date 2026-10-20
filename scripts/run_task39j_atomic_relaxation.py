"""Isolated39j worker; preserve technical failure and checkpoints separately."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task39j_atomic_relaxation import run
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        (args.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        report=run(ROOT,args.output,smoke=args.smoke);print('[GIADA Task39j] '+report['decision'],flush=True)
    except BaseException as exc:
        (args.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();sys.exit(1)
