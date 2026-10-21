"""Checkpointed GPU worker; opened regressions only after selection freeze."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,sys,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41d_precision_response as d
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        d.p.save_json(args.output/'revision.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        spec,_,_,_=d.contract(ROOT)
        shutil.copy2(ROOT/d.prior.contract(ROOT)[0]['checkpoint'],args.output/'P110_source.pt')
        report=d.run(ROOT,args.output,smoke=args.smoke);print('[GIADA Task41d] '+report['decision'],flush=True)
    except BaseException as exc:
        d.p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()));traceback.print_exc();raise
