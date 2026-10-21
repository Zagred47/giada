"""Isolated Task40b worker; checkpoint before every selection boundary."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task40b_calcium_precision import run
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
    try:
        result=run(ROOT,a.output,smoke=a.smoke);print('[GIADA Task40b] '+result['decision'],flush=True)
    except BaseException as exc:
        (a.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();sys.exit(1)
