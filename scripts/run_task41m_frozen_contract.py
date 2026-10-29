"""No-training worker for preregistered41m, including durable failures."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41m_frozen_contract as m

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        m.p.save_json(a.output/'revision.json',dict(revision=revision,dirty=False))
        result=m.run(ROOT,a.output,preflight=a.preflight)
        print('[GIADA Task41m] '+json.dumps({k:result[k] for k in ('valid','complete','decision') if k in result}),flush=True)
    except BaseException as error:
        m.p.save_json(a.output/'failure_report.json',dict(valid=False,scientific_no_go=False,
            budget_exhausted=isinstance(error,m.BudgetReached),technical_failure=not isinstance(error,m.BudgetReached),error=repr(error),traceback=traceback.format_exc(),training_updates_performed=0))
        raise

if __name__=='__main__':main()
