"""Isolated39h worker with explicit technical-failure receipts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task39h_relaxation_identification import run
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    try:
        (args.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        report=run(ROOT,args.output);print('[GIADA Task39h] '+report['decision'],flush=True)
    except BaseException as exc:
        phase=args.output/'checkpoint_phase.json'
        report=dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc(),checkpoint_phase=json.loads(phase.read_text()) if phase.exists() else None)
        (args.output/'failure_report.json').write_text(json.dumps(report,indent=2));print('[GIADA failure] '+json.dumps(report),flush=True);traceback.print_exc();sys.exit(1)
