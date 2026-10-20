"""Isolated frozen Task40 worker; preserve infrastructure failures separately."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher.task40_causal_reinsertion import run
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        import torch,platform
        (args.output/'execution.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            training=False,tf32=False,state_dtype='float32',J1_affine_correction_dtype='float64'),indent=2))
        (args.output/'revision.json').write_text(json.dumps(dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())))
        report=run(ROOT,args.output);print('[GIADA Task40] '+report['decision'],flush=True)
    except BaseException as exc:
        (args.output/'failure_report.json').write_text(json.dumps(dict(valid=False,technical_failure=True,scientific_no_go=False,error=repr(exc),traceback=traceback.format_exc()),indent=2))
        traceback.print_exc();sys.exit(1)
