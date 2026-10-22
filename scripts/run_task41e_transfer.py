"""Checkpointed Task41e worker shared by the notebook and Colab console."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41e_transfer_model as m
from src.giada_teacher import task41e_transfer_training as training


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        assert not dirty,'Scientific worker requires the pinned clean checkout'
        m.p.save_json(args.output/'revision.json',dict(revision=revision,dirty_runtime=False))
        spec,*_=m.contract(ROOT)
        shutil.copy2(ROOT/m.d.prior.contract(ROOT)[0]['checkpoint'],args.output/'P110_source.pt')
        report=training.run(ROOT,args.output,smoke=args.smoke)
        print('[GIADA Task41e] '+report['decision'],flush=True)
        return 0
    except BaseException as exc:
        m.p.save_json(args.output/'failure_report.json',dict(valid=False,technical_failure=True,scientific_no_go=False,
                    error=repr(exc),traceback=traceback.format_exc()))
        traceback.print_exc()
        raise


if __name__=='__main__':raise SystemExit(main())
