"""Isolated CUDA worker for IE-P2; durable reports on every technical failure."""
import faulthandler
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


if __name__=='__main__':
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    (output/'code_revision.txt').write_text(revision,encoding='utf-8')
    shutil.copy2(ROOT/'experiments/ie_p2_residual_cost_preregistration_v1.json',output/'preregistration.json')
    with (output/'fault.log').open('w') as fault:
        faulthandler.enable(file=fault)
        try:
            from src.giada_teacher.ie_residual_cost import run
            run(ROOT,output,revision)
        except Exception as error:
            (output/'failure_report.json').write_text(json.dumps({'valid':False,
                'scientific_no_go':False,'error':str(error),'traceback':traceback.format_exc(),
                'code_revision':revision},indent=2),encoding='utf-8')
            raise
