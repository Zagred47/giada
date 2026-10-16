"""Isolated CUDA worker for IE-P5, durable failure reporting."""
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
    (output/'code_revision.txt').write_text(revision)
    shutil.copy2(ROOT/'experiments/ie_p5_accuracy_execution_preregistration_v1.json',output/'preregistration.json')
    try:
        from src.giada_teacher.ie_accuracy_execution import run
        run(ROOT,output,revision)
    except Exception as error:
        (output/'failure_report.json').write_text(json.dumps({'valid':False,'scientific_no_go':False,
            'error':str(error),'traceback':traceback.format_exc(),'code_revision':revision},indent=2))
        raise
