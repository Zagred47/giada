"""Frozen evaluation recovery only: no optimizer, training or native generation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41n_evaluation_recovery as r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        spec,*_=r.n.contract(ROOT)
        assert torch.cuda.is_available() and torch.cuda.get_device_name()==spec['gpu'] and torch.__version__==spec['torch']
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
        with torch.inference_mode():result=r.run(torch,ROOT,a.source,a.output)
        print('[GIADA41n recovery] '+result['decision'],flush=True)
    except BaseException as error:
        a.output.mkdir(exist_ok=True)
        r.n.p.save_json(a.output/'failure_report.json',dict(valid=False,technical_failure=True,error=repr(error),traceback=traceback.format_exc(),training_updates_performed=0,selection_recomputed=False,thresholds_changed=False));raise
if __name__=='__main__':main()
