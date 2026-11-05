"""41o frozen causal diagnostics; no training, no implicit model selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41o_causal_localization as o

def main():
    original,out=map(Path,sys.argv[1:])
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        spec=o.n.contract(ROOT)[0]
        assert torch.cuda.is_available() and torch.cuda.get_device_name()==spec['gpu'] and torch.__version__==spec['torch']
        archives=list(Path('/kaggle/input').rglob('task41n_recovery_final.zip'));assert len(archives)==1
        with torch.inference_mode():r=o.run(torch,ROOT,original,archives[0],out)
        print('[GIADA41o] '+r['decision'],flush=True)
    except BaseException as error:
        out.mkdir(exist_ok=True);o.n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
