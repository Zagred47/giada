"""Pinned45e evaluation-only worker: no training or selection entry point."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45e_evaluation_recovery as r
def main():
    source,native,saved,control,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        p=r.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==p['torch'] and torch.cuda.get_device_name()==p['gpu']
        print('[GIADA45e recovery] pinned GPU available; evaluating saved models only',flush=True)
        r.recover(torch,ROOT,source,native,saved,control,out)
    except BaseException as ex:
        report=dict(valid=False,complete=False,error=repr(ex),traceback=traceback.format_exc());r.n.p.save_json(out/'failure_report.json',report);r.n.p.save_json(out/'final_report.json',report);raise
if __name__=='__main__':main()
