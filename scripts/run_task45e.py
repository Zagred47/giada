"""Pinned GPU worker; native fixtures reused, no biological threshold edits."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45e_anchored_repair as e
def main():
    source,native,checkpoints,saved,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        prior=e.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==prior['torch'] and torch.cuda.get_device_name()==prior['gpu']
        e.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),NEURON_loaded=False))
        e.n.p.save_json(out/'preregistration.json',e.spec(ROOT)[0]);print('[GIADA45e] GPU preflight passed; all inputs hashed before fit',flush=True)
        e.evaluate(torch,ROOT,source,native,checkpoints,saved,out)
    except BaseException as ex:
        failure=dict(valid=False,complete=False,technical_failure=True,error=repr(ex),traceback=traceback.format_exc())
        e.n.p.save_json(out/'failure_report.json',failure);e.n.p.save_json(out/'final_report.json',failure);raise
if __name__=='__main__':main()
