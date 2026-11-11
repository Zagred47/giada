"""One45d GPU worker, checksummed mounted inputs and no native library loading."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45d_learnability_execution as d
def main():
    source,native,checkpoints,out=map(Path,sys.argv[1:5]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        prior=d.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==prior['torch'] and torch.cuda.get_device_name()==prior['gpu']
        d.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),NEURON_loaded=False))
        d.n.p.save_json(out/'preregistration.json',d.spec(ROOT)[0]);print('[GIADA45d] GPU preflight passed; verifying inputs',flush=True)
        d.evaluate(torch,ROOT,source,native,checkpoints,out)
    except BaseException as e:
        failure=dict(valid=False,complete=False,technical_failure=True,error=repr(e),traceback=traceback.format_exc())
        d.n.p.save_json(out/'failure_report.json',failure);d.n.p.save_json(out/'final_report.json',failure);raise
if __name__=='__main__':main()
