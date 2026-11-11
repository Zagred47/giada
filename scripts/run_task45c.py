"""One GPU process; reuse checksummed native fixtures, never load NEURON."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45c_preserving_kinetics as a
def main():
    source,fixtures,out=map(Path,sys.argv[1:4]);out.mkdir(exist_ok=True)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        prior=a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.cuda.get_device_name()==prior['gpu'] and torch.__version__==prior['torch']
        p=a.spec(ROOT)[0];a.n.p.save_json(out/'preregistration.json',p)
        test=torch.arange(12,device='cuda',dtype=torch.float64).reshape(3,4);assert bool(torch.isfinite(test@test.T).all())
        a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),native_phase_reused=True))
        print('[GIADA45c] CUDA preflight valid; verifying frozen sources',flush=True)
        a.evaluate(torch,ROOT,source,fixtures,out)
    except BaseException as error:
        a.n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
