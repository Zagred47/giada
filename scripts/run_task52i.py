import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52i_solution_retention as i
if __name__=='__main__':
    source,cost,recovery,package,warm,out=map(Path,sys.argv[1:7]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=i.f.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        i.f.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52i',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        i.run(torch,ROOT,source,cost,recovery,package,warm,out)
    except BaseException as e:
        i.f.a.n.p.save_json(out/'failure_report.json',dict(valid=False,task='52i',error=repr(e),traceback=traceback.format_exc()));raise
