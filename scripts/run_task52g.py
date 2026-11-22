import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52g_gradient_preservation as g
if __name__=='__main__':
    source,cost,recovery,package,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=g.f.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        g.f.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52g',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        g.run(torch,ROOT,source,cost,recovery,package,out)
    except BaseException as error:
        g.f.a.n.p.save_json(out/'failure_report.json',dict(valid=False,complete=False,task='52g',error=repr(error),traceback=traceback.format_exc()));raise
