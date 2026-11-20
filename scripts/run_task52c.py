import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52c_component_repair as q
if __name__=='__main__':
    source,c,d,package,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=q.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        q.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52c',torch=torch.__version__,gpu=torch.cuda.get_device_name()));q.run(torch,ROOT,source,c,d,package,out)
    except BaseException as error:
        q.a.n.p.save_json(out/'failure_report.json',dict(valid=False,complete=False,task='52c',error=repr(error),traceback=traceback.format_exc()));raise
