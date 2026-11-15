"""Single explicit GPU entry point; immutable inputs and partial reports retained."""
import json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task48_state_minimality as q

def main():
    source,c,d,out=map(Path,sys.argv[1:5]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pinned=q.a.n.contract(ROOT)[0]
        assert torch.cuda.is_available() and torch.__version__==pinned['torch'] and torch.cuda.get_device_name()==pinned['gpu']
        q.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),protocol=q.identity(q.plan(ROOT))))
        with torch.inference_mode():q.run(torch,ROOT,source,c,d,out)
    except BaseException as ex:
        q.a.n.p.save_json(out/'failure_report.json',dict(valid=False,complete=False,task='48',technical_failure=True,error=repr(ex),traceback=traceback.format_exc(),training_updates=0,thresholds_changed=False))
        raise
if __name__=='__main__':main()
