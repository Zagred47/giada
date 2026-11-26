import sys,traceback,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52i_solution_retention as i
if __name__=='__main__':
    source,cost,recovery,package,warm,out=map(Path,sys.argv[1:7]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=i.f.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        i.f.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52j',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        p=json.loads((ROOT/'experiments/task52j_preregistration_v1.json').read_text());i.run(torch,ROOT,source,cost,recovery,package,warm,out,override=p)
        report=json.loads((out/'final_report.json').read_text());report['task']='52j';report['next_phase']='53 with unresolved limitations; no52k';i.f.a.n.p.save_json(out/'final_report.json',report)
    except BaseException as e:
        i.f.a.n.p.save_json(out/'failure_report.json',dict(valid=False,task='52j',error=repr(e),traceback=traceback.format_exc()));raise
