import sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52h_ih_preservation as h
if __name__=='__main__':
    source,cost,recovery,package,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=False)
    try:
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=h.f.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        h.f.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52h',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        h.run(torch,ROOT,source,cost,recovery,package,out)
    except h.FeasibilityBlocked as e:
        h.f.a.n.p.save_json(out/'final_report.json',dict(valid=True,execution_valid=True,complete=False,task='52h',preflight_blocked=True,full_training_started=False,training_updates=0,scientific_acceptance_passed=False,promotion_authorized=False,thresholds_changed=False,abstraction_changed=False,reason=str(e),qualification='Bounded fit-only diagnostic; not proof of global infeasibility'))
        print('[GIADA52h] feasibility preflight blocked full training; diagnostic report saved',flush=True)
    except BaseException as e:
        h.f.a.n.p.save_json(out/'failure_report.json',dict(valid=False,task='52h',error=repr(e),traceback=traceback.format_exc()));raise
