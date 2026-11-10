"""45b pinned GPU preflight and isolated native/training/evaluation processes."""
import json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task45b_rate_coverage as a

def gpu_preflight():
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    prior=a.n.contract(ROOT)[0]
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==prior['gpu'] and torch.__version__==prior['torch']
    q=torch.arange(12,device='cuda',dtype=torch.float64).reshape(3,4)
    assert bool(torch.isfinite(q@q.T).all())
    return torch

def main():
    source,out=map(Path,sys.argv[1:3]);phase=sys.argv[3] if len(sys.argv)>3 else 'driver';out.mkdir(exist_ok=True)
    try:
        if phase=='preflight':
            torch=gpu_preflight();a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),finite=True))
        elif phase=='native':
            teacher=ROOT.parent/'neuron_as_deep_net';rev=json.loads((ROOT/'experiments/task35_full_local_compartment_preregistration.json').read_text())['teacher_revision']
            assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==rev
            a.native(ROOT,teacher,out)
        elif phase=='gpu':a.evaluate(gpu_preflight(),ROOT,source,out)
        else:
            a.n.p.save_json(out/'preregistration.json',a.spec(ROOT)[0])
            for part in ('preflight','native','gpu'):
                print('[GIADA45b] phase '+part,flush=True)
                subprocess.run([sys.executable,'-u',__file__,str(source),str(out),part],check=True,cwd=ROOT)
    except BaseException as error:
        a.n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,phase=phase,error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
