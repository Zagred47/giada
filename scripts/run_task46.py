"""Isolate CUDA/native phases, fail loudly and preserve each completed block."""
import json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task46_spatial_cost as a
def gpu():
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    p=a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==p['torch'] and torch.cuda.get_device_name()==p['gpu']
    return torch
def preflight_output(out):
    scratch=out.parent/'preflight46'/'models';scratch.mkdir(parents=True,exist_ok=False);return scratch
def main():
    source,ck45c,ck45d,out=map(Path,sys.argv[1:5]);phase=sys.argv[5] if len(sys.argv)>5 else 'driver';out.mkdir(exist_ok=True);current=phase
    try:
        if phase=='preflight':
            torch=gpu();scratch=preflight_output(out)
            a.load_frozen(torch,ROOT,source,ck45c,ck45d,scratch,'cuda')
            a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,frozen_P_verified=True,training_updates=0,new_payload_upload=False))
        elif phase=='native':
            teacher=ROOT.parent/'neuron_as_deep_net';assert subprocess.check_output(['git','-C',str(teacher),'rev-parse','HEAD'],text=True).strip()==a.spec(ROOT)[0]['teacher_revision']
            a.native(ROOT,teacher,out)
        elif phase=='gpu':
            torch=gpu()
            with torch.inference_mode():a.evaluate(torch,ROOT,source,ck45c,ck45d,out)
        else:
            a.n.p.save_json(out/'preregistration.json',a.spec(ROOT)[0])
            for current in ('preflight','native','gpu'):
                print('[GIADA'+a.spec(ROOT)[0]['task']+'] phase '+current,flush=True)
                executable=os.environ.get('GIADA_NATIVE_PYTHON',sys.executable) if current=='native' else sys.executable
                env=dict(os.environ,PATH=str(Path(executable).parent)+os.pathsep+os.environ.get('PATH',''))
                subprocess.run([executable,'-u',__file__,str(source),str(ck45c),str(ck45d),str(out),current],cwd=ROOT,check=True,env=env)
    except BaseException as ex:
        report=dict(valid=False,complete=False,technical_failure=True,task=a.spec(ROOT)[0]['task'],phase=current,error=repr(ex),traceback=traceback.format_exc(),training_updates=0,selection_performed=False,thresholds_changed=False)
        a.n.p.save_json(out/'failure_report.json',report)
        if not (out/'final_report.json').exists():a.n.p.save_json(out/'final_report.json',report)
        raise
if __name__=='__main__':main()
