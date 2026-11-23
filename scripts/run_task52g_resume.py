import hashlib,json,sys,zipfile,os
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52g_gradient_preservation as g
if __name__=='__main__':
    source,cost,recovery,package,saved,out=map(Path,sys.argv[1:7]);spec=json.loads((ROOT/os.environ.get('GIADA_RESUME_SPEC','experiments/task52g_resume_v1.json')).read_text());out.mkdir(exist_ok=False)
    with saved.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==spec['source_sha256']
    with zipfile.ZipFile(saved) as z:
        assert z.testzip() is None and json.loads(z.read('preregistration.json'))==g.plan(ROOT)
        for name in z.namelist():
            p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name
            if name.endswith('/'):continue
            target=out/(('source_'+name) if name in ('dispatch_request.json','console_process.log','process.log','failure_report.json') else name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    if spec.get('recovery_only_checkpoint'):
        import shutil
        shutil.copy2(out/'recovery_latest.pt',out/'resumable_training.pt')
    g.f.a.n.p.save_json(out/'resume_request.json',spec)
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    pin=g.f.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
    state=torch.load(out/'resumable_training.pt',map_location='cpu',weights_only=True);assert state['update']==spec['checkpoint_update']
    if not spec.get('recovery_only_checkpoint'):
        exact=torch.load(out/f"checkpoint_{spec['checkpoint_update']}.pt",map_location='cpu',weights_only=True);assert all(torch.equal(v,exact[k]) for k,v in state['model'].items())
    g.f.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52g-resume',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
    g.run(torch,ROOT,source,cost,recovery,package,out)
