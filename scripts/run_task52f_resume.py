import hashlib,json,sys,traceback,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52f_encoder_credit as q
if __name__=='__main__':
    source,cost,recovery,package,saved,out=map(Path,sys.argv[1:7]);spec=json.loads((ROOT/'experiments/task52f_resume_v1.json').read_text());out.mkdir(exist_ok=False)
    try:
        with saved.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==spec['source_sha256']
        with zipfile.ZipFile(saved) as z:
            assert z.testzip() is None
            assert json.loads(z.read('preregistration.json'))==q.plan(ROOT)
            for name in z.namelist():
                path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts and '\\' not in name
                if name.endswith('/'):continue
                dest=out/name
                if name in ('failure_report.json','process.log'):dest=out/('source_'+name)
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
        q.a.n.p.save_json(out/'resume_request.json',spec)
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        pin=q.a.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.__version__==pin['torch'] and torch.cuda.get_device_name()==pin['gpu']
        checkpoint=torch.load(out/'resumable_training.pt',map_location='cpu',weights_only=True);assert checkpoint['update']==600
        exact=torch.load(out/'checkpoint_600.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v,exact[k]) for k,v in checkpoint['model'].items())
        q.a.n.p.save_json(out/'gpu_preflight.json',dict(valid=True,task='52f-resume',torch=torch.__version__,gpu=torch.cuda.get_device_name()))
        q.run(torch,ROOT,source,cost,recovery,package,out)
    except BaseException as err:
        q.a.n.p.save_json(out/'failure_report.json',dict(valid=False,complete=False,error=repr(err),traceback=traceback.format_exc()));raise
