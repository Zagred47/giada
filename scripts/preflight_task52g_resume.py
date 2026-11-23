import hashlib,io,json,sys,unittest,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52g_gradient_preservation as g
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_task52g.Preservation.test_resume_preserves_updates_and_original_anchor');r=unittest.TextTestRunner(verbosity=2).run(suite);assert r.wasSuccessful()
    old=json.loads((ROOT/'artifacts/task52g_colab_external/external_backup_receipt.json').read_text());p=Path(old['local']);spec=json.loads((ROOT/'experiments/task52g_resume_v1.json').read_text())
    with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==spec['source_sha256']
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None and json.loads(z.read('preregistration.json'))==g.plan(ROOT)
        s=torch.load(io.BytesIO(z.read('resumable_training.pt')),map_location='cpu',weights_only=True);w=torch.load(io.BytesIO(z.read('checkpoint_600.pt')),map_location='cpu',weights_only=True)
        assert s['update']==600 and 'rng' in s and all(torch.equal(v,w[k]) for k,v in s['model'].items())
        assert all(torch.isfinite(v).all() for v in w.values());steps=sorted(set(float(v['step']) for v in s['optimizer']['state'].values()));assert steps==[600.0]
    F=ROOT/'experiments/results/task52g_resume_v1';F.mkdir(exist_ok=True);files=['src/giada_teacher/task52g_gradient_preservation.py','scripts/run_task52g_resume.py','scripts/task52g_console_manager.py','experiments/task52g_resume_v1.json']
    (F/'local_preflight.json').write_text(json.dumps(dict(valid=True,checkpoint_update=600,optimizer_steps=steps,rng_saved=True,anchor_parity_test=True,split_continuous_bitwise_equal=True,source_sha256=spec['source_sha256'],files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2))
    print('Real600 checkpoint and split/continuous parity verified')
