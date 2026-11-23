"""Prepare verified recovery-only620 continuation; no new selection point."""
import hashlib,io,json,subprocess,sys,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task52g_gradient_preservation as g
F=ROOT/'experiments/results/task52g_resume620_v1';F.mkdir(exist_ok=True)
spec=json.loads((ROOT/'experiments/task52g_resume620_v1.json').read_text())
p=ROOT/'artifacts/task52g_resume_colab_external/recovery_1795424882.zip'
assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['source_sha256']
with zipfile.ZipFile(p) as z:
    assert z.testzip() is None and json.loads(z.read('preregistration.json'))==g.plan(ROOT)
    state=torch.load(io.BytesIO(z.read('recovery_latest.pt')),map_location='cpu',weights_only=True)
    assert state['update']==620 and state['recovery_only'] and 'rng' in state and 'diagnostic_sums' in state
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert {float(v['step']) for v in state['optimizer']['state'].values()}=={620.0}
files=['src/giada_teacher/task52g_gradient_preservation.py','scripts/run_task52g_resume.py','scripts/task52g_console_manager.py','experiments/task52g_resume620_v1.json']
(F/'local_preflight.json').write_text(json.dumps(dict(valid=True,checkpoint_update=620,optimizer_restored=True,rng_saved=True,exact_diagnostic_sums=True,selection_point_added=False,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}),indent=2))
r=json.loads((ROOT/'experiments/results/task52g_resume_v1/dispatch_request.json').read_text())
r.update(session='giada-task52g-resume620-reserve',remote_root='/content/giada_task52g_resume620',local_output='artifacts/task52g_resume620_external',revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),resumed_from=620,resume_spec='experiments/task52g_resume620_v1.json',manager_sha256=hashlib.sha256((ROOT/'scripts/task52g_console_manager.py').read_bytes()).hexdigest())
r['inputs'][-1].update(local=p.relative_to(ROOT).as_posix(),sha256=spec['source_sha256'],bytes=p.stat().st_size)
(F/'dispatch_request.json').write_text(json.dumps(r,indent=2))
print('Verified620 recovery request prepared')
