"""GPU smoke before scientific dispatch; output is never selected."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/content/giada39j_preflight')
if not ROOT.exists():subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
for name,rel in [('task39j_atomic_relaxation.py','src/giada_teacher/task39j_atomic_relaxation.py'),('task39j_preregistration_v1.json','experiments/task39j_preregistration_v1.json'),('task39i_frozen.pt','experiments/results/task39i_recovered_v1/frozen_checkpoint.pt'),('task39i_freeze.json','experiments/results/task39i_recovered_v1/selection_freeze.json')]:
    path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2('/content/'+name,path)
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39j_atomic_relaxation as j
import torch
torch.set_num_threads(1)
output=Path('/content/giada39j_smoke_'+str(int(time.time())))
start=time.perf_counter();report=j.run(ROOT,output,smoke=True)
report.update(wall_seconds=time.perf_counter()-start,output=str(output),source_sha256=j.p.digest(ROOT/'src/giada_teacher/task39j_atomic_relaxation.py'),gpu=torch.cuda.get_device_name())
spec=json.loads((ROOT/j.CONFIG).read_text());quality=json.loads((ROOT/j.p.CONFIG).read_text())
weights=j.parent_weights(torch,ROOT,spec,'cuda');models=[j.make_model(torch,weights,a,spec['seeds'],'cuda') for a in j.arms()]
with torch.no_grad():
    j.i.old.recurrence_preflight(output,spec,models,torch,'cuda')
    dev,held=j.i.h.development(ROOT,spec,quality)
    normal=json.loads((ROOT/spec['normalizers']).read_text())
    common,_,ranks,panels=j.i.e.checkpoint_scores(torch,[dict(key='preflight',model=models[0])],dev,held,normal,spec,'cuda',quality)
report.update(selection_preflight_valid=all(r is not None for r in ranks),initial_panel=panels,recurrence_preflight=json.loads((output/'recurrence_preflight.json').read_text()))
assert report['selection_preflight_valid']
Path('/content/task39j_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
