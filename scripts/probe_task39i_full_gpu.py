"""Exercise the actual complete worker preparation and sampling path on CUDA."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,shutil,subprocess,sys,traceback
from pathlib import Path
BASE=Path('/content/giada39i_full_preflight');ROOT=BASE/'repo';BASE.mkdir(exist_ok=True)
if not ROOT.exists():subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
for src,dst in [('task39i_coupled_relaxation.py','src/giada_teacher/task39i_coupled_relaxation.py'),
                ('task39d_current_timing.py','src/giada_teacher/task39d_current_timing.py'),
                ('task39i_preregistration_v1.json','experiments/task39i_preregistration_v1.json')]:shutil.copy2('/content/'+src,ROOT/dst)
sys.path.insert(0,str(ROOT))
from src.giada_teacher import task39i_coupled_relaxation as i
output=BASE/'trial/artifacts'
try:
    report=i.run(ROOT,output,preflight_only=True)
    report['parity']=json.loads((output/'optimizer_preflight.json').read_text())
    report['sk_transfer']=json.loads((output/'sk_transfer_preflight.json').read_text())
    Path('/content/task39i_gpu_preflight.json').write_text(json.dumps(report,indent=2))
    print('[GIADA Task39i full preflight] passed;all numerical updates rolled back',flush=True)
except BaseException:
    Path('/content/task39i_full_preflight_failure.json').write_text(json.dumps(dict(valid=False,technical_failure=True,traceback=traceback.format_exc()),indent=2))
    raise
