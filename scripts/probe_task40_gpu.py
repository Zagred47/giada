"""Prelaunch CUDA graph and ordering probe; never selects/retrains candidates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/content/giada40_preflight')
if not ROOT.exists():
    subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(ROOT)],check=True,stdout=subprocess.DEVNULL)
for rel in ['src/giada_teacher/task40_causal_reinsertion.py','experiments/task40_preregistration_v1.json']:
    shutil.copy2('/content/'+Path(rel).name,ROOT/rel)
sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task40_causal_reinsertion as t
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
with torch.inference_mode():
    report=t.preflight(torch,ROOT,'cuda')
Path('/content/task40_gpu_preflight.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
