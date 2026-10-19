"""Isolated representative residual-attribution preflight, zero training."""
import json,subprocess,sys
from pathlib import Path
root=Path('/content/giada39h_preflight/repo')
code="""import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,sys
from pathlib import Path
import torch
root=Path('/content/giada39h_preflight/repo');sys.path.insert(0,str(root))
from src.giada_teacher import task39h_relaxation_identification as h
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
spec=json.loads((root/h.CONFIG).read_text());quality=json.loads((root/h.p.CONFIG).read_text())
quality['families']={'same_v_history':quality['families']['same_v_history']}
out=Path('/content/giada39h_preflight/residual_probe');out.mkdir(exist_ok=False)
h.d.oracle_audit(torch,root,out,spec,quality,'cuda',parent_loader=h.load_parents,timing_fn=h.f.timing_audit)
rows=h.residual_oracle_audit(torch,root,out,spec,quality,'cuda')
Path('/content/task39h_residual_preflight.json').write_text(json.dumps(dict(valid=True,rows=len(rows),training_updates=0,baseline_reproduced=True,source_sha256=h.p.digest(root/'src/giada_teacher/task39h_relaxation_identification.py')),indent=2))
"""
subprocess.run([sys.executable,'-u','-c',code],check=True)
