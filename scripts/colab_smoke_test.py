"""Benign console/GPU/file transfer preflight, not a research experiment."""
import hashlib
import json
from pathlib import Path
import sys
import torch

assert torch.cuda.is_available(), 'CUDA not available; do not launch research training'
x=torch.arange(4096,device='cuda',dtype=torch.float32).reshape(64,64)/4096
y=x@x.T
torch.cuda.synchronize()
assert bool(torch.isfinite(y).all())
result=dict(valid=True,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            python=sys.version,finite=True,checksum_probe=float(y.sum().cpu()))
path=Path('/content/giada_console_smoke.json')
path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result),flush=True)
print('SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
