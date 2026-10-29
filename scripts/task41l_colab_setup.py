"""Pin the same CUDA/Torch backend, then test it in a fresh subprocess."""
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','--quiet','--upgrade','torch==2.11.0+cu130','--index-url','https://download.pytorch.org/whl/cu130'],check=True)
code="""import torch,json,hashlib
from pathlib import Path
assert torch.cuda.is_available()
assert torch.cuda.get_device_name()=='Tesla T4'
assert torch.__version__=='2.11.0+cu130'
torch.backends.cuda.matmul.allow_tf32=False
x=torch.arange(4096,device='cuda',dtype=torch.float32).reshape(64,64)/4096
y=x@x.T;torch.cuda.synchronize();assert torch.isfinite(y).all()
p=Path('/content/giada_task41l_gpu.json');p.write_text(json.dumps(dict(valid=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__,finite=True)))
print(p.read_text());print('SHA256 '+hashlib.sha256(p.read_bytes()).hexdigest())
"""
subprocess.run([sys.executable,'-c',code],check=True)
