"""Standalone CUDA preflight; no scientific candidate is retained."""
import os,json,subprocess,sys,tempfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.giada_teacher import task41j_operator_reset as j

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
assert torch.cuda.is_available()
spec,*_=j.contract(ROOT)
with tempfile.TemporaryDirectory() as folder:result=j.preflight(torch,ROOT,Path(folder),'cuda',spec)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert revision==os.environ['GIADA_PREFLIGHT_REVISION']
result.update(revision=revision,gpu=torch.cuda.get_device_name(),torch=torch.__version__,platform='Kaggle',
              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),scientific_training_performed=False)
Path(os.environ['GIADA_PREFLIGHT_OUTPUT']).write_text(json.dumps(result,indent=2))
print('[GIADA Task41j] GPU preflight '+json.dumps(result),flush=True)
