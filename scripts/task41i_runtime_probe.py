"""Read-only CUDA smoke test; no scientific training."""
import json
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Task41i requires CUDA"
x = torch.arange(1024, device="cuda", dtype=torch.float32)
assert float((x * 2).sum()) == 1047552.
torch.cuda.synchronize()
report = dict(valid=True, cuda=True, gpu=torch.cuda.get_device_name(),
              torch=torch.__version__, cuda_version=torch.version.cuda,
              scientific_training_performed=False)
Path('/content/task41i_runtime_probe.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
