import json
from pathlib import Path
import torch
assert torch.cuda.is_available()
x=torch.arange(1024,device='cuda',dtype=torch.float32)
assert float((x*x).sum())==357389824.
report=dict(valid=True,GPU=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
Path('/content/task41f_runtime_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
