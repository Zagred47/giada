from pathlib import Path
import hashlib
import json
import torch
p=Path('/content/task39e_external_smoke.txt');p.write_bytes(b'GIADA external checkpoint transport preflight v1\n')
assert torch.cuda.is_available()
print(json.dumps(dict(gpu=torch.cuda.get_device_name(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=True)),flush=True)
