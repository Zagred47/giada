"""Verify directly downloaded frozen and60k states independently of the VM."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json
folder=ROOT/'artifacts/task41d_primary_external'
freeze=json.loads((folder/'selection_freeze.json').read_text());body={k:v for k,v in freeze.items() if k!='freeze_sha256'}
assert hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()==freeze['freeze_sha256']
assert hashlib.sha256((folder/'frozen_checkpoint.pt').read_bytes()).hexdigest()==freeze['checkpoint_sha256']
checks={}
for name in ('frozen_checkpoint.pt','checkpoint_60000.pt'):
    path=folder/name;obj=torch.load(path,map_location='cpu',weights_only=False);tensors=[]
    def walk(a):
        if torch.is_tensor(a):tensors.append(a)
        elif isinstance(a,dict):
            for v in a.values():walk(v)
        elif isinstance(a,(list,tuple)):
            for v in a:walk(v)
    walk(obj);assert all(torch.isfinite(a).all() for a in tensors)
    checks[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size,tensors=len(tensors),all_finite=True)
write_json(folder/'freeze_recovery_receipt.json',dict(valid=True,runtime_independent=True,training_updates=60000,checkpoint_hash_valid=True,freeze_hash_valid=True,files=checks,nominees=freeze['nominees'],thresholds_changed=False,scientific_quality_decision_pending=True))
print(json.dumps(checks))
