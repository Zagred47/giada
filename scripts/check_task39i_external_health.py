"""Read-only verification of the latest OFF-VM checkpoint and Adam state."""
import hashlib,io,json,time,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    launch=json.loads((ROOT/'experiments/results/task39i_protocol_v1/launch.json').read_text())
    folder=ROOT/launch['external_destination'];receipt=json.loads((folder/'external_backup_receipt.json').read_text())
    archive=Path(receipt['local']);sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert receipt['valid'] and receipt['runtime_independent'] and sha==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')]
        step=max(int(n.split('_')[1].split('.')[0]) for n in names);finite={}
        for name in (f'checkpoint_{step}.pt',f'optimizer_{step}.pt'):
            tensors=[]
            def visit(value):
                if torch.is_tensor(value):tensors.append(value)
                elif isinstance(value,dict):
                    for v in value.values():visit(v)
                elif isinstance(value,(tuple,list)):
                    for v in value:visit(v)
            visit(torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False))
            finite[name]=dict(tensors=len(tensors),finite=bool(tensors) and all(bool(torch.isfinite(x).all()) for x in tensors));assert finite[name]['finite']
    status=json.loads((folder/'job_status.json').read_text())
    print(json.dumps(dict(valid=True,step=step,numerical_verification=finite,archive_sha256=sha,
        archive_bytes=archive.stat().st_size,backup_age_seconds=round(time.time()-receipt['verified_at']),
        state=status['state'],last_progress=status.get('last_progress'),runtime_independent=True,
        limits='Finite status applies to the verified saved checkpoint,not unsaved live updates.')),flush=True)
if __name__=='__main__':main()
