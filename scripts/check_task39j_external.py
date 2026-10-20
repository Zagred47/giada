"""Inspect the newest externally verified39j bundle without touching Colab."""
import hashlib,io,json,zipfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'artifacts/task39j_primary_external'
receipt=json.loads((folder/'external_backup_receipt.json').read_text());archive=Path(receipt['local'])
assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['remote']['sha256']
def finite(obj):
    if torch.is_tensor(obj):return bool(torch.isfinite(obj).all())
    if isinstance(obj,dict):return all(finite(v) for v in obj.values())
    if isinstance(obj,(tuple,list)):return all(map(finite,obj))
    return True
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    steps=sorted(int(Path(n).stem.split('_')[1]) for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt'))
    latest=max(steps)
    params=torch.load(io.BytesIO(z.read(f'checkpoint_{latest}.pt')),map_location='cpu',weights_only=True)
    optimizer=torch.load(io.BytesIO(z.read(f'optimizer_{latest}.pt')),map_location='cpu',weights_only=False)
    report=dict(valid=finite(params) and finite(optimizer),step=latest,parameters_finite=finite(params),optimizer_finite=finite(optimizer),
                external_archive=str(archive),sha256=receipt['remote']['sha256'],runtime_independent=True,
                final_present='final_report.json' in z.namelist(),failure_present='failure_report.json' in z.namelist())
path=folder/'health.json';path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
