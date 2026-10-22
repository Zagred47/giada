"""Assemble a content-identical final backup from verified base and delta ZIPs."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/task41e_reserve_external'
manifest=json.loads((folder/'final_delta_manifest.json').read_text())
old=json.loads((folder/'external_backup_receipt.json').read_text())
base=folder/Path(old['local'].replace('\\','/')).name
delta=folder/'recovery_delta.zip'
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(base)==old['remote']['sha256']
assert delta.stat().st_size==manifest['delta']['bytes']
assert sha(delta)==manifest['delta']['sha256']
assert manifest['status']['state']=='finished'
assert manifest['status']['worker_exit_code']==0
target=folder/'giada_task41e_result_recovered.zip'
temp=target.with_suffix('.zip.partial')
with zipfile.ZipFile(base) as b,zipfile.ZipFile(delta) as d:
    bm={i.filename:i for i in b.infolist()};dm={i.filename:i for i in d.infolist()}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as out:
        for item in manifest['manifest']:
            name=item['name'];source=d if name in dm else b
            info=source.getinfo(name)
            assert info.file_size==item['size'] and info.CRC==item['crc'],name
            with source.open(name) as src,out.open(name,'w',force_zip64=True) as dst:
                shutil.copyfileobj(src,dst,1024*1024)
with zipfile.ZipFile(temp) as z:
    assert z.testzip() is None
    actual={i.filename:(i.file_size,i.CRC) for i in z.infolist()}
    assert actual=={r['name']:(r['size'],r['crc']) for r in manifest['manifest']}
    import torch
    def finite(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(x) for x in value.values())
        if isinstance(value,(tuple,list)):return all(finite(x) for x in value)
        return True
    params={name:finite(torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False))
            for name in z.namelist() if name.endswith('.pt')}
    assert all(params.values())
temp.replace(target)
receipt=dict(valid=True,runtime_independent=True,local=str(target),bytes=target.stat().st_size,
             sha256=sha(target),crc_all_members_valid=True,final_members_verified=len(actual),
             parameters_finite=params,finished_status=manifest['status'],
             reconstruction='Same complete final member contents; recompressed ZIP, not byte-identical to remote ZIP.',
             sources={'base_sha256':old['remote']['sha256'],'delta_sha256':manifest['delta']['sha256']})
(folder/'final_recovery_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
