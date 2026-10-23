"""Package frozen selection and CRC-verified cached metrics, never training."""
import hashlib,json,struct,zlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_evaluation_primary_external'
dest=ROOT/'artifacts/task41h_frozen_resume';dest.mkdir(exist_ok=True)
receipt=json.loads((D/'external_backup_receipt.json').read_text())
source=Path(receipt['local'])
with source.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
members={};provenance={}
def retain(name,raw,origin):
    if not (name.endswith('.json') or name=='frozen_checkpoint.pt'):return
    if name.endswith('.json'):json.loads(raw)
    members[name]=raw;provenance[name]=dict(source=origin,sha256=hashlib.sha256(raw).hexdigest())
with zipfile.ZipFile(source) as z:
    for n in z.namelist():
        if n.endswith('.json') or n=='frozen_checkpoint.pt':retain(n,z.read(n),source.name)
# A prefix may contain newer complete JSONs even without a central directory.
parts=sorted(D.glob('recovery_saved_causal_results_9fd4af37ef6240229467b22b6cf0f3a2.zip.part[0-9][0-9][0-9][0-9]'))
prefix=dest/'latest_prefix.zip.partial'
with prefix.open('wb') as out:
    for k,part in enumerate(parts):
        assert part.name.endswith(f'.part{k:04d}')
        with part.open('rb') as f:
            while block:=f.read(4*1024*1024):out.write(block)
salvaged=[]
with prefix.open('rb') as f:
    while True:
        head=f.read(30)
        if len(head)!=30 or head[:4]!=b'PK\x03\x04':break
        _,version,flags,method,tm,dt,crc,cs,us,nl,xl=struct.unpack('<IHHHHHIIIHH',head)
        name=f.read(nl).decode();extra=f.read(xl)
        if flags&8:break
        pos=0
        while pos+4<=len(extra):
            tag,length=struct.unpack_from('<HH',extra,pos);body=extra[pos+4:pos+4+length];pos+=4+length
            if tag==1:
                vals=list(struct.unpack('<'+'Q'*(len(body)//8),body));j=0
                if us==0xffffffff:us=vals[j];j+=1
                if cs==0xffffffff:cs=vals[j]
        if f.tell()+cs>prefix.stat().st_size:break
        if not name.endswith('.json'):f.seek(cs,1);continue
        payload=f.read(cs);raw=zlib.decompress(payload,-15) if method==8 else payload
        assert method in (0,8) and len(raw)==us and zlib.crc32(raw)&0xffffffff==crc
        retain(name,raw,prefix.name);salvaged.append(name)
freeze=json.loads(members['selection_freeze.json']);claimed=freeze.pop('freeze_sha256')
assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
assert hashlib.sha256(members['frozen_checkpoint.pt']).hexdigest()==freeze['checkpoint_sha256']
assert json.loads(members['selection_parity.json'])['all60_rank_vectors_bitwise_equal']
seal=json.loads((ROOT/'experiments/results/task41h_recovery_v1/input_seal.json').read_text())
manifest=dict(valid=True,source_archive_sha256=receipt['remote']['sha256'],members=provenance,
    original_seal=seal,freeze_sha256=claimed,training_updates_performed=0,thresholds_changed=False)
members['frozen_resume_manifest.json']=json.dumps(manifest,indent=2).encode()
archive=dest/'frozen_resume_input.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name,raw in members.items():z.writestr(name,raw)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
result=dict(archive=str(archive),sha256=digest,bytes=archive.stat().st_size,salvaged=salvaged,
    cached_reports={k:len(json.loads(v).get('reports',{})) for k,v in members.items() if k.endswith('causal_results.json')})
(dest/'package_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
