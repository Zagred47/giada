"""Recover complete CRC-valid JSON members from truncated local ZIP transfers."""
import json,struct,zlib
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'artifacts/task41g_primary_external'
reports=[]
for src in sorted(D.glob('*.zip.terminal.partial')):
    if not src.exists():continue  # supervisor may atomically finish a transfer
    members=[];dest=D/'salvaged'/src.name
    with src.open('rb') as f:
        while True:
            h=f.read(30)
            if len(h)!=30 or h[:4]!=b'PK\x03\x04':break
            _,version,flags,method,tm,dt,crc,csize,usize,nlen,xlen=struct.unpack('<IHHHHHIIIHH',h)
            name=f.read(nlen).decode('utf-8');extra=f.read(xlen)
            if flags&8:break
            pos=0
            while pos+4<=len(extra):
                tag,length=struct.unpack_from('<HH',extra,pos);body=extra[pos+4:pos+4+length];pos+=4+length
                if tag==1:
                    vals=list(struct.unpack('<'+'Q'*(len(body)//8),body));j=0
                    if usize==0xffffffff:usize=vals[j];j+=1
                    if csize==0xffffffff:csize=vals[j]
            if f.tell()+csize>src.stat().st_size:break
            if not name.endswith('.json'):
                f.seek(csize,1);continue
            payload=f.read(csize)
            raw=zlib.decompress(payload,-15) if method==8 else payload
            assert method in (0,8) and len(raw)==usize and zlib.crc32(raw)&0xffffffff==crc
            target=(dest/name).resolve();assert target.is_relative_to(dest.resolve())
            json.loads(raw)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            members.append({'member':name,'bytes':usize,'crc_valid':True})
    reports.append({'source':str(src),'members':members,'final_archive_verified':False})
    print(src.name,json.dumps([r['member'] for r in members]),flush=True)
(D/'salvage_receipt.json').write_text(json.dumps(reports,indent=2))
