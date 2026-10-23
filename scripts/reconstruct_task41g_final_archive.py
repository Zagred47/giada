"""Attempt exact final ZIP recovery from its prefix plus unchanged earlier members.

Never replaces a source. Only equality with the remotely recorded whole-file
SHA can qualify the output as the recovered final archive.
"""
import hashlib,json,struct,zipfile
from pathlib import Path
R=Path(__file__).resolve().parents[1];D=R/'artifacts/task41g_primary_external'

def local_header(f):
    start=f.tell();h=f.read(30)
    if len(h)!=30 or h[:4]!=b'PK\x03\x04':return None
    v=list(struct.unpack('<IHHHHHIIIHH',h));name=f.read(v[-2]);extra=f.read(v[-1]);cs,us=v[7],v[8]
    assert not v[2]&8,'Data descriptor unsupported'
    i=0
    while i+4<=len(extra):
        tag,n=struct.unpack_from('<HH',extra,i);b=extra[i+4:i+4+n];i+=4+n
        if tag==1:
            vals=list(struct.unpack('<'+'Q'*(len(b)//8),b));j=0
            if us==0xffffffff:us=vals[j];j+=1
            if cs==0xffffffff:cs=vals[j]
    return dict(start=start,header=h+name+extra,values=v,name=name.decode('utf-8'),csize=cs,usize=us,data=f.tell(),end=f.tell()+cs)

def main():
    receipt=json.loads((D/'external_backup_receipt.json').read_text());old=Path(receipt['local'])
    expected=json.loads((D/'remote_recovery_manifest.json').read_text());assert expected['label']=='final'
    prefix=D/'recovery_final.zip.terminal.partial';target=D/'recovery_final_reconstructed.zip'
    rows=[]
    with prefix.open('rb') as f:
        while f.tell()<prefix.stat().st_size:
            row=local_header(f)
            if row is None:break
            rows.append(row)
            if row['end']>prefix.stat().st_size:break
            f.seek(row['end'])
    last=rows[-1]
    with zipfile.ZipFile(old) as z:
        infos={i.filename:i for i in z.infolist()}
        assert last['name'] in infos
        template=infos['development_gpu.json']
        # Preserve the original walk order after the cut member; all newly
        # generated files are in the already recovered top-level prefix.
        oldnames=list(infos);tail=oldnames[oldnames.index(last['name'])+1:]
        assert not set(tail)&{r['name'] for r in rows}
        with old.open('rb') as src,prefix.open('rb') as pre,target.open('wb') as out:
            emitted=[]
            for row in rows:
                info=infos.get(row['name'],template)
                if row['end']<=prefix.stat().st_size:
                    stream=pre;stream.seek(row['start']);count=row['end']-row['start']
                else:
                    src.seek(infos[row['name']].header_offset);original=local_header(src)
                    assert original['header']==row['header']
                    available=prefix.stat().st_size-row['data'];pre.seek(row['data']);src.seek(original['data'])
                    while available:
                        n=min(1024*1024,available);assert pre.read(n)==src.read(n);available-=n
                    stream=src;stream.seek(original['start']);count=original['end']-original['start']
                offset=out.tell()
                while count:
                    b=stream.read(min(count,8*1024*1024));assert b;out.write(b);count-=len(b)
                emitted.append((row,info,offset))
            for name in tail:
                info=infos[name];src.seek(info.header_offset);row=local_header(src);assert row['name']==name
                offset=out.tell();src.seek(row['start']);count=row['end']-row['start']
                while count:
                    b=src.read(min(count,8*1024*1024));assert b;out.write(b);count-=len(b)
                emitted.append((row,info,offset))
            start=out.tell()
            for row,info,offset in emitted:
                v=row['values'];name=row['name'].encode('utf-8');assert row['csize']<2**32 and row['usize']<2**32 and offset<2**32
                # zipfile writes ZIP64 local headers for streamed files but
                # ordinary central entries when actual sizes fit32bits.
                version=info.create_version if row['name'] in infos else template.create_version
                extract=info.extract_version if row['name'] in infos else template.extract_version
                central=struct.pack('<IHHHHHHIIIHHHHHII',0x02014b50,version+(info.create_system<<8),extract,v[2],v[3],v[4],v[5],v[6],row['csize'],row['usize'],len(name),len(info.extra),len(info.comment),info.volume,info.internal_attr,info.external_attr,offset)
                out.write(central+name+info.extra+info.comment)
            size=out.tell()-start
            out.write(struct.pack('<IHHHHIIH',0x06054b50,0,0,len(emitted),len(emitted),size,start,0))
    with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    exact=digest==expected['sha256'] and target.stat().st_size==expected['bytes']
    with zipfile.ZipFile(target) as z:crc=z.testzip() is None
    value=dict(valid=exact and crc,local=str(target),sha256=digest,bytes=target.stat().st_size,expected=expected,byte_identical_to_remote=exact,crc_all_members_valid=crc,runtime_independent=True,members=len(emitted),method='Exact local ZIP prefix and byte-matching remainder member; unchanged earlier members; regenerated central directory; whole-file SHA decisive')
    (D/'final_reconstruction_receipt.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(value))
if __name__=='__main__':main()
