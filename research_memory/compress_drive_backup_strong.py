"""Single-file ZIP_LZMA transport with a larger dictionary; no DB changes."""
import hashlib,json,lzma,struct,zipfile
from pathlib import Path
from .mirror import ROOT

class LargeDictionary(zipfile.LZMACompressor):
    def _init(self):
        props=lzma._encode_filter_properties({'id':lzma.FILTER_LZMA1,'dict_size':128*1024*1024})
        self._comp=lzma.LZMACompressor(lzma.FORMAT_RAW,filters=[lzma._decode_filter_properties(lzma.FILTER_LZMA1,props)])
        return struct.pack('<BBH',9,4,len(props))+props

def main():
    info=json.loads((ROOT/'drive_exports/latest_local.json').read_text());folder=Path(info['folder'])
    source=folder/info['filename'];archive=source.with_suffix('.sqlite.strong.zip');original=zipfile.LZMACompressor
    assert not archive.exists(),'Preserve existing archive'
    zipfile.LZMACompressor=LargeDictionary
    try:
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_LZMA,allowZip64=True) as z:
            with source.open('rb') as src,z.open(source.name,'w',force_zip64=True) as dst:
                done=0;last=-1
                for chunk in iter(lambda:src.read(1024*1024),b''):
                    dst.write(chunk);done+=len(chunk);band=done//(64*1024*1024)
                    if band!=last:
                        print(json.dumps(dict(phase='compress',input_bytes=done,total_bytes=source.stat().st_size)),flush=True);last=band
    finally:zipfile.LZMACompressor=original
    checksum=hashlib.sha256()
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==[source.name] and z.testzip() is None
        with z.open(source.name) as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
    assert checksum.hexdigest()==info['sha256']
    transport=dict(filename=archive.name,format='ZIP_LZMA',dictionary_bytes=128*1024*1024,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sqlite_member=source.name,member_sha256_verified=True,
        reason='100MiB connector limit; larger dictionary only,original SQLite bytes unchanged,no splitting or record removal')
    assert transport['bytes']<=100*1024*1024,transport
    manifest=json.loads((folder/'manifest.json').read_text());manifest['transport']=transport
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (folder/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## Dizionario di compressione esteso\n\nTrasporto finale: '+archive.name+'; ZIP_LZMA con dizionario128MiB. Unico SQLite originale, SHA256 dei byte estratti verificato. Il precedente archivio locale rimane conservato ma non e\' il trasporto finale.\n')
    info['transport']=transport;(ROOT/'drive_exports/latest_local.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(transport),flush=True)
if __name__=='__main__':main()
