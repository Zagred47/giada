"""Alternative single SQLite ZIP transport; bounded fast LZMA match search.

Preserves the slow archival copy and verifies every decompressed byte. Does not
overwrite latest_local.json, so concurrent archival compression cannot race it.
"""
import hashlib,json,lzma,struct,zipfile
from pathlib import Path
from .mirror import ROOT

class FastDictionary(zipfile.LZMACompressor):
    def _init(self):
        filters={'id':lzma.FILTER_LZMA1,'dict_size':128*1024*1024,
                 'mode':lzma.MODE_FAST,'mf':lzma.MF_HC4,'nice_len':32,'depth':16}
        props=lzma._encode_filter_properties(filters)
        self._comp=lzma.LZMACompressor(lzma.FORMAT_RAW,filters=[filters])
        return struct.pack('<BBH',9,4,len(props))+props

def main():
    info=json.loads((ROOT/'drive_exports/latest_local.json').read_text());folder=Path(info['folder'])
    source=folder/info['filename'];archive=source.with_suffix('.sqlite.fast.zip');assert not archive.exists()
    original=zipfile.LZMACompressor;zipfile.LZMACompressor=FastDictionary
    try:
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_LZMA,allowZip64=True) as z:z.write(source,source.name)
    finally:zipfile.LZMACompressor=original
    h=hashlib.sha256()
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==[source.name]
        with z.open(source.name) as stream:
            for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
    assert h.hexdigest()==info['sha256']
    transport=dict(filename=archive.name,format='ZIP_LZMA',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   sqlite_member=source.name,member_sha256_verified=True,compression='HC4 modeFAST dict128MiB; original bytes unchanged')
    info['transport']=transport;info['connector_size_valid']=transport['bytes']<=100*1024*1024
    (folder/'fast_manifest.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8');print(json.dumps(transport),flush=True)
    assert info['connector_size_valid'],'Keep original stronger compression for upload'

if __name__=='__main__':main()
