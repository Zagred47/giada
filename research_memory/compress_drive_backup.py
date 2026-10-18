"""Single lossless transport archive; original SQLite remains one unchanged member."""
import hashlib
import json
import zipfile
from pathlib import Path
from .mirror import ROOT

def main():
    info=json.loads((ROOT/'drive_exports/latest_local.json').read_text())
    folder=Path(info['folder']);source=folder/info['filename'];archive=source.with_suffix('.sqlite.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_LZMA,allowZip64=True) as z:z.write(source,source.name)
    checksum=hashlib.sha256()
    with zipfile.ZipFile(archive) as z:
        assert z.namelist()==[source.name] and z.testzip() is None
        with z.open(source.name) as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
    assert checksum.hexdigest()==info['sha256']
    transport=dict(filename=archive.name,format='ZIP_LZMA',bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sqlite_member=source.name,
        member_sha256_verified=True,reason='Drive connector100MiB upload limit; no splitting or record removal')
    manifest=json.loads((folder/'manifest.json').read_text());manifest['transport']=transport
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (folder/'README.md').open('a',encoding='utf-8') as f:
        f.write('\n## Trasporto unico senza perdita\n\nIl connettore limita ogni upload a100MiB. Il file originale e\' conservato intero in `'+archive.name+'` (ZIP_LZMA), senza suddivisione. Estrarre con Python zipfile o7-Zip, poi verificare SHA256 del SQLite come sopra. Il checksum dell\'archivio e\' distinto, in manifest.json.\n')
    info['transport']=transport;(ROOT/'drive_exports/latest_local.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(transport),flush=True)

if __name__=='__main__':main()
