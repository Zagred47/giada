"""Native, lossless, single-member SQLite transport with stream-hash verification."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from .mirror import ROOT
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',default='C:/Program Files/7-Zip/7z.exe');args=parser.parse_args()
    info=json.loads((ROOT/'drive_exports/latest_local.json').read_text());folder=Path(info['folder']);source=folder/info['filename']
    archive=source.with_suffix('.sqlite.7z');assert not archive.exists(),'Preserve existing archive'
    subprocess.run([args.exe,'a','-t7z','-mx=7','-m0=LZMA2:d=256m','-mmt=2','-bsp0',str(archive),source.name],cwd=folder,check=True)
    subprocess.run([args.exe,'t','-bsp0',str(archive)],check=True)
    proc=subprocess.Popen([args.exe,'x','-so',str(archive),source.name],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    h=hashlib.sha256();count=0
    for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):h.update(chunk);count+=len(chunk)
    proc.wait();assert proc.returncode==0 and count==info['bytes'] and h.hexdigest()==info['sha256']
    transport=dict(filename=archive.name,format='7z_LZMA2',dictionary_bytes=256*1024*1024,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),sqlite_member=source.name,member_sha256_verified=True,
        reason='Single complete original SQLite; native compression only; no rows removed or database split')
    assert transport['bytes']<=100*1024*1024,transport
    manifest=json.loads((folder/'manifest.json').read_text());manifest['transport']=transport
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (folder/'README.md').open('a',encoding='utf-8') as f:f.write('\n## Trasporto lossless\n\nArchivio7z con unico SQLite originale. Estrarre con7-Zip e verificare lo SHA256 del DB sopra indicato. Nessun record rimosso.\n')
    info['transport']=transport;(ROOT/'drive_exports/latest_local.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(transport),flush=True)
if __name__=='__main__':main()
