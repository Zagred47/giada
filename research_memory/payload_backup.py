"""Portable, independently verified compressed report copies for each DB backup."""
import gzip,hashlib,json,shutil
from pathlib import Path
from .payload_store import active

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def copy_reports(db,folder):
    source=Path(db).parent/'raw_reports';folder=Path(folder);files=[]
    for i,path in enumerate(sorted(source.rglob('*.txt.gz')),1):
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):raise ValueError('Unsafe report source')
        sha=path.name.split('.')[0];h=hashlib.sha256();count=0
        with gzip.open(path,'rb') as f:
            while block:=f.read(1024*1024):h.update(block);count+=len(block)
        if h.hexdigest()!=sha:raise ValueError('Raw report SHA failed before backup')
        rel=path.relative_to(source);target=folder/'raw_reports'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);compressed=digest(path)
        if digest(target)!=compressed:raise ValueError('Compressed backup copy mismatch')
        files.append(dict(path=rel.as_posix(),raw_sha256=sha,raw_bytes=count,compressed_bytes=path.stat().st_size,compressed_sha256=compressed))
        if i%100==0:print('Verified compressed report backup '+str(i),flush=True)
    manifest=folder/'payload_manifest.json';manifest.write_text(json.dumps(dict(format='giada-raw-report-backup-v2',files=files),indent=2))
    return dict(payload_manifest='payload_manifest.json',payload_manifest_sha256=digest(manifest),payload_files=len(files),payload_bytes=sum(r['compressed_bytes'] for r in files),single_file=False,restore_requires='SQLite file and sibling raw_reports directory; use Mirror.connect/open_readonly adapter')

def verify_reports(folder,info):
    if not info.get('payload_manifest'):return
    folder=Path(folder).resolve();manifest=folder/info['payload_manifest']
    if manifest.resolve().parent!=folder or digest(manifest)!=info['payload_manifest_sha256']:raise ValueError('Report manifest missing/corrupt')
    value=json.loads(manifest.read_text())
    if len(value['files'])!=info['payload_files']:raise ValueError('Report count mismatch')
    for row in value['files']:
        p=folder/'raw_reports'/row['path']
        if not p.resolve().is_relative_to(folder/'raw_reports') or p.is_symlink():raise ValueError('Unsafe report backup path')
        if p.stat().st_size!=row['compressed_bytes'] or digest(p)!=row['compressed_sha256']:raise ValueError('Compressed report backup failed: '+row['path'])
