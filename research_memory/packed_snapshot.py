"""Lossless versioned JSON export, preserving the operative readable snapshot."""
import gzip
import json
import os
import shutil
import uuid
from pathlib import Path
from .packed_database import digest


def pack_snapshot(path):
    path=Path(path);archive=path.with_suffix('.json.gz');manifest=path.with_suffix('.json.manifest.json')
    temp=path.parent/('snapshot-'+uuid.uuid4().hex+'.tmp')
    try:
        with path.open('rb') as source,temp.open('xb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as gz:shutil.copyfileobj(source,gz)
        with gzip.open(temp,'rb') as stream:
            import hashlib
            checksum=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
        if checksum.hexdigest()!=digest(path):raise ValueError('JSON export changed during packing')
        info=dict(format='json-gzip-lossless-v1',json_sha256=digest(path),json_bytes=path.stat().st_size,compressed_sha256=digest(temp),compressed_bytes=temp.stat().st_size)
        os.replace(temp,archive)
        manifest.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        return info
    finally:temp.unlink(missing_ok=True)


def restore_snapshot_if_missing(path):
    path=Path(path)
    if path.exists():return False
    archive=path.with_suffix('.json.gz');manifest=path.with_suffix('.json.manifest.json')
    if not archive.exists() and not manifest.exists():return False
    info=json.loads(manifest.read_text())
    if info['format']!='json-gzip-lossless-v1' or digest(archive)!=info['compressed_sha256']:raise ValueError('JSON archive hash mismatch')
    temp=path.parent/('snapshot-'+uuid.uuid4().hex+'.tmp')
    try:
        with gzip.open(archive,'rb') as source,temp.open('xb') as target:shutil.copyfileobj(source,target)
        if digest(temp)!=info['json_sha256'] or temp.stat().st_size!=info['json_bytes']:raise ValueError('JSON content hash mismatch')
        try:os.link(temp,path)
        except FileExistsError:return False
        return True
    finally:temp.unlink(missing_ok=True)


if __name__=='__main__':
    from .mirror import ROOT
    print(json.dumps(pack_snapshot(ROOT/'data/airtable_snapshot.json')))
