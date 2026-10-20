"""Lossless versioned JSON export, preserving the operative readable snapshot."""
import gzip
import lzma
import json
import os
import shutil
import uuid
from pathlib import Path
from .packed_database import digest


def pack_snapshot(path,compression='gzip'):
    if compression not in ('gzip','xz'):raise ValueError('Unsupported snapshot compression')
    path=Path(path);archive=path.with_suffix('.json.gz' if compression=='gzip' else '.json.xz');manifest=path.with_suffix('.json.manifest.json')
    temp=path.parent/('snapshot-'+uuid.uuid4().hex+'.tmp')
    try:
        with path.open('rb') as source,temp.open('xb') as target:
            if compression=='gzip':
                with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as gz:shutil.copyfileobj(source,gz)
            else:
                filters=[dict(id=lzma.FILTER_LZMA2,dict_size=128*1024*1024,mode=lzma.MODE_FAST,mf=lzma.MF_HC4,nice_len=32,depth=16)]
                with lzma.LZMAFile(target,'w',format=lzma.FORMAT_XZ,filters=filters) as xz:shutil.copyfileobj(source,xz)
        opener=gzip.open if compression=='gzip' else lzma.open
        with opener(temp,'rb') as stream:
            import hashlib
            checksum=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):checksum.update(chunk)
        if checksum.hexdigest()!=digest(path):raise ValueError('JSON export changed during packing')
        info=dict(format='json-'+compression+'-lossless-v1',json_sha256=digest(path),json_bytes=path.stat().st_size,compressed_sha256=digest(temp),compressed_bytes=temp.stat().st_size)
        os.replace(temp,archive)
        manifest.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        return info
    finally:temp.unlink(missing_ok=True)


def restore_snapshot_if_missing(path):
    path=Path(path)
    if path.exists():return False
    manifest=path.with_suffix('.json.manifest.json')
    if not manifest.exists():
        if path.with_suffix('.json.gz').exists() or path.with_suffix('.json.xz').exists():raise ValueError('Missing JSON archive manifest')
        return False
    info=json.loads(manifest.read_text())
    formats={'json-gzip-lossless-v1':('.json.gz',gzip.open),'json-xz-lossless-v1':('.json.xz',lzma.open)}
    if info['format'] not in formats:raise ValueError('Unknown JSON archive format')
    suffix,opener=formats[info['format']];archive=path.with_suffix(suffix)
    if digest(archive)!=info['compressed_sha256']:raise ValueError('JSON archive hash mismatch')
    temp=path.parent/('snapshot-'+uuid.uuid4().hex+'.tmp')
    try:
        with opener(archive,'rb') as source,temp.open('xb') as target:shutil.copyfileobj(source,target)
        if digest(temp)!=info['json_sha256'] or temp.stat().st_size!=info['json_bytes']:raise ValueError('JSON content hash mismatch')
        try:os.link(temp,path)
        except FileExistsError:return False
        return True
    finally:temp.unlink(missing_ok=True)


if __name__=='__main__':
    from .mirror import ROOT
    print(json.dumps(pack_snapshot(ROOT/'data/airtable_snapshot.json',compression='xz')))
