"""Verified single-file local SQLite backup when external upload is deferred."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from .mirror import ROOT,write_json
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--raw-sqlite',action='store_true')
    parser.add_argument('--scope',default='Complete authoritative local memory at snapshot time; inspect the latest per-task audit receipts for scientific scope.')
    args=parser.parse_args()
    info=json.loads((ROOT/'drive_exports/latest_local.json').read_text());folder=Path(info['folder'])
    if args.raw_sqlite:
        path=folder/info['filename'];h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==info['sha256'] and path.stat().st_size==info['bytes']
        assert info['integrity_check']=='ok' and not info['foreign_key_violations']
        receipt=dict(format='sqlite-local-backup-v1',snapshot_utc=info['snapshot_utc'],sqlite_sha256=info['sha256'],
            sqlite_bytes=info['bytes'],semantic_records=info['semantic_records'],local_sqlite=str(path),local_archive=str(path),
            archive_sha256=info['sha256'],archive_bytes=info['bytes'],archive_format='SQLite3',
            integrity_check='ok',foreign_key_errors=[],valid=True,single_file=True,
            scope=args.scope,
            drive_updated=False,upload_deferred_by_user=True,airtable_accessed=False)
        write_json(ROOT/'data/local_backup_receipt.json',receipt)
        write_json(ROOT/'data/research_memory.sqlite.manifest.json',receipt)
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest.update(scope=args.scope,drive_updated=False,upload_deferred_by_user=True,
                        post_upload_verification='Not requested: local backup only.')
        write_json(folder/'manifest.json',manifest)
        print(json.dumps(receipt),flush=True);return
    archive=folder/(Path(info['filename']).with_suffix('.sqlite.7z').name)
    exe='C:/Program Files/7-Zip/7z.exe'
    listing=subprocess.check_output([exe,'l','-slt',str(archive)],text=True).split('----------',1)[1]
    assert listing.count('\nPath = ')==1 and ('Path = '+info['filename']) in listing
    proc=subprocess.Popen([exe,'x','-so',str(archive),info['filename']],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    h=hashlib.sha256();count=0
    for chunk in iter(lambda:proc.stdout.read(1024*1024),b''):h.update(chunk);count+=len(chunk)
    proc.wait();assert proc.returncode==0 and count==info['bytes'] and h.hexdigest()==info['sha256']
    transport=dict(filename=archive.name,format='7z_LZMA2',dictionary_bytes=512*1024*1024,
        bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        sqlite_member=info['filename'],member_sha256_verified=True)
    previous=json.loads((ROOT/'data/drive_backup_receipt.json').read_text())
    receipt=dict(format='sqlite-local-backup-v1',snapshot_utc=info['snapshot_utc'],sqlite_sha256=info['sha256'],
        sqlite_bytes=info['bytes'],semantic_records=info['semantic_records'],
        local_sqlite=str(folder/info['filename']),local_archive=str(archive),archive_sha256=transport['sha256'],archive_bytes=transport['bytes'],
        archive_format=transport['format'],sqlite_member=info['filename'],single_original_sqlite_member_sha256_verified=True,
        integrity_check=info['integrity_check'],foreign_key_errors=info['foreign_key_violations'],valid=True,single_file=True,
        scope='Complete Task40b audit and all prior memory;B6 nominated for Task41 preparation only;thresholds unchanged;no independent confirmation or speedup claim.',
        drive_updated=False,upload_deferred_by_user=True,
        reason='User explicitly chose local storage after connector100MiB cap. No further upload attempts.',
        last_verified_drive_backup=previous,airtable_accessed=False)
    assert receipt['integrity_check']=='ok' and not receipt['foreign_key_errors']
    info['transport']=transport;write_json(ROOT/'drive_exports/latest_local.json',info)
    manifest=json.loads((folder/'manifest.json').read_text());manifest.update(transport=transport,upload_deferred_by_user=True)
    write_json(folder/'manifest.json',manifest)
    write_json(ROOT/'data/local_backup_receipt.json',receipt)
    write_json(ROOT/'data/research_memory.sqlite.manifest.json',receipt)
    write_json(ROOT/'data/drive_backup_pending.json',dict(status='deferred_by_user',local_backup_valid=True,
        local_archive=str(archive),sqlite_sha256=info['sha256'],records=info['semantic_records'],drive_updated=False,
        last_verified_drive_url=previous['drive_url'],scope=receipt['scope'],airtable_accessed=False))
    print(json.dumps(dict(valid=True,records=info['semantic_records'],local_archive=str(archive),drive_updated=False)),flush=True)
if __name__=='__main__':main()
