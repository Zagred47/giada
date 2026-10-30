"""Authorized conservative cleanup; no model, result, or live DB deletion."""
import hashlib,json,re,shutil,sqlite3,time,zipfile
from pathlib import Path
from research_memory.backup_retention import rotate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/storage_cleanup_20261030.json'

def main():
    before=shutil.disk_usage(ROOT).free;start=time.time()
    report=dict(start_free_bytes=before,removed_transport=[],removed_temporaries=[],skipped=[],phase='verifying_backup_survivors')
    def save():
        report['elapsed_seconds']=time.time()-start
        temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(OUT)
    save()
    report['backups']=rotate(apply=True);report['phase']='transport';save()
    artifacts=(ROOT/'artifacts').resolve();groups={}
    for p in artifacts.rglob('*'):
        if p.is_file() and re.fullmatch(r'.+\.zip\.part\d{4}',p.name):
            base=Path(re.sub(r'\.part\d{4}$','',str(p)));groups.setdefault(base,[]).append(p)
    for number,(archive,parts) in enumerate(groups.items(),1):
        parts=sorted(parts)
        try:
            assert archive.resolve().is_relative_to(artifacts) and all(p.resolve().is_relative_to(artifacts) and not p.is_symlink() for p in parts)
            assert archive.is_file() and not archive.is_symlink(),'Missing complete archive'
            assert [p.name for p in parts]==[archive.name+f'.part{k:04d}' for k in range(len(parts))],'Incomplete or non-contiguous parts'
            stamps={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in [archive,*parts]}
            assert sum(stamps[p][0] for p in parts)==stamps[archive][0],'Part sizes do not reconstruct archive'
            digest=hashlib.sha256()
            with archive.open('rb') as full:
                for part in parts:
                    with part.open('rb') as split:
                        while chunk:=split.read(4*1024*1024):
                            assert full.read(len(chunk))==chunk,'Part bytes differ from preserved archive'
                            digest.update(chunk)
                assert not full.read(1)
            with zipfile.ZipFile(archive) as z:assert z.testzip() is None,'Archive CRC failed'
            assert all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in stamps.items()),'File changed during verification'
            row=dict(archive=str(archive.relative_to(ROOT)),sha256=digest.hexdigest(),bytes=sum(stamps[p][0] for p in parts),parts=[str(p.relative_to(ROOT)) for p in parts],removed=[])
            report['removed_transport'].append(row);save()
            for part in parts:
                assert part.resolve().parent==archive.resolve().parent and re.fullmatch(re.escape(archive.name)+r'\.part\d{4}',part.name)
                part.unlink();row['removed'].append(str(part.relative_to(ROOT)));save()
            print(f'Transport {number}/{len(groups)}: removed redundant parts for {archive.parent.name}/{archive.name}',flush=True)
        except (AssertionError,OSError,zipfile.BadZipFile,RuntimeError) as e:
            report['skipped'].append(dict(archive=str(archive.relative_to(ROOT)),reason=str(e)));save()
            print(f'Transport {number}/{len(groups)}: preserved (not verified duplicate)',flush=True)
    # Only unfinished exports older than the preserved, complete current JSON.
    data=ROOT/'research_memory/data';snapshot=data/'airtable_snapshot.json'
    for p in data.glob('airtable_snapshot.json*.tmp'):
        assert p.resolve().parent==data.resolve() and not p.is_symlink()
        if p.stat().st_mtime_ns<snapshot.stat().st_mtime_ns:
            size=p.stat().st_size;p.unlink();report['removed_temporaries'].append(dict(path=str(p.relative_to(ROOT)),bytes=size));save()
    with sqlite3.connect((data/'research_memory.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as c:
        report['live_DB_snapshot_hash']=c.execute("select value from _mirror_meta where key='snapshot_hash'").fetchone()[0]
        report['live_DB_records']=c.execute('select count(*) from v_records').fetchone()[0]
    report['end_free_bytes']=shutil.disk_usage(ROOT).free
    report['free_bytes_gained']=report['end_free_bytes']-before
    report['bytes_removed']=report['backups']['bytes_removed']+sum(r['bytes'] for r in report['removed_transport'] if len(r['removed'])==len(r['parts']))+sum(r['bytes'] for r in report['removed_temporaries'])
    report['phase']='complete';save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('removed_transport','skipped','backups')},indent=2),flush=True)

if __name__=='__main__':main()
