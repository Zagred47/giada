"""Local-only, fail-closed rotation: retain two SHA-verified full snapshots."""
import hashlib,json,re,shutil
from datetime import datetime,timezone
from pathlib import Path
from .mirror import ROOT

STAMP=re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}Z')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def rotate(directory=None,keep=2,apply=False,progress=print):
    directory=(Path(directory) if directory else ROOT/'drive_exports').resolve(strict=True)
    if keep<2:raise ValueError('At least two full verified backups must survive')
    folders=sorted([p for p in directory.iterdir() if p.is_dir() and STAMP.fullmatch(p.name)],key=lambda p:p.name,reverse=True)
    if len(folders)<keep:raise ValueError('Not enough backups; nothing may be removed')
    survivors=[]
    for folder in folders[:keep]:
        if folder.is_symlink() or folder.resolve().parent!=directory:raise ValueError('Unsafe snapshot path')
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        path=folder/manifest['filename']
        if path.resolve().parent!=folder or path.is_symlink() or path.suffix!='.sqlite':raise ValueError('Unsafe SQLite path')
        assert manifest['integrity_check']=='ok' and not manifest['foreign_key_violations']
        assert path.stat().st_size==manifest['bytes']
        progress('Verifying retained snapshot '+folder.name,flush=True)
        assert sha(path)==manifest['sha256'],'Retained backup SHA mismatch: '+str(path)
        from .payload_backup import verify_reports
        verify_reports(folder,manifest)
        survivors.append(dict(folder=str(folder),filename=path.name,bytes=manifest['bytes'],sha256=manifest['sha256']))
    plan=[]
    for folder in folders[keep:]:
        if folder.is_symlink() or folder.resolve().parent!=directory:raise ValueError('Unsafe old snapshot path')
        # Never follow junctions, symlinks or unexpected directories during deletion.
        paths=list(folder.rglob('*'))
        if any(p.is_symlink() or p.resolve().is_relative_to(folder)==False or getattr(p.lstat(),'st_file_attributes',0)&0x400 for p in paths):
            raise ValueError('Linked snapshot content; manual review required: '+str(folder))
        files=[p for p in paths if p.is_file()]
        manifest=folder/'manifest.json'
        if not manifest.exists():raise ValueError('Unidentified snapshot; manual review required: '+str(folder))
        plan.append(dict(folder=str(folder),bytes=sum(p.stat().st_size for p in files),manifest=json.loads(manifest.read_text(encoding='utf-8'))))
    report=dict(policy='latest-two-full-verified-local-snapshots-v1',created_utc=datetime.now(timezone.utc).isoformat(),apply=apply,retained=survivors,removed=[],planned=plan,bytes_removed=0)
    log=directory/'retention_history';log.mkdir(exist_ok=True)
    receipt=log/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    def save():
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(receipt)
    save()
    if apply:
        for row in plan:
            target=Path(row['folder'])
            assert target.parent==directory and STAMP.fullmatch(target.name) and target not in folders[:keep]
            shutil.rmtree(target)
            report['removed'].append(row['folder']);report['bytes_removed']+=row['bytes'];save()
            progress('Removed superseded snapshot '+target.name,flush=True)
    report['receipt']=str(receipt);save();return report

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    r=rotate(apply=a.apply);print(json.dumps({k:v for k,v in r.items() if k!='planned'},indent=2))
