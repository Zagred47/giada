"""Seal complete recovered artifacts and verify all checkpoint tensors and ZIP bytes."""
import hashlib,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(folder):
    assert json.loads((folder/'final_report.json').read_text())['valid']
    assert json.loads((folder/'recovered_nonselectable_diagnostics.json').read_text())['valid']
    assert not json.loads((folder/'comparison_with_39h.json').read_text())['baseline_backend_pass_disagreements']
    finite={}
    for path in sorted(folder.glob('*.pt')):
        count=[0]
        def visit(value):
            if torch.is_tensor(value):assert bool(torch.isfinite(value).all()),path;count[0]+=1
            elif isinstance(value,dict):
                for v in value.values():visit(v)
            elif isinstance(value,(tuple,list)):
                for v in value:visit(v)
        visit(torch.load(path,map_location='cpu',weights_only=False));assert count[0]>0
        finite[path.name]=dict(tensors=count[0],finite=True,sha256=digest(path))
    (folder/'all_checkpoints_finite.json').write_text(json.dumps(dict(valid=True,files=finite),indent=2)+'\n',encoding='utf-8')
    code=folder/'recovery_code';code.mkdir(exist_ok=False)
    for name in ('recover_task39i_evaluation.py','task39i_cpu_scanner.py','task39i_recovery_diagnostics.py','compare_task39i_recovery.py','seal_task39i_recovery.py'):
        shutil.copy2(ROOT/'scripts'/name,code/name)
    manifest={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(folder.rglob('*')) if p.is_file()}
    (folder/'complete_recovery_manifest.json').write_text(json.dumps(dict(valid=True,training_updates_performed=0,thresholds_changed=False,files=manifest),indent=2)+'\n',encoding='utf-8')
    target=folder.parent/'giada_task39i_complete_recovered_2026-10-20'
    assert not target.with_suffix('.zip').exists()
    archive=Path(shutil.make_archive(str(target),'zip',folder))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,meta in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256'],name
    receipt=dict(valid=True,archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),zip_crc_valid=True,all_members_sha256_verified=True,all_checkpoint_and_optimizer_tensors_finite=True,checkpoint_files=len(finite),runtime_independent=True,drive_copy_verified=False)
    archive.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
