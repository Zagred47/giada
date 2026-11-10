"""GPU-only42 completion; no native phase, mechanism compile or teacher checkout."""
import hashlib,json,os,sys,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task42_axial_exploration as x

def restore_saved_native(archive,out,plan):
    with Path(archive).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['source_sha256']
    assert Path(archive).stat().st_size==plan['source_bytes']
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        for name,sha in plan['members'].items():
            assert Path(name).name==name
            raw=source.read(name);assert hashlib.sha256(raw).hexdigest()==sha,name
            dest=out/('recovered_frozen41r.pt' if name=='frozen41r.pt' else name)
            assert not dest.exists();dest.write_bytes(raw)
    return {name:sha for name,sha in plan['members'].items() if name.startswith('native_')}


def find_verified_archive(input_root,plan,explicit=None):
    candidates=[Path(explicit)] if explicit else sorted(Path(input_root).rglob(plan['source_archive']))
    verified=[]
    for path in candidates:
        if not path.is_file() or path.stat().st_size!=plan['source_bytes']:continue
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if sha==plan['source_sha256']:verified.append(path)
    if not verified:
        inventory=[str(p.relative_to(input_root)) for p in Path(input_root).rglob('*') if p.is_file()][:30]
        raise RuntimeError(f'No verified saved42 archive. Candidate count={len(candidates)}; input inventory={inventory}')
    return verified[0]

def main():
    source,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
    try:
        plan=json.loads((ROOT/'experiments/task42_gpu_recovery_v1.json').read_text())
        original=json.loads((ROOT/'experiments/task42_preregistration_v1.json').read_text())
        assert hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()==plan['original_protocol_canonical_sha256']
        archive=find_verified_archive(Path('/kaggle/input'),plan,os.environ.get('GIADA_TASK42_NATIVE_ARCHIVE'))
        hashes=restore_saved_native(archive,out,plan)
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        prior=x.n.contract(ROOT)[0];assert torch.cuda.is_available() and torch.cuda.get_device_name()==prior['gpu'] and torch.__version__==prior['torch']
        print('[GIADA42 recovery]60 saved native cases restored; native phase disabled',flush=True)
        with torch.inference_mode():report=x.evaluate(torch,ROOT,source,out)
        a=torch.load(out/'recovered_frozen41r.pt',map_location='cpu',weights_only=True);b=torch.load(out/'frozen41r.pt',map_location='cpu',weights_only=True)
        assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==sha for name,sha in hashes.items())
        x.n.p.save_json(out/'recovery_integrity.json',dict(valid=True,native_regenerated=False,native_members_unchanged=hashes,frozen_state_bitwise_unchanged=True,training_updates=0,selection_performed=False,thresholds_changed=False,original_archive_sha256=plan['source_sha256']))
        report.update(native_regenerated=False,recovery_integrity_passed=True)
        x.n.p.save_json(out/'final_report.json',report)
    except BaseException as error:
        # An integrity check after evaluation must not leave a completed report
        # that could be mistaken for a valid recovered result.
        if (out/'final_report.json').exists():
            (out/'final_report.json').rename(out/'partial_evaluation_report.json')
        x.n.p.save_json(out/'failure_report.json',dict(valid=False,technical_failure=True,native_regenerated=False,error=repr(error),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
