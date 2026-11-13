"""Bounded mathematical/paired-worker tests before any46GPU execution."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    rows=[]
    for module in ('tests.test_task46','tests.test_task44'):
        q=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True);rows.append(dict(module=module,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr));print(q.stderr,flush=True)
    names=['src/giada_teacher/task46_spatial_cost.py','scripts/run_task46.py','scripts/task46_kaggle.py','experiments/task46_preregistration_v1.json']
    report=dict(valid=all(q['returncode']==0 for q in rows),tests=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},scope='CPU real frozenP reconstruction, intervention isolation, axial/full-core equality, short full paired worker and prior44regression tests. NativeLinux/CUDA checks still mandatory; not scientific46results.')
    write_json(ROOT/'experiments/results/task46_protocol_v1/local_preflight.json',report);assert report['valid']
if __name__=='__main__':main()
