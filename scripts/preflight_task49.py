"""Hash inputs, test all49 policies on real P, build pinned-dispatch notebook."""
import hashlib,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import task49_kaggle as k
from research_memory.mirror import write_json
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
if __name__=='__main__':
    k.F.mkdir(exist_ok=True);plans=k.prior.prior.plans();checks=[]
    for path,sha in zip(('artifacts/task41n_primary_external/recovery_final.zip','artifacts/task45c_cost_input/checkpoint45c.zip.bin','artifacts/task45d_evaluation_input/evaluation45d.zip.bin'),(plans[0]['source_sha256'],plans[1]['sha256'],plans[2]['sha256'])):
        assert digest(ROOT/path)==sha;checks.append(dict(path=path,sha256=sha,bytes=(ROOT/path).stat().st_size))
    write_json(k.F/'input_verified.json',dict(valid=True,new_payload_upload=False,scope='Local inputs; notebook separately verifies mounted hashes',checks=checks))
    r=subprocess.run([sys.executable,'-m','unittest','tests.test_task49','-v'],cwd=ROOT,env=dict(os.environ),capture_output=True,text=True)
    (k.F/'local_tests.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert r.returncode==0,r.stdout+r.stderr
    files=['src/giada_teacher/task49_causal_initialization.py','scripts/run_task49.py','scripts/task49_kaggle.py','tests/test_task49.py','experiments/task49_preregistration_v1.json']
    write_json(k.F/'local_preflight.json',dict(valid=True,tests=6,real_frozen_P_mini_run=True,local_device='CPU',remote_GPU_preflight_required=True,files_sha256={name:digest(ROOT/name) for name in files}))
    write_json(ROOT/'notebooks/giada_task49_causal_initialization.ipynb',k.notebook('PINNED_BY_CONSOLE_DISPATCH'));print('49 six tests, input hashes and notebook preflight passed',flush=True)
