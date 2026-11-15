"""Bounded local integration test, exact mounted-input hashes, notebook build."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import task48_kaggle as k
from research_memory.mirror import write_json

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    k.F.mkdir(exist_ok=True)
    paths=[ROOT/'artifacts/task41n_primary_external/recovery_final.zip',ROOT/'artifacts/task45c_cost_input/checkpoint45c.zip.bin',ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin']
    plans=k.prior.plans();expected=[plans[0]['source_sha256'],plans[1]['sha256'],plans[2]['sha256']];checks=[]
    for path,want in zip(paths,expected):
        assert digest(path)==want;checks.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=want,bytes=path.stat().st_size))
    write_json(k.F/'input_verified.json',dict(valid=True,scope='Local exact bytes; mounted bytes independently hash-checked by notebook before worker',new_payload_upload=False,checks=checks))
    result=subprocess.run([sys.executable,'-m','unittest','tests.test_task48','-v'],cwd=ROOT,env=dict(os.environ),capture_output=True,text=True)
    (k.F/'local_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8');assert result.returncode==0,result.stdout+result.stderr
    files=['src/giada_teacher/task48_state_minimality.py','scripts/run_task48.py','scripts/task48_kaggle.py','tests/test_task48.py','experiments/task48_preregistration_v1.json']
    write_json(k.F/'local_preflight.json',dict(valid=True,tests=5,real_frozen_P_mini_run=True,local_device='CPU',remote_GPU_preflight_required=True,files_sha256={name:digest(ROOT/name) for name in files}))
    write_json(ROOT/'notebooks/giada_task48_family_state_diagnostics.ipynb',k.notebook('PINNED_BY_CONSOLE_DISPATCH'))
    print('48 local integration, hashes and notebook preflight passed',flush=True)
if __name__=='__main__':main()
