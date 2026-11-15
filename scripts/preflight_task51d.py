import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import task51d_kaggle as k
from research_memory.mirror import write_json
if __name__=='__main__':
    k.F.mkdir(exist_ok=True);proof=json.loads((ROOT/'experiments/results/task51c_protocol_v1/input_verified.json').read_text())
    for row in proof['checks']:
        with (ROOT/row['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    write_json(k.F/'input_verified.json',proof);r=subprocess.run([sys.executable,'-m','unittest','tests.test_task51d','-v'],cwd=ROOT,capture_output=True,text=True);(k.F/'local_tests.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    names=['src/giada_teacher/task51d_local_relative.py','scripts/run_task51d.py','scripts/task51d_kaggle.py','tests/test_task51d.py','experiments/task51d_preregistration_v1.json']
    write_json(k.F/'local_preflight.json',dict(valid=True,tests=3,real_P_pipeline=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}));write_json(ROOT/'notebooks/giada_task51d_local_relative.ipynb',k.notebook('PINNED_BY_CONSOLE_DISPATCH'));print('51d preflight passed',flush=True)
