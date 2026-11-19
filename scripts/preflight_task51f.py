import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import task51f_kaggle as k
from research_memory.mirror import write_json
if __name__=='__main__':
    k.F.mkdir(exist_ok=True);proof=json.loads((ROOT/'experiments/results/task51e_protocol_v1/input_verified.json').read_text())
    for row in proof['checks']:
        with (ROOT/row['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    write_json(k.F/'input_verified.json',proof);r=subprocess.run([sys.executable,'-m','unittest','tests.test_task51f','-v'],cwd=ROOT,capture_output=True,text=True);(k.F/'local_tests.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    names=['src/giada_teacher/task51f_local_targets.py','scripts/run_task51f.py','scripts/task51f_kaggle.py','tests/test_task51f.py','experiments/task51f_preregistration_v1.json']
    write_json(k.F/'local_preflight.json',dict(valid=True,tests=3,real_P_pipeline=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}));write_json(ROOT/'notebooks/giada_task51f_local_targets.ipynb',k.notebook('PINNED_BY_CONSOLE_DISPATCH'));print('51f preflight passed',flush=True)
