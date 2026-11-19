import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import task52_kaggle as k
import task52_input
from research_memory.mirror import write_json
if __name__=='__main__':
    k.F.mkdir(exist_ok=True);task52_input.build();proof=json.loads((ROOT/'experiments/results/task51j_protocol_v1/input_verified.json').read_text())
    for row in proof['checks']:
        with (ROOT/row['path']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256']
    write_json(k.F/'input_verified.json',proof);r=subprocess.run([sys.executable,'-m','unittest','tests.test_task52','-v'],cwd=ROOT,capture_output=True,text=True);(k.F/'local_tests.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
    names=['src/giada_teacher/task52_closed_loop.py','scripts/run_task52.py','scripts/task52_kaggle.py','scripts/task52_input.py','tests/test_task52.py','experiments/task52_preregistration_v1.json']
    write_json(k.F/'local_preflight.json',dict(valid=True,tests=3,real_frozen_models=True,files_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}));write_json(ROOT/'notebooks/giada_task52_closed_loop.ipynb',k.notebook('PINNED_BY_CONSOLE_DISPATCH'));print('52 preflight passed',flush=True)
