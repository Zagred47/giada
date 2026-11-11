"""Incremental dispatch-only test after extending read-only monitoring window."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    file=ROOT/'experiments/results/task45d_protocol_v1/local_preflight.json';p=json.loads(file.read_text());assert p['valid']
    for name,digest in p['source_sha256'].items():
        if name!='scripts/task45d_kaggle.py':assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    r=subprocess.run([sys.executable,'-m','unittest','tests.test_task45d.Task45d.test_notebook_inputs_and_timers','-v'],cwd=ROOT,text=True,capture_output=True);print(r.stderr);assert r.returncode==0
    p['dispatch_extension_test']=dict(valid=True,scope='Only provider timeout/monitoring window; no worker or protocol change',stderr=r.stderr)
    p['source_sha256']['scripts/task45d_kaggle.py']=hashlib.sha256((ROOT/'scripts/task45d_kaggle.py').read_bytes()).hexdigest();write_json(file,p)
if __name__=='__main__':main()
