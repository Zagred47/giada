"""Same scientific protocol; regression tests for cache lifetime and retry."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    rows=[]
    for module in ('tests.test_task45d_cache','tests.test_task45d','tests.test_task45c','tests.test_task45c_design'):
        r=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True);rows.append(dict(module=module,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout);print(r.stderr,flush=True)
    names=['src/giada_teacher/task45d_learnability_execution.py','src/giada_teacher/task40_causal_reinsertion.py','scripts/run_task45d.py','scripts/task45d_retry_kaggle.py','experiments/task45d_preregistration_v1.json']
    report=dict(valid=all(r['returncode']==0 for r in rows),tests=rows,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},scope='Same-process cache regression and actual CPU backward/Adam, plus prior mathematical and pipeline tests. GPU transition guard still runs before registered training. Historical source hashes checked by inherited contracts; no changed protocol.')
    write_json(ROOT/'experiments/results/task45d_retry_v1/local_preflight.json',report);assert report['valid']
    sys.path.insert(0,str(ROOT/'scripts'));import task45d_retry_kaggle as k
    for cell in k.old.notebook('a'*40,k.old.plans()[0]['source_sha256'])['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'retry-notebook','exec')
if __name__=='__main__':main()
