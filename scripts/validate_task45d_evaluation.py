"""Bounded evaluation-only resume tests, with source-bound receipt."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    rows=[]
    for module in ('tests.test_task45d_evaluation_recovery','tests.test_task45d'):
        r=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True);rows.append(dict(module=module,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout);print(r.stderr,flush=True)
    names=['src/giada_teacher/task45d_evaluation_recovery.py','src/giada_teacher/task45d_learnability_execution.py','scripts/run_task45d_evaluation_recovery.py','scripts/task45d_evaluation_kaggle.py','experiments/task45d_evaluation_recovery_v1.json','experiments/task45d_preregistration_v1.json']
    result=dict(valid=all(r['returncode']==0 for r in rows),tests=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},scope='CPU short complete resume with real checkpoint class, mocked native/cost controls; skip existing block and forbid training/nomination calls. No scientific result or CUDA benchmark claim.')
    write_json(ROOT/'experiments/results/task45d_evaluation_v1/local_preflight.json',result);assert result['valid']
if __name__=='__main__':main()
