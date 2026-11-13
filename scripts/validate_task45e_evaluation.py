"""Real short eval-only resume, frozen weights/selection and skip-block test."""
import hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    rows=[]
    for module in ('tests.test_task45e_evaluation_recovery','tests.test_task45e'):
        result=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True);rows.append(dict(module=module,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));print(result.stderr,flush=True)
    names=['src/giada_teacher/task45e_evaluation_recovery.py','src/giada_teacher/task45d_evaluation_recovery.py','src/giada_teacher/task45e_anchored_repair.py','scripts/run_task45e_evaluation_recovery.py','scripts/task45e_evaluation_kaggle.py','experiments/task45e_evaluation_recovery_v1.json']
    report=dict(valid=all(q['returncode']==0 for q in rows),tests=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},scope='Bounded CPU real checkpoint class and full evaluation-only engine, forbid fit/choose, skip completed block, same freeze bytes and weights, original45e cost closure. No real GPU result claimed.')
    write_json(ROOT/'experiments/results/task45e_evaluation_v1/local_preflight.json',report);assert report['valid']
if __name__=='__main__':main()
