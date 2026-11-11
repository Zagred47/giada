"""Local bounded CPU fixtures, no scientific45d execution or CUDA claim."""
import hashlib,subprocess,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    records=[];start=time.time()
    for name in ('tests.test_task45d','tests.test_task45c','tests.test_task45c_design'):
        r=subprocess.run([sys.executable,'-m','unittest',name,'-v'],cwd=ROOT,text=True,capture_output=True);records.append(dict(module=name,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout,flush=True);print(r.stderr,flush=True)
    files=['src/giada_teacher/task45d_learnability_execution.py','src/giada_teacher/task45c_preserving_kinetics.py','scripts/run_task45d.py','scripts/task45d_kaggle.py','experiments/task45d_preregistration_v1.json']
    report=dict(valid=all(r['returncode']==0 for r in records),elapsed_seconds=time.time()-start,tests=records,source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},scope='CPU actual checkpoint training/materialization, tiny complete fit and mocked-native short full evaluator. Not real scientific45d, not GPU timing. Runtime paired-source/stepper GPU preflight still mandatory.')
    write_json(ROOT/'experiments/results/task45d_protocol_v1/local_preflight.json',report);assert report['valid']
if __name__=='__main__':main()
