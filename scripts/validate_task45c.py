"""Bounded local CPU tests and source-bound receipt, not CUDA qualification."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    start=time.time();records=[]
    for module in ('tests.test_task45c','tests.test_task45c_design','tests.test_task45b'):
        cmd=[sys.executable,'-m','unittest',module,'-v'];r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        records.append(dict(command=cmd[1:],returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stdout,flush=True);print(r.stderr,flush=True)
    files=('src/giada_teacher/task45c_preserving_kinetics.py','scripts/run_task45c.py','scripts/task45c_kaggle.py','experiments/task45c_preregistration_v1.json','experiments/task45c_transport_v1.json')
    receipt=dict(valid=all(r['returncode']==0 for r in records),tests=records,elapsed_seconds=time.time()-start,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        scope='Local CPU mathematical/functional tests, including tiny synthetic end-to-end evaluator and actual immutable f011 reconstruction. Source/native methods mocked in tiny test; not a real CUDA or scientific45c result. Runtime GPU and full native/reference floor checks still mandatory.')
    write_json(ROOT/'experiments/results/task45c_protocol_v1/local_preflight.json',receipt);assert receipt['valid']
if __name__=='__main__':main()
