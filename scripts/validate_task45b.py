"""Persist local test evidence; does not claim a native Linux/CUDA execution."""
import hashlib,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    start=time.time();results=[]
    # Console launcher wrappers intentionally configure shared module globals.
    # Isolate their tests just as each production worker is a fresh process.
    for module in ('tests.test_task45b','tests.test_task45','tests.test_task44'):
        cmd=[sys.executable,'-m','unittest',module,'-v'];result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
        results.append(dict(command=cmd[1:],returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        print(result.stdout,flush=True);print(result.stderr,flush=True)
    receipt=dict(valid=all(r['returncode']==0 for r in results),tests=results,elapsed_seconds=time.time()-start,
        scope='Local CPU functional tests with verified immutable learned source; native-phase interface/calibration mocked, not a Linux NEURON/GPU qualification. Runtime must pass pinned GPU and native floors separately.',thresholds_changed=False)
    receipt['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('src/giada_teacher/task45b_rate_coverage.py','scripts/run_task45b.py','scripts/task45b_kaggle.py','experiments/task45b_preregistration_v1.json')}
    write_json(ROOT/'experiments/results/task45b_protocol_v1/local_preflight.json',receipt)
    assert receipt['valid']
if __name__=='__main__':main()
