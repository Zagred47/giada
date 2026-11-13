"""Actual CPU fit/isolation/full short worker preflight, not a GPU claim."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    tests=[]
    for module in ('tests.test_task45e','tests.test_task45d_cache'):
        r=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True);tests.append(dict(module=module,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));print(r.stderr,flush=True)
    files=['src/giada_teacher/task45e_anchored_repair.py','scripts/run_task45e.py','scripts/task45e_kaggle.py','experiments/task45e_preregistration_v1.json']
    report=dict(valid=all(r['returncode']==0 for r in tests),tests=tests,source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},scope='Actual CPU models, off-target mutation negative control, tiny all-arm refinement, short paired evaluator+execution probes, and notebook compilation. GPU resident cost/input probes still mandatory in worker.')
    write_json(ROOT/'experiments/results/task45e_protocol_v1/local_preflight.json',report);assert report['valid']
if __name__=='__main__':main()
