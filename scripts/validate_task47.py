"""Real CPU frozen-weight worker tests plus inherited environment evidence."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def main():
    folder=ROOT/'experiments/results/task47_protocol_v1';folder.mkdir(exist_ok=True)
    rows=[]
    for module in ('tests.test_task47','tests.test_task46'):
        result=subprocess.run([sys.executable,'-m','unittest',module,'-v'],cwd=ROOT,capture_output=True,text=True)
        rows.append(dict(module=module,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr));print(result.stderr,flush=True)
    names=['src/giada_teacher/task47_independent_confirmation.py','scripts/run_task47.py','scripts/task47_kaggle.py','experiments/task47_preregistration_v1.json']
    proof=dict(valid=all(r['returncode']==0 for r in rows),tests=rows,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},scope='Real CPU frozenP short worker, paired intervention identity, disjoint geometry/input design, unchanged46regressions. Not GPU/native47scientific results.')
    write_json(folder/'local_preflight.json',proof);assert proof['valid']
    for name in ('input_verified.json','native_environment_verified.json'):
        source=ROOT/'experiments/results/task46_retry_v1'/name;old=json.loads(source.read_text());assert old['valid']
        write_json(folder/name,dict(valid=True,inherited_verified_evidence=old,source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Identical mounted inputs and pinned environment; no new payload.47native floors and CUDA preflight are regenerated before candidate judgments.'))
if __name__=='__main__':main()
