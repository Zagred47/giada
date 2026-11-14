import hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
if __name__=='__main__':
    r=subprocess.run([sys.executable,'-m','unittest','tests.test_task47_cahva_audit','-v'],cwd=ROOT,capture_output=True,text=True)
    names=['scripts/audit_task47_cahva.py','scripts/finalize_task47_cahva_audit.py','src/giada_teacher/task37_ionic_playground.py','tests/test_task47_cahva_audit.py','experiments/task47_preregistration_v1.json']
    proof=dict(valid=r.returncode==0,stdout=r.stdout,stderr=r.stderr,source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},training_performed=False,thresholds_changed=False)
    write_json(ROOT/'experiments/results/task47_cahva_audit_v1/tests.json',proof);print(r.stderr);assert proof['valid']
