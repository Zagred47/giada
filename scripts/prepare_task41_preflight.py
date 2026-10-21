import zipfile,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41_independent_confirmation as c
paths=['src/giada_teacher/task41_independent_confirmation.py','src/giada_teacher/task33_observable_synaptic_feedback.py',
    c.CONFIG,'experiments/results/task41_protocol_v1/old_native_phase_preflight.json','scripts/run_task41_independent_confirmation.py']
target=ROOT/'artifacts/task41_preflight_payload.zip';assert not target.exists()
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name in paths:z.write(ROOT/name,name)
print(json.dumps(dict(path=str(target),sha256=c.p.digest(target),bytes=target.stat().st_size)))
