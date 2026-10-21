"""Small source-only upload bundle for the actual CUDA smoke."""
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=['src/giada_teacher/task40b_calcium_precision.py','scripts/run_task40b_calcium_precision.py',
       'experiments/task40b_preregistration_v1.json']
files+=['experiments/results/task40_audited_v1/'+n for n in ('audit.json','attribution_audit.json','synthesis.json')]
target=ROOT/'artifacts/task40b_preflight_payload.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(ROOT/name,name)
print(str(target),target.stat().st_size)
