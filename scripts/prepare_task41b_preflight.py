import sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41b_local_precision as c
files=[c.CONFIG,'src/giada_teacher/task41b_local_precision.py','src/giada_teacher/task41b_evaluation.py',
    'scripts/run_task41b_local_precision.py','experiments/results/task41_audited_v1/audit.json','experiments/results/task41_audited_v1/synthesis.json']
with zipfile.ZipFile(ROOT/'artifacts/task41b_preflight_payload.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in files:z.write(ROOT/name,name)
print('Task41b source preflight payload ready')
