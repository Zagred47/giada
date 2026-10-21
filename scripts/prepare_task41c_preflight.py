import sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.giada_teacher import task41c_frozen_confirmation as c
spec,_,_,_=c.contract(ROOT)
with zipfile.ZipFile(ROOT/'artifacts/task41c_preflight_payload.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in [c.CONFIG,*spec['sources']]:z.write(ROOT/name,name)
print('Task41c preflight payload ready')
