"""Prefetch final immutable parts concurrently; never talks to the GPU kernel."""
import concurrent.futures,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/task41h_frozen_evaluation_primary_external'
STAGE=D/'prefetch';STAGE.mkdir(exist_ok=True)
os.environ['GIADA_COLAB_PROFILE']='primary'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fetch(part):
    name=Path(part['remote']).name;target=D/name;stage=STAGE/name
    if target.exists() and target.stat().st_size==part['bytes'] and sha(target)==part['sha256']:return dict(name=name,reused=True)
    for attempt in range(3):
        result=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'download','-s','giada-task41h-frozen-resume',part['remote'],str(stage)],cwd=ROOT,capture_output=True,timeout=240)
        if result.returncode==0 and stage.exists() and stage.stat().st_size==part['bytes'] and sha(stage)==part['sha256']:
            stage.replace(target);return dict(name=name,valid=True)
        time.sleep(2)
    raise RuntimeError('Final part transfer failed: '+name)
deadline=time.monotonic()+10800
while True:
    text=(D/'console.log').read_text(errors='replace')
    matches=[line[len('[GIADA recovery] '):] for line in text.splitlines() if line.startswith('[GIADA recovery] {"label": "final"')]
    if matches:receipt=json.loads(matches[-1]);break
    assert time.monotonic()<deadline,'Final manifest not observed'
    time.sleep(10)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    rows=list(pool.map(fetch,receipt['parts']))
(D/'prefetch_receipt.json').write_text(json.dumps(dict(valid=True,sha256=receipt['sha256'],parts=rows),indent=2))
print(json.dumps(dict(valid=True,parts=len(rows),final_parts_prefetched=True)),flush=True)
