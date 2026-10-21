"""Console smoke of real captured training before scientific launch."""
import json,os,subprocess,sys,zipfile
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path('/content/giada40b_preflight')
if not root.exists():subprocess.run(['git','clone','--depth','1','--branch','codex/surrogate-validity-audit','https://github.com/Zagred47/giada.git',str(root)],check=True,stdout=subprocess.DEVNULL)
with zipfile.ZipFile('/content/task40b_preflight_payload.zip') as z:
    for name in z.namelist():
        path=root/name
        assert path.resolve().is_relative_to(root.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
output=Path('/content/task40b_gpu_smoke_v5');assert not output.exists()
subprocess.run([sys.executable,'-u',str(root/'scripts/run_task40b_calcium_precision.py'),str(output),'--smoke'],cwd=root,check=True)
report=json.loads((output/'preflight.json').read_text());report['captured_training']=json.loads((output/'training_graph_preflight.json').read_text());report['smoke']=json.loads((output/'final_report.json').read_text())
assert report['valid'] and report['smoke']['parameters_finite']
Path('/content/task40b_gpu_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
