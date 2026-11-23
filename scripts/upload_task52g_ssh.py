"""Official CLI SSH binary streaming, without copying credentials or chunk files."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];request_path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'experiments/results/task52g_protocol_v1/dispatch_request.json';F=request_path.parent;r=json.loads(request_path.read_text());base=[sys.executable,str(ROOT/'scripts/colab_profile_cli.py'),'--auth=oauth2','ssh','-s',r['session'],'--'];receipts=[]
def send(local,remote,digest):
    code="import sys,hashlib,json,shutil;from pathlib import Path;p=Path("+repr(remote)+");p.parent.mkdir(parents=True,exist_ok=True);f=p.open('wb');shutil.copyfileobj(sys.stdin.buffer,f);f.close();d=hashlib.sha256(p.read_bytes()).hexdigest();assert d=="+repr(digest)+";print(json.dumps(dict(remote=str(p),sha256=d,bytes=p.stat().st_size)))"
    with local.open('rb') as stream:done=subprocess.run(base+['python3','-c',code],stdin=stream,capture_output=True,text=False,timeout=300)
    if done.returncode:raise RuntimeError('SSH input transfer failed; credentials/error URLs suppressed; exit '+str(done.returncode))
    receipt=json.loads(done.stdout.decode().strip());receipts.append(receipt);print(json.dumps(receipt),flush=True)
manager=ROOT/'scripts/task52g_console_manager.py'
send(manager,'/content/task52g_console_manager.py',hashlib.sha256(manager.read_bytes()).hexdigest())
for item in r['inputs']:
    local=ROOT/item['local']
    with local.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    send(local,item['remote'],item['sha256'])
local=request_path;send(local,'/content/task52g_launch.json',hashlib.sha256(local.read_bytes()).hexdigest())
code="import json,torch;assert torch.cuda.is_available();assert torch.cuda.get_device_name()=='Tesla T4';x=torch.ones(16,16,device='cuda');assert torch.isfinite(x@x).all();print(json.dumps(dict(valid=True,gpu=torch.cuda.get_device_name(),torch=torch.__version__)))"
done=subprocess.run(base+['python3','-c',code],capture_output=True,text=True,timeout=120);assert done.returncode==0;gpu=json.loads(done.stdout.strip());(F/'gpu_preflight.json').write_text(json.dumps(gpu,indent=2));(F/'input_verified.json').write_text(json.dumps(dict(valid=True,receipts=receipts,transport='official CLI SSH stdin',same_hashes=True),indent=2));print('All inputs and GPU verified',flush=True)
