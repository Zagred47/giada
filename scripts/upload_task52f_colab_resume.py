"""Bounded small console transfers; verify full hashes remotely before launch."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'experiments/results/task52f_resume_colab_v1'
req=json.loads((F/'dispatch_request.json').read_text());session=req['session'];parts=[]
transport=ROOT/'artifacts/task52f_colab_transport';transport.mkdir(exist_ok=True)
def cli(*args):
    for attempt in range(3):
        r=subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),*map(str,args)],cwd=ROOT,capture_output=True,text=True,timeout=150)
        if r.returncode==0:return
        print(f'Transfer retry {attempt+1}: exit {r.returncode}; credential-bearing error suppressed',flush=True)
        time.sleep(2)
    raise RuntimeError('Console transfer failed after three attempts')
for item in req['inputs']:
    entries=[]
    with (ROOT/item['local']).open('rb') as f:
        i=0
        while block:=f.read(4*1024*1024):
            local=transport/(Path(item['remote']).name+f'.part{i:04d}')
            local.write_bytes(block);remote='/content/'+local.name
            cli('upload','-s',session,local,remote)
            entries.append(dict(remote=remote,sha256=hashlib.sha256(block).hexdigest()))
            local.unlink();i+=1
            print(f"[52f transfer] {Path(item['remote']).name} part {i}",flush=True)
    parts.append(dict(remote=item['remote'],sha256=item['sha256'],parts=entries))
manifest=F/'transport_manifest.json';manifest.write_text(json.dumps(parts,indent=2))
cli('upload','-s',session,manifest,'/content/task52f_transport_manifest.json')
cli('upload','-s',session,F/'dispatch_request.json','/content/task52f_resume_request.json')
cli('upload','-s',session,ROOT/'scripts/task52f_colab_resume_manager.py','/content/task52f_colab_resume_manager.py')
cli('exec','-s',session,'-f','scripts/assemble_task52f_colab_resume.py')
cli('download','-s',session,'/content/task52f_input_verified.json',F/'input_verified.json')
assert json.loads((F/'input_verified.json').read_text())['valid']
print('[52f transfer] All five inputs verified remotely',flush=True)
