"""Pin a clean committed Task39i worker and console manager before launch."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    request=dict(task='39i',revision=revision)
    for field,path in [('worker_sha256','scripts/run_task39i_coupled_relaxation.py'),('manager_sha256','scripts/task39e_console_manager.py')]:
        committed=subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT)
        assert committed==(ROOT/path).read_bytes(),path
        request[field]=hashlib.sha256(committed).hexdigest()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'Commit tracked changes first'
    target=ROOT/'experiments/results/task39i_protocol_v1/dispatch_request.json'
    assert not target.exists(),'Existing dispatch request; inspect rather than overwrite'
    target.write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(request))
if __name__=='__main__':main()
