"""Independently preserve the completed VM while the full final ZIP downloads."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/task41h_evaluation_primary_external'
request=json.loads((ROOT/'experiments/results/task41h_recovery_v1/dispatch_request.json').read_text())
session='giada-task41h-eval-primary';remote='/content/giada_task41h_'+request['revision'][:12]
os.environ['GIADA_COLAB_PROFILE']='primary'
def cli(*args):
    return subprocess.run(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),*map(str,args)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
driver=None
with (folder/'final_guard_console.log').open('a',encoding='utf-8') as log:
    while True:
        r=cli('download','-s',session,remote+'/job_status.json',folder/'guard_status.json')
        if not r.returncode:
            state=json.loads((folder/'guard_status.json').read_text())
            if state.get('state')=='finished':
                if state.get('worker_exit_code')!=0:raise RuntimeError('Scientific recovery failed; final guard does not infer success')
                if driver is None:
                    live=cli('status','-s',session)
                    if 'Status: IDLE' in live.stdout:
                        driver=subprocess.Popen(['powershell.exe','-NoProfile','-File',str(ROOT/'scripts/colab.ps1'),'exec','-s',session,'-f','scripts/task41h_final_keepalive.py','--timeout','7200'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                        print('Post-evaluation final-transfer guard active',flush=True)
                receipt=folder/'external_backup_receipt.json'
                if receipt.exists():
                    value=json.loads(receipt.read_text())
                    if value.get('valid') and value['remote']['label']=='final':
                        with Path(value['local']).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
                        assert actual==value['remote']['sha256']
                        ack=cli('upload','-s',session,receipt,remote+'/final_guard_ack.json');assert not ack.returncode
                        print('Final archive SHA verified and acknowledged',flush=True);break
        time.sleep(20)
