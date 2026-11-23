import hashlib,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52g_resume_kaggle as r
ROOT=r.ROOT;F=ROOT/'experiments/results/task52h_protocol_v1';F.mkdir(exist_ok=True)
r.F=F;r.DEST=ROOT/'artifacts/task52h_kaggle_external';r.PREFIX='task52h'
SLUG='alessandrobelli/giada-task52h-ih-preservation'
def build(revision):
    folder=r.build(revision,write_notebook=False);doc=json.loads((folder/'notebook.ipynb').read_text())
    checks=[(Path(i['remote']).name,i['sha256']) for i in json.loads((ROOT/'experiments/results/task52g_resume620_v1/dispatch_request.json').read_text())['inputs'][:4]]
    for cell in doc['cells']:
        if cell['cell_type']=='code':
            cell['source']=[('CHECKS='+repr(checks)+'\n') if line.startswith('CHECKS=') else line.replace('giada_task52g_resume640','giada_task52h').replace('task52g','task52h').replace('GIADA52g','GIADA52h').replace('run_task52h_resume.py','run_task52h.py').replace('all five input hashes','all four input hashes').replace('resumed_from=640','resumed_from=0') for line in cell['source']]
            cell['source']=[line.replace(",GIADA_RESUME_SPEC='experiments/task52h_resume640_v1.json'",'') for line in cell['source']]
            compile(''.join(cell['source']),'52h-notebook','exec')
        else:cell['source']=['# GIADA52h — Ih coverage × worst-current preservation\nFour paired arms, original52d initialization, same10states, thresholds and selection. Fit-only feasibility gate before800updates. No online analytic gates. Cap52j.']
    dest=ROOT/'artifacts/task52h_notebook';dest.mkdir(exist_ok=True);r.save(dest/'notebook.ipynb',doc)
    meta=json.loads((ROOT/'artifacts/task52f_notebook/kernel-metadata.json').read_text());meta.update(id=SLUG,title='GIADA Task52h Ih Preservation');r.save(dest/'kernel-metadata.json',meta);r.save(ROOT/'notebooks/giada_task52h_ih_preservation.ipynb',doc);return dest
def launch():
    assert json.loads((F/'local_preflight.json').read_text())['valid'] and json.loads((F/'preregistration_receipt.json').read_text())['valid']
    for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert not (F/'kaggle_launch.json').exists() and not (F/'launch_pending.json').exists()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();folder=build(rev);a=r.api();r.save(F/'launch_pending.json',dict(revision=rev,at=time.time()))
    result=a.kernels_push(str(folder),timeout='43200',acc='NvidiaTeslaT4');assert not result.error,result.error
    r.save(F/'kaggle_launch.json',dict(ref=result.ref,version=result.version_number,url=result.url,revision=rev,thresholds_changed=False));(F/'launch_pending.json').unlink();print(result.url,flush=True)
if __name__=='__main__':
    if sys.argv[1]=='launch':launch()
    elif sys.argv[1]=='supervise':r.supervise()
    else:build('PINNED_BY_CONSOLE_DISPATCH')
