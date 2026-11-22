import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task52f_kaggle as prior
import task52f_resume_input as inputs
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task52f_resume_v1'
prior.F=F;k.FOLDER=F;k.DEST=ROOT/'artifacts/task52f_resume_external';k.SLUG='alessandrobelli/giada-task52f-resume600';k.LAUNCH_SCOPE=dict(resumed_from=600,target_update=800,retraining_from_zero=False,thresholds_changed=False,abstraction_changed=False)
def notebook(revision):
    doc=prior.notebook(revision);digest=json.loads((ROOT/'experiments/task52f_resume_v1.json').read_text())['source_sha256']
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('giada_task52f','giada_task52f_resume').replace('run_task52f.py','run_task52f_resume.py').replace("str(package[0]),str(out)","str(package[0]),str(resumes[0]),str(out)")
            if line.startswith(' env='):
                lines.extend([" resumes=list(Path('/kaggle/input').rglob('resume52f.zip.bin'));assert len(resumes)==1\n",f" with resumes[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={digest!r}\n"])
            lines.append(line)
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA52f recovery from600\nRestore optimizer and sampling stream, finish601-800 and evaluations. No new warmup, no threshold or architecture change.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task52f_resume_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));meta=prior.base.metadata(True,k.SLUG,'GIADA52f Resume600');meta['dataset_sources'] += [prior.inputs.DATASET,inputs.DATASET];write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task52f_resume.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    elif sys.argv[1]=='supervise':prior.supervise()
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','recovery_input_verified.json'):assert json.loads((F/name).read_text())['valid']
            for name,digest in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
        k.main()
