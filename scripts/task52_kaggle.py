import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task51j_kaggle as prior
import task46_kaggle as base
import task52_input as inputs
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task52_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task52_kaggle_external';k.SLUG='alessandrobelli/giada-task52-closed-loop';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task52_final\.zip|process\.log)$';k.STATUS_LABEL='52';k.LAUNCH_SCOPE=dict(exploratory=True,P_frozen=True,thresholds_changed=False,training_updates=0,selection_recomputed=False)
def notebook(revision):
    doc=prior.notebook(revision);proof=json.loads((F/'package.json').read_text())
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('giada_task51j','giada_task52').replace('run_task51j.py','run_task52.py').replace('task51j_final','task52_final').replace('GIADA51j','GIADA52')
            line=line.replace('str(saved[0]),str(out)','str(saved[0]),str(package[0]),str(out)')
            lines.append(line)
            if line.startswith(' with saved[0].open'):
                lines.extend([" package=list(Path('/kaggle/input').rglob('models52.zip.bin'));assert len(package)==1\n",f" with package[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={proof['sha256']!r}\n"])
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA Task52 — frozen compact memory in closed loop\nCanonical64 discrete reference, P and static learned-readout controls, frozen L10 and imposed/oracle/reset diagnostics; blocks and reversal interventions. No training or new model selection. New nativeNEURON audit not performed; existing thresholds unchanged.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task52_notebook';folder.mkdir(exist_ok=True);write_json(folder/'notebook.ipynb',notebook(revision));meta=base.metadata(True,k.SLUG,'GIADA Task52 Closed Loop');meta['dataset_sources'].append(inputs.DATASET);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task52_closed_loop.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json','frozen_input_verified.json'):assert json.loads((F/name).read_text())['valid']
            assert json.loads((F/'package.json').read_text())['sha256']==json.loads((F/'frozen_input_verified.json').read_text())['sha256']
            for name,d in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==d
        k.main()
