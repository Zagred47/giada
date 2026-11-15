"""Task48 console dispatch, existing private inputs only, no new data upload."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task46_kaggle as prior
from research_memory.mirror import write_json
k=prior.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task48_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task48_kaggle_external';k.SLUG='alessandrobelli/giada-task48-family-state-diagnostics';k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task48_final\.zip|process\.log)$';k.STATUS_LABEL='48'
k.LAUNCH_SCOPE=dict(exploratory=True,training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,frozen_candidate='45e/P',task47_passed=False)

def notebook(revision):
    plans=prior.plans();doc=prior.previous.original(revision,plans[0]['source_sha256'])
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        lines=[]
        for line in cell['source']:
            line=line.replace('giada_task41n_recovery','giada_task48').replace('run_task41n_recovery.py','run_task48.py').replace('task41n_recovery_final','task48_final').replace('GIADA41n recovery','GIADA48').replace('threading.Timer(5400','threading.Timer(10800').replace('str(sources[0]),str(out)','str(sources[0]),str(checkpoints[0]),str(saved[0]),str(out)')
            lines.append(line)
            if line.startswith(' with sources[0].open'):
                lines.append(" checkpoints=list(Path('/kaggle/input').rglob('checkpoint45c.zip.bin'));saved=list(Path('/kaggle/input').rglob('evaluation45d.zip.bin'));assert len(checkpoints)==len(saved)==1\n")
                for name,plan in zip(('checkpoints','saved'),plans[1:]):lines.append(f" with {name}[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=={plan['sha256']!r}\n")
        cell['source']=lines
    doc['cells'][0]['source']=['# GIADA Task48 — family state diagnostics (exploratory)\nFrozen P and canonical64/32, shared state/input probes; no training or model selection. Exact algebraic collisions are not assumed reachable. Near-neighbours, PCA and local sensitivity do not prove a globally minimal state. External V/Ca clamp only; Task47 original NO-GO and separate post-hoc event limits retained. Same existing private inputs; all output preserved even on technical failure.']
    return doc

def bundle(revision):
    folder=ROOT/'artifacts/task48_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',notebook(revision));write_json(folder/'kernel-metadata.json',prior.metadata(True,k.SLUG,'GIADA Task48 family state diagnostics'));return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='build':write_json(ROOT/'notebooks/giada_task48_family_state_diagnostics.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH'))
    else:
        if sys.argv[1]=='launch':
            for name in ('local_preflight.json','preregistration_receipt.json','input_verified.json'):assert json.loads((F/name).read_text())['valid'],name
            import hashlib
            for name,sha in json.loads((F/'local_preflight.json').read_text())['files_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
        k.main()
