"""Reuse four hash-verified private inputs; one version-pinned45e notebook."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task45d_evaluation_kaggle as previous
from research_memory.mirror import write_json
k=previous.k;ROOT=k.ROOT;F=ROOT/'experiments/results/task45e_protocol_v1'
k.FOLDER=F;k.DEST=ROOT/'artifacts/task45e_kaggle_external';k.SLUG='alessandrobelli/giada-task45e-anchored-repair'
k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45e_final\.zip|process\.log)$';k.STATUS_LABEL='45e'
k.LAUNCH_SCOPE=dict(exploratory=True,thresholds_changed=False,abstraction_changed=False,baseline='Frozen45cJ',training_performed=True)
def notebook(revision,sha):
    doc=previous.notebook(revision,sha)
    for cell in doc['cells']:
        if cell['cell_type']!='code':continue
        cell['source']=[line.replace('run_task45d_evaluation_recovery.py','run_task45e.py').replace('task45d_eval_final','task45e_final').replace('giada_task45d_eval_recovery','giada_task45e')
            .replace('str(saved[0]),str(checkpoints[0]),str(out)','str(checkpoints[0]),str(saved[0]),str(out)')
            .replace('training_performed=False,selection_recomputed=False,evaluation_only=True',"training_performed=True,selection_recomputed=(out/'nomination_freeze.json').exists(),evaluation_only=False") for line in cell['source']]
    doc['cells'][0]['source']=['# GIADA45e — anchored ionic repair\nP isolates45dNap transplantation onto frozen45cJ. Nap support×path-loss matrix; Ih rate-only versus joint heads. Same abstraction, thresholds, physics and precision. Opened audits remain exploratory. New FIT/DEV seeds; freeze before paired panels. Exact execution separately measured.']
    return doc
def bundle(revision):
    folder=ROOT/'artifacts/task45e_notebook';folder.mkdir(exist_ok=True)
    write_json(folder/'notebook.ipynb',notebook(revision,previous.old.plans()[0]['source_sha256']))
    meta=previous.old.metadata(folder,True,k.SLUG,'GIADA Task45e anchored repair');meta['dataset_sources'].append(previous.plan()['dataset']);write_json(folder/'kernel-metadata.json',meta);return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1] in ('build','prepare'):
        write_json(ROOT/'notebooks/giada_task45e_anchored_repair.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',previous.old.plans()[0]['source_sha256']))
        if sys.argv[1]=='prepare':
            proof=json.loads((ROOT/'experiments/results/task45d_evaluation_v1/input_verified.json').read_text());assert proof['valid'] and proof['readback_verified']
            assert k.sha(ROOT/'artifacts/task45d_evaluation_input/evaluation45d.zip.bin')==previous.plan()['sha256']
            original=json.loads((ROOT/'experiments/results/task45d_protocol_v1/input_verified.json').read_text());assert original['valid']
            assert json.loads((F/'local_preflight.json').read_text())['valid']
            write_json(F/'input_verified.json',dict(valid=True,reuses_same4private_inputs=True,readback_verified=True,original3=original,compact=proof,worker_rehashes_all_before_fit=True,new_upload=False))
    else:k.main()
