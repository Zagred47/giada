"""Single private evaluation-only recovery, existing Kaggle outputs read-only."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import task41n_recovery_kaggle as k
from research_memory.mirror import write_json
F=k.ROOT/'experiments/results/task45c_cost_recovery_v1'
k.FOLDER=F;k.DEST=k.ROOT/'artifacts/task45c_cost_recovery_external';k.SLUG='alessandrobelli/giada-task45c-cost-recovery'
k.OUTPUT_PATTERN=r'(kaggle_receipt\.json|task45c_cost_final\.zip|process\.log)$';k.STATUS_LABEL='45c cost recovery'
k.LAUNCH_SCOPE=dict(training_updates_performed=0,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False)
original=k.notebook
def notebook(revision,sha):
    doc=original(revision,sha)
    for cell in doc['cells']:
        if cell['cell_type']=='code':
            lines=[]
            for line in cell['source']:
                line=line.replace('giada_task41n_recovery','giada_task45c_cost_recovery').replace('task41n_recovery_final','task45c_cost_final').replace('run_task41n_recovery.py','run_task45c_cost_recovery.py')
                line=line.replace('str(sources[0]),str(out)',"str(sources[0]),str(saved[0]),str(out)")
                lines.append(line)
                if line.startswith(' with sources[0].open'):
                    lines.append(" saved=list(Path('/kaggle/input').rglob('task45c_final.zip'));assert len(saved)==1\n")
            cell['source']=lines
    doc['cells'][0]['source']=['# GIADA45c — cost-only checkpoint recovery\nNo training, selection, native or accuracy rerun. Existing45c archive hash checked. Original thresholds/precision and cost protocol.']
    return doc
def bundle(revision):
    folder=k.ROOT/'artifacts/task45c_cost_notebook';folder.mkdir(exist_ok=True)
    source=json.loads((k.ROOT/'experiments/task41n_recovery_v1.json').read_text())
    write_json(folder/'notebook.ipynb',notebook(revision,source['source_sha256']))
    write_json(folder/'kernel-metadata.json',dict(id=k.SLUG,title='GIADA45c cost-only recovery',code_file='notebook.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[k.DATASET],kernel_sources=['alessandrobelli/giada-task45c-preserving-kinetics'],competition_sources=[],model_sources=[]))
    return folder
k.bundle=bundle
if __name__=='__main__':
    F.mkdir(exist_ok=True)
    if sys.argv[1]=='prepare':
        receipt=json.loads((k.ROOT/'artifacts/task45c_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid']
        proof=dict(valid=True,source_archive_sha256=receipt['remote']['sha256'],no_local_payload_upload=True,training_performed=False,selection_recomputed=False,thresholds_changed=False,reason='Cost evaluation ran outside inference context. Freeze and all accuracy/stress evaluations already persisted. Recover cost only from immutable saved checkpoints.')
        write_json(F/'preregistration_receipt.json',proof);write_json(F/'input_verified.json',proof)
        write_json(k.ROOT/'notebooks/giada_task45c_cost_recovery.ipynb',notebook('PINNED_BY_CONSOLE_DISPATCH',json.loads((k.ROOT/'experiments/task41n_recovery_v1.json').read_text())['source_sha256']))
    else:k.main()
