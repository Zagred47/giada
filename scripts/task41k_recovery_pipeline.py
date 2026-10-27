"""Recover failed evaluation only, preserving the original protocol source hashes."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import task41k_kaggle_pipeline as p

source=p.notebook_source
def notebook_source(*args,**kwargs):
    text=source(*args,**kwargs)
    assert text.count('scripts/run_task41k_learning_alignment.py')==1
    return text.replace('scripts/run_task41k_learning_alignment.py','scripts/run_task41k_evaluation_recovery.py')
p.notebook_source=notebook_source


def recover():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    active=json.loads((p.FOLDER/'kaggle_active_launch.json').read_text())
    state,error=p.status(api,active)
    assert active['mode']=='evaluate' and any(word in state for word in ('error','fail')), (state,error)
    receipt=json.loads((p.DEST/f"v{active['version']}"/'external_backup_receipt.json').read_text())
    assert receipt['kaggle']==active and receipt['technical_failure'] and receipt['all_recovered_tensors_finite']
    archive=Path(receipt['local']);assert p.sha(archive)==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        freeze=json.loads(z.read('selection_freeze.json'))
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p.ROOT,text=True).strip()
    p.write_json(p.FOLDER/'evaluation_recovery_launch.json',dict(original_launch=active,
        revision=revision,input_sha256=receipt['remote']['sha256'],freeze_sha256=freeze['freeze_sha256'],
        checkpoint_sha256=freeze['checkpoint_sha256'],training_updates_performed=0,
        selection_recomputed=False,thresholds_changed=False,backup_verified=True))
    p.publish_resume(api,receipt)
    p.launch(api,revision,'evaluate',None,receipt['remote']['sha256'])


if __name__=='__main__':
    if sys.argv[1:] == ['recover-evaluation']:recover()
    else:p.main()
