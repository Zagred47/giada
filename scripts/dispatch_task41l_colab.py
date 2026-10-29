"""Seal the frozen Kaggle training output for console-only Colab evaluation."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import write_json
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    folder=ROOT/'experiments/results/task41l_colab_evaluation_v1';folder.mkdir(exist_ok=True)
    assert not (folder/'dispatch_request.json').exists(),'Inspect existing request; never launch a duplicate'
    backup=json.loads((ROOT/'artifacts/task41l_kaggle_external/external_backup_receipt.json').read_text())
    assert backup['valid'] and backup['all_recovered_tensors_finite'] and backup['report']['training_complete']
    archive=Path(backup['local']);assert sha(archive)==backup['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        freeze=json.loads(z.read('selection_freeze.json'));claimed=freeze.pop('freeze_sha256')
        assert hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()==claimed
        assert hashlib.sha256(z.read('frozen_checkpoint.pt')).hexdigest()==freeze['checkpoint_sha256']
    revision=backup['kaggle']['revision']
    worker=subprocess.check_output(['git','show',revision+':scripts/run_task41l_regression_attribution.py'],cwd=ROOT)
    request=dict(task='41l',revision=revision,profile='primary',session='giada-task41l-eval-primary',evaluation_only=True,selection_recomputed=False,thresholds_changed=False,
        worker_sha256=hashlib.sha256(worker).hexdigest(),manager_sha256=sha(ROOT/'scripts/task41l_colab_manager.py'),archive='/content/task41l_training.zip',archive_sha256=sha(archive),
        frozen_checkpoint_sha256=freeze['checkpoint_sha256'],freeze_sha256=claimed,nominee=freeze['nominee'],GPU='Tesla T4',torch='2.11.0+cu130',inputs=[],DB_backup='local only')
    for path,remote in [(archive,request['archive']),(ROOT/'artifacts/task41b_opened_inputs.zip','/content/task41b_opened_inputs.zip'),(ROOT/'artifacts/task41d_opened41c.zip','/content/task41d_opened41c.zip'),(ROOT/'artifacts/task41l_frozen_input/task41k_frozen_diagnostic.zip.bin','/content/task41k_frozen_diagnostic.zip.bin')]:
        request['inputs'].append(dict(local=str(path),remote=remote,sha256=sha(path),bytes=path.stat().st_size))
    write_json(folder/'dispatch_request.json',request);print(json.dumps(dict(valid=True,revision=revision,selection_recomputed=False,training_updates_authorized=0,inputs=len(request['inputs']))))
if __name__=='__main__':main()
