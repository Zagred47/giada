"""Record technical cost-only recovery, never overwrite original failure."""
import json,sys,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;v2='--v2' in sys.argv;folder=repo/('experiments/results/task45c_cost_recovery_v2' if v2 else 'experiments/results/task45c_cost_recovery_v1')
    result='--result' in sys.argv
    basefile=(folder/'memory_receipt.json') if result else (repo/('experiments/results/task45c_cost_recovery_v1/result_memory_receipt.json' if v2 else 'experiments/results/task45c_protocol_v1/result_memory_receipt.json'))
    baseline=json.loads(basefile.read_text())['verification']
    m=IncrementalBatchMirror();assert m.baseline_hash==baseline['snapshot_hash']
    if result:
        receipt=json.loads((repo/('artifacts/task45c_cost_recovery_v2_external/external_backup_receipt.json' if v2 else 'artifacts/task45c_cost_recovery_external/external_backup_receipt.json')).read_text());assert receipt['valid']
        with zipfile.ZipFile(receipt['local']) as z:
            for name in ('final_report.json','failure_report.json','cost.json'):
                if name in z.namelist():m.local_upsert('observations',{'Codice stabile':'observations-giada-task45c-cost-recovery-'+('v2-' if v2 else '')+name,'Nome':'45c cost recovery '+name,'Descrizione':z.read(name).decode()})
    elif v2:
        transport=json.loads((repo/'experiments/task45c_cost_transport_v2.json').read_text())
        m.local_upsert('observations',{'Codice stabile':'observations-giada-task45c-cost-recovery-transport-v2','Nome':'45c cost recovery explicit private checkpoint upload authorization','Descrizione':json.dumps(dict(user_authorized_private_checkpoint_upload=True,authorization='Sì, autorizzo il caricamento dell’archivio con i checkpoint sul mio account Kaggle come Dataset privato. Completa solo il benchmark, senza rifare training o selezione né cambiare le soglie.',transport=transport,previous_failure='Kaggle did not mount failed notebook output; worker never started',training_performed=False,selection_recomputed=False,thresholds_changed=False))})
    else:
        m.local_upsert('observations',{'Codice stabile':'observations-giada-task45c-cost-recovery-v1','Nome':'45c technical benchmark failure and cost-only recovery','Descrizione':json.dumps(dict(cause='Benchmark outside torch.inference_mode; CUDA graph stepper attempted autograd with inference tensors',source_sha256='04a117d26354747cd54092158d509ea1388f6dcd2d8e0fa4dce11e76cd6b8986',training_repeated=False,selection_repeated=False,accuracy_repeated=False,thresholds_changed=False,abstraction_changed=False,original_failure_retained=True,DB_backup_local_only=True))})
    m.commit_batch();v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if result else 'memory_receipt.json'),dict(valid=True,verification=v,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
if __name__=='__main__':main()
