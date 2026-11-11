"""Record technical cost-only recovery, never overwrite original failure."""
import json,sys,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task45c_cost_recovery_v1'
    result='--result' in sys.argv
    baseline=json.loads(((folder/'memory_receipt.json') if result else (repo/'experiments/results/task45c_protocol_v1/result_memory_receipt.json')).read_text())['verification']
    m=IncrementalBatchMirror();assert m.baseline_hash==baseline['snapshot_hash']
    if result:
        receipt=json.loads((repo/'artifacts/task45c_cost_recovery_external/external_backup_receipt.json').read_text());assert receipt['valid']
        with zipfile.ZipFile(receipt['local']) as z:
            for name in ('final_report.json','failure_report.json','cost.json'):
                if name in z.namelist():m.local_upsert('observations',{'Codice stabile':'observations-giada-task45c-cost-recovery-'+name,'Nome':'45c cost recovery '+name,'Descrizione':z.read(name).decode()})
    else:
        m.local_upsert('observations',{'Codice stabile':'observations-giada-task45c-cost-recovery-v1','Nome':'45c technical benchmark failure and cost-only recovery','Descrizione':json.dumps(dict(cause='Benchmark outside torch.inference_mode; CUDA graph stepper attempted autograd with inference tensors',source_sha256='04a117d26354747cd54092158d509ea1388f6dcd2d8e0fa4dce11e76cd6b8986',training_repeated=False,selection_repeated=False,accuracy_repeated=False,thresholds_changed=False,abstraction_changed=False,original_failure_retained=True,DB_backup_local_only=True))})
    m.commit_batch();v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if result else 'memory_receipt.json'),dict(valid=True,verification=v,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
if __name__=='__main__':main()
