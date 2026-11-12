"""Evaluation-only recovery: preserve original training/selection and old failures."""
import argparse,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task45d_evaluation_v1'
    before=folder/'memory_receipt.json' if a.result else repo/'experiments/results/task45d_retry_v1/result_memory_receipt.json'
    baseline={} if a.validate_only else json.loads(before.read_text())['verification'];m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=''+table+'-giada-task45d-evaluation-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    if not a.result:
        p=json.loads((repo/'experiments/task45d_evaluation_recovery_v1.json').read_text());put('observations','authorized',Nome='45d complete missing evaluation blocks only',Descrizione=json.dumps(dict(user_authorized=True,source=p,training_performed=False,selection_recomputed=False,thresholds_changed=False,abstraction_changed=False,worker_source='Saved final checkpoint tensors only. Load nomination bytes, skip9complete blocks, run9missing then41stress and registered45d cost. Original3h time-limit output preserved.')))
    else:
        r=json.loads((repo/'artifacts/task45d_evaluation_external/external_backup_receipt.json').read_text());assert r['valid'];path=Path(r['local'])
        art=put('artifacts','output',Nome='45d missing-evaluation recovery output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':r['remote']['sha256'],'Dimensione byte':r['remote']['bytes']});obs=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='45d evaluation '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        compare=(folder/'comparison.json').read_text();obs.append(put('observations','comparison',Nome='All45d arms vs frozen45cJ, paired audit and speed',Descrizione=compare,**{'Artefatti dettagliati':[art]}))
        experiment=json.loads((repo/'experiments/results/task45d_protocol_v1/preregistration_receipt.json').read_text())['records']['experiments-giada-task45d-matrix-v1']
        put('findings','result',Nome='45d recovered evaluation and comparison',Esito='Misto',Esperimenti=[experiment],Osservazioni=obs,Risultato=json.dumps(r['report']),Limitazioni='Recovery never retrained or reselected; no altered thresholds. Completion and descriptive speed are not release qualification. Original41 and45c limits explicit.')
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if a.result else 'memory_receipt.json'),dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45d evaluation schema valid; no writes')
