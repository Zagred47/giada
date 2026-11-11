"""Keep failed45d attempt intact and register the same-protocol cache retry."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');args=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task45d_retry_v1'
    original=json.loads((repo/'experiments/results/task45d_protocol_v1/preregistration_receipt.json').read_text())
    baseline={} if args.validate_only else json.loads((folder/'memory_receipt.json' if args.result else repo/'experiments/results/task45d_protocol_v1/result_memory_receipt.json').read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task45d-cache-retry-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    experiment=original['records']['experiments-giada-task45d-matrix-v1']
    if not args.result:
        put('observations','cause',Nome='45d failure before first update: inference cache reused by backward',Descrizione=json.dumps(dict(original_archive_sha256='d30239255f15fcb52dbf9dba7c78ebac961f4fe4daf1bf823ffcdfd9b5b9eb80',original_elapsed_seconds=181.4849,training_updates=0,selection_performed=False,fix='Clone resident float32/64 readout constants outside inference mode after GPU preflight; preserve all values bitwise. Immutable historical solver source untouched.',new_runtime_guard='Two scratch Adam steps for both fixed/adaptive features after actual GPU inference preflight; discarded before registered training',thresholds_changed=False,abstraction_changed=False,registered_budget_changed=False,user_authorized_retry=True)))
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task45d-matrix-v1','Stato':'Preregistrato'})
    else:
        r=json.loads((repo/'artifacts/task45d_retry_external/external_backup_receipt.json').read_text());assert r['valid'];path=Path(r['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['remote']['sha256']
        art=put('artifacts','output',Nome='45d cache retry persisted output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':r['remote']['sha256'],'Dimensione byte':r['remote']['bytes']});observations=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):observations.append(put('observations','raw-'+name,Nome='45d retry '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        put('findings','result',Nome='45d same-protocol cache retry outcome',Esito='Misto',Esperimenti=[experiment],Osservazioni=observations,Risultato=json.dumps(r['report']),Limitazioni='Original failed attempt preserved. Technical completion is not quality/speedup qualification; full comparative audit required.')
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task45d-matrix-v1','Stato':'Concluso'})
    for code,arm in original['records'].items():
        if code.startswith('arms-giada-task45d-'):
            name=code.removeprefix('arms-giada-task45d-').removesuffix('-v1');fields=dict(Nome='45d cache retry '+name,Stato=('Interrotta' if r['technical_failure'] else 'Completata') if args.result else 'Pianificata',Braccio=[arm],Seed='17/29/43 unchanged; new technical attempt, not a new scientific protocol')
            if args.result:fields['Artefatti prodotti']=[art]
            put('runs',name,**fields)
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if args.result else 'memory_receipt.json'),dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45d retry SQLite verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45d retry schema valid; no writes')
