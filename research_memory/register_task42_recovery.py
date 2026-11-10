"""Technical42 correction and GPU-only recovery, preserving the original attempt."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task42_recovery_v1';folder.mkdir(exist_ok=True)
    plan=json.loads((repo/'experiments/task42_gpu_recovery_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task42_protocol_v1/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task42-recovery-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','gpu-only',Nome='Task42 GPU-only evaluation recovery; saved native traces',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Complete original42 evaluations after seed-axis adapter bug; no native replay, training, selection or changed quality gates.'})
    if not args.result:
        art=put('artifacts','input',Nome='Original failed42 full archive, verified saved native traces',Tipo='Report',Versione='v1',Percorso='artifacts/task42_kaggle_external/task42_final.zip',**{'SHA-256':plan['source_sha256'],'Dimensione byte':plan['source_bytes']})
        with zipfile.ZipFile(repo/'artifacts/task42_kaggle_external/task42_final.zip') as z:
            floor=json.loads(z.read('native_floor.json'))
            failure=json.loads(z.read('failure_report.json'))
        obs=put('observations','technical-failure',Nome='42 first attempt: completed native phase, adapter failure before paired GPU results',Descrizione=json.dumps(dict(failure=failure,native_floor=floor,cause=plan['adapter_fix'],scientific_no_go=False)),**{'Artefatti dettagliati':[art]})
        finding=put('findings','cause',Nome='42 seed axis was flattened into site axis',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='GPU imposed reference frame(3,12,2) was flattened to72sites and expanded to3seeds instead of preserving3x24. All60 native cases and15/15 passive checks retained. No valid paired candidate verdict from that attempt.',Limitazioni='Technical instrumentation failure, not model failure. Saved weights finite. Original gates remain.')
        protocol=put('protocols','gpu',Nome='42 GPU-only recovery unchanged original contract',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Original42 quality criteria and per-case floor eligibility; no new thresholds. Exact raw native member and logical checkpoint equality.','Regole di arresto':plan['stop']})
        for name in ('formula','41r_imposed','41r_autonomous'):
            arm=put('arms',name,Nome='42 recovery '+name,Ruolo='Baseline' if name=='formula' else 'Trattamento',Protocollo=[protocol])
            put('runs',name,Nome='42 recovery '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen; original42 protocols and geometry')
        put('decisions','recover',Nome='User authorized GPU-only42 recovery, no native repetition',Esito='Continuare',Motivazione=plan['scope'],Risultati=[finding],**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task42_recovery_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent']
        path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='42 recovery full persisted GPU output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='42 recovery raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='42 recovered GPU paired results',Descrizione=json.dumps(receipt['report']),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='42 recovery result requires audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(receipt['report']),Limitazioni='Exploratory;41 failures preserved, original42 thresholds, no automatic43 or speedup qualification.')
        for name in ('formula','41r_imposed','41r_autonomous'):put('runs',name,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task42 GPU recovery memory verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task42 recovery schema preview valid; no writes')
