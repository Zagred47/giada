"""Register verified42 transport separately from the two earlier technical attempts."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task42_transport_recovery_v1'
    plan=json.loads((repo/'experiments/task42_verified_transport_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task42_recovery_v1/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task42-transport-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','gpu-only',Nome='Task42 GPU-only recovery with verified explicit input transport',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Finish original paired42 GPU evaluations with saved native input; transport instrumentation only, not new scientific experiment.'})
    if not args.result:
        observations=[]
        for label,value in [('transport',plan['transfer_readback']),('CPU_mount',plan['actual_cpu_mount']),('prior_failure',json.loads((repo/'artifacts/task42_recovery_external/external_backup_receipt.json').read_text()))]:
            observations.append(put('observations',label,Nome='Task42 verified transport '+label,Descrizione=json.dumps(value)))
        finding=put('findings','technical-cause',Nome='42 prior recovery stopped at input discovery, before GPU evaluation',Esito='Misto',Esperimenti=[exp],Osservazioni=observations,Risultato=plan['cause']+' '+plan['correction'],Limitazioni='No new model quality result; native generation, weights and acceptance thresholds unchanged.')
        protocol=put('protocols','verified',Nome='42 GPU recovery input readback plus CPU mounted preflight',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Original42 metrics/gates and recovered native integrity; no threshold or candidate change.','Regole di arresto':plan['stop']})
        for name in ('formula','41r_imposed','41r_autonomous'):
            arm=put('arms',name,Nome='42 verified-input '+name,Ruolo='Baseline' if name=='formula' else 'Trattamento',Protocollo=[protocol])
            put('runs',name,Nome='42 verified-input '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen; original60 native cases')
        put('decisions','retry-GPU-only',Nome='Authorize GPU-only42 after actual CPU input preflight passed',Esito='Continuare',Motivazione=plan['correction'],Risultati=[finding],**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task42_transport_recovery_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='42 verified-input recovery full persisted output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='42 verified-input raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='42 verified-input GPU results',Descrizione=json.dumps(receipt['report']),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='42 verified-input completion requires audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(receipt['report']),Limitazioni='Exploratory original42;41 limitations and old FP32 controls remain, no automatic43 qualification or speedup claim.')
        for name in ('formula','41r_imposed','41r_autonomous'):put('runs',name,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task42 verified transport memory verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task42 transport schema preview valid; no writes')
