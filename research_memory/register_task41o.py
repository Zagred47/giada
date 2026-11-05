"""41o preregistration/results, compact SQLite only; never Airtable."""
import argparse,json,zipfile,hashlib
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41o_causal_v1';plan=json.loads((repo/'experiments/task41o_preregistration_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task41n_recovery_v1/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41o-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','causal',Nome='Task41o causal localization before further repair',Tipo='Forensic',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Distinguish arithmetic, equilibrium, relaxation, accumulation and crossing sensitivity. Oracle controls are not candidates; no threshold or abstraction change.'})
    if not args.result:
        claims=[]
        for name,text in plan['hypotheses'].items():
            claims.append(put('claims',name,Nome='Task41o '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=text,**{'Condizioni di falsificazione':'Registered paired causal controls fail to rescue baseline failures, or produce regressions; report case-level interactions, not only aggregate gains.','Limiti':'Opened41n diagnostic panel; not an independent confirmation or evidence that an oracle is deployable.'}))
        put('experiments','causal',**{'Ipotesi':claims})
        art=put('artifacts','plan',Nome='Task41o preregistration',Tipo='Report',Versione='v1',Percorso='experiments/task41o_preregistration_v1.json',**{'SHA-256':hashlib.sha256((repo/'experiments/task41o_preregistration_v1.json').read_bytes()).hexdigest()})
        protocol=put('protocols','surgery',Nome='Task41o paired frozen coefficient/state interventions',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Artefatto preregistrazione':[art],'Criteri di successo':plan['success'],'Regole di arresto':plan['stop']})
        obs=put('observations','audit',Nome='Task41n opened stress localization and memory queries',Descrizione=(folder/'audit.json').read_text())
        put('findings','scope',Nome='Eleven timing failures, not broad current amplitude failure',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='All eleven stress failures involve timing only; gate/amplitude/charge predicates pass. Local channel oracle rescues are diagnostics only. Old FP32 floor failures retained.')
        for name in plan['arms']:
            arm=put('arms',name,Nome='Task41o '+name,Ruolo='Baseline' if name=='learned32' else 'Trattamento',Protocollo=[protocol],**{'Configurazione residua':'Frozen weights, original inputs/seeds/quality. Diagnostic-only coefficient/state surgery; no selection.'})
            put('runs',name,Nome='Task41o '+name,Stato='Pianificata',Braccio=[arm],Seed='Frozen17/29/43; same24 opened stress histories')
    else:
        receipt=json.loads((repo/'artifacts/task41o_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent']
        put('observations','launch',Nome='Task41o actual pinned private Kaggle submission',Descrizione=(folder/'kaggle_launch.json').read_text())
        art=put('artifacts','result',Nome='Task41o persisted verified final archive',Tipo='Report',Versione='v1',Percorso=receipt['local'],**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(receipt['local']) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='Task41o raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        report=receipt['report'];obs=put('observations','summary',Nome='Task41o causal contrasts',Descrizione=json.dumps(report),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='Task41o completed causal diagnosis' if not receipt['technical_failure'] else 'Task41o technical interruption',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(report.get('contrasts',report)),Limitazioni='No model improvement or speedup claimed; oracles are diagnostic, not inference; original FP32 failures retained.')
        for name in plan['arms']:put('runs',name,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task41o memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41o schema preview valid; no writes')
