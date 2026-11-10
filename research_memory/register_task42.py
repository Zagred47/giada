"""Prospective42 paired axial protocol and complete local result registration."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');parser.add_argument('--result',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task42_protocol_v1';folder.mkdir(exist_ok=True)
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task41v_protocol_v1/task42_transition_memory_receipt.json'
    plan=json.loads((repo/'experiments/task42_preregistration_v1.json').read_text())
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();baseline={} if args.validate_only else json.loads(previous.read_text())['verification']
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    ids={}
    def put(table,suffix,**fields):
        code=f'{table}-giada-task42-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','axial',Nome='Task42 exploratory axial port, frozen41r and paired references',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Validate native axial reference floors and distinguish imposed-path ionic errors from coupled amplification; preserve41 NO-GO.'})
    if not args.result:
        claims=[put('claims',key,Nome='Task42 '+key,Tipo='Ipotesi',Stato='Aperta',Enunciato=text,Limiti='Exploratory40ms panel; no long-tail/full-cell or independent model qualification.') for key,text in plan['hypotheses'].items()]
        put('experiments','axial',Ipotesi=claims)
        artifact=put('artifacts','plan',Nome='Task42 unchanged-gate prospective protocol',Tipo='Report',Versione='v1',Percorso='experiments/task42_preregistration_v1.json',**{'SHA-256':hashlib.sha256((repo/'experiments/task42_preregistration_v1.json').read_bytes()).hexdigest()})
        protocol=put('protocols','paired',Nome='Task42 native and frozen paired axial matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Artefatto preregistrazione':[artifact],'Criteri di successo':'Native floor and original quality gates, all outcomes reported by topology/coupling/site; no automatic43 qualification.','Regole di arresto':'Stop on source/GPU/hash/nonfinite/parity error; failing native floor blocks candidate judgment but retains diagnostics.'})
        for name in ('formula','41r_imposed','41r_autonomous'):
            arm=put('arms',name,Nome='Task42 '+name,Ruolo='Baseline' if name=='formula' else 'Trattamento',Protocollo=[protocol],**{'Configurazione residua':'Frozen41r; no training or selection; identical protocols, local states, classical axial coupling.'})
            put('runs',name,Nome='Task42 '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen model seeds')
    else:
        backup=json.loads((repo/'artifacts/task42_kaggle_external/external_backup_receipt.json').read_text());assert backup['valid'] and backup['runtime_independent']
        path=Path(backup['local'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==backup['remote']['sha256']
        art=put('artifacts','final',Nome='Task42 externally recovered full output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':backup['remote']['sha256'],'Dimensione byte':backup['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='Task42 raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='Task42 reference floors and all paired modes',Descrizione=json.dumps(backup['report']),**{'Artefatti dettagliati':[art]})
        put('findings','result',Nome='Task42 exploratory result requires audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(backup['report']),Limitazioni=json.dumps(plan['limits']))
        for name in ('formula','41r_imposed','41r_autonomous'):put('runs',name,Stato='Interrotta' if backup['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task42 local memory verified and exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task42 schema preview valid; no writes')
