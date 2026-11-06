"""Prospective paired41s hypotheses, fits and results; SQLite only."""
import argparse,json,hashlib,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41s_protocol_v1';plan=json.loads((repo/'experiments/task41s_preregistration_v1.json').read_text());manifest=json.loads((folder/'fit_manifest.json').read_text());assert manifest['valid']
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task41r_protocol_v1/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification'];m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41s-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','factorial',Nome='Task41s existing-head direct timing2x2',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Distinguish nonlinear correction failure, temporal objective support and history/duration coverage, plus K_Pst gate attribution. Same abstraction and gates; no stochastic training.'})
    if not args.result:
        claims=[]
        for name,text in plan['hypotheses'].items():claims.append(put('claims',name,Nome='Task41s '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=text,**{'Condizioni di falsificazione':'No rescue of baseline failures, newly failed acquired cases or no paired factor benefit. Record interactions, all arms and case-level regressions.','Limiti':'Opened diagnostic41n data; no sealed confirmation; linearized fit may not transfer to nonlinear autonomous system.'}))
        put('experiments','factorial',**{'Ipotesi':claims})
        art=put('artifacts','plan',Nome='Task41s fixed protocol and deterministic fit manifest',Tipo='Report',Versione='v1',Percorso='experiments/task41s_preregistration_v1.json',**{'SHA-256':hashlib.sha256((repo/'experiments/task41s_preregistration_v1.json').read_bytes()).hexdigest()})
        protocol=put('protocols','paired',Nome='Task41s frozen-head causal current Jacobian matrix',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Artefatto preregistrazione':[art],'Criteri di successo':plan['progress'],'Regole di arresto':plan['stop']})
        put('observations','fit_manifest',Nome='Task41s locally frozen deterministic fits before GPU access',Descrizione=json.dumps(manifest),**{'Artefatti dettagliati':[art]})
        put('observations','linearization_audit',Nome='Task41s fit-only predicted versus actual nonlinear residual',Descrizione=(folder/'fit_linearization_audit.json').read_text(),**{'Artefatti dettagliati':[art]})
        for name in plan['arms']:
            arm=put('arms',name,Nome='Task41s '+name,Ruolo='Baseline' if name=='baseline' else 'Trattamento',Protocollo=[protocol],**{'Configurazione residua':plan['trainable_heads'] if name!='baseline' else 'Original frozen dual, no update.'})
            put('runs',name,Nome='Task41s '+name,Stato='Pianificata',Braccio=[arm],Seed='Frozen17/29/43; registered disjoint fit/development input seeds',**{'Validità tecnica':'Local exact affine/current Jacobian finite-difference tests and zero-update materialization passed. Original GPU/Torch identity and weight hashes must pass before measurement.'})
    else:
        receipt=json.loads((repo/'artifacts/task41s_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent']
        path=__import__('pathlib').Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='Task41s verified persisted full output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        put('observations','launch',Nome='Task41s actual pinned Kaggle submission',Descrizione=(folder/'kaggle_launch.json').read_text())
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='Task41s raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        report=receipt['report'];obs=put('observations','summary',Nome='Task41s factorial quality/regression/cost summary',Descrizione=json.dumps(report),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='Task41s direct timing factorial outcome',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(report.get('arms',report)),Limitazioni='No independent confirmation, Task42 authorization or qualified speedup. Original failed FP32 controls218/234 remain explicit. All arms reported; baseline preserved.')
        for name in plan['arms']:put('runs',name,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid'];write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task41s memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41s schema preview valid; no writes')
