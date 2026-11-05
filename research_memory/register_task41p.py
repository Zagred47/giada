"""Preregister and record actual41p training in local compact SQLite only."""
import argparse,json,hashlib,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41p_repair_v1';plan=json.loads((repo/'experiments/task41p_preregistration_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if args.result else repo/'experiments/results/task41o_causal_v1/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification'];m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41p-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','repair',Nome='Task41p targeted existing-head tail repair',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Translate41o equilibrium/update attribution into a non-oracle repair; paired arm/seed optimizer streams; protect acquired cases and measure cost.'})
    if not args.result:
        claims=[]
        for name,text in plan['hypotheses'].items():claims.append(put('claims',name,Nome='Task41p '+name,Tipo='Ipotesi',Stato='Aperta',Enunciato=text,**{'Condizioni di falsificazione':'No reduction of eleven opened stress failures, a newly failed acquired case, or development regression; report every budget and arm.','Limiti':'Diagnostic repair after opened41n; not a sealed independent confirmation or evidence for unbounded scaling.'}))
        put('experiments','repair',**{'Ipotesi':claims})
        art=put('artifacts','plan',Nome='Task41p preregistration and fixed quality gates',Tipo='Report',Versione='v1',Percorso='experiments/task41p_preregistration_v1.json',**{'SHA-256':hashlib.sha256((repo/'experiments/task41p_preregistration_v1.json').read_bytes()).hexdigest()})
        protocol=put('protocols','paired',Nome='Task41p equilibrium versus joint Nap calibration',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Artefatto preregistrazione':[art],'Criteri di successo':plan['success'],'Regole di arresto':plan['stop']})
        put('observations','rationale',Nome='Task41o causal evidence motivating41p',Descrizione=json.dumps(dict(learned32=61,learned64=61,oracle_u=69,oracle_z=61,oracle_both=72,teacher_previous=72,total=72,interpretation='Equilibrium repair first; Nap residual needs joint coefficients. Oracles are not deployable improvements. FP32 canonical failures remain separate.')))
        for name in plan['arms']:
            arm=put('arms',name,Nome='Task41p '+name,Ruolo='Baseline' if name=='baseline' else 'Trattamento',Protocollo=[protocol],**{'Configurazione residua':plan['trainable'] if name!='baseline' else 'Original frozen dual; no updates. All arms same quality/cost gates.'})
            put('runs',name,Nome='Task41p '+name,Stato='Pianificata',Braccio=[arm],Seed='Models17/29/43; shared new fit and development seeds per protocol',**{'Validità tecnica':'CPU real-checkpoint zero-update identity, registered-head isolation and finite loss/gradients verified. Actual GPU parity required before training.'})
    else:
        receipt=json.loads((repo/'artifacts/task41p_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent']
        path=__import__('pathlib').Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='Task41p complete external checkpoint/result archive',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        put('observations','launch',Nome='Task41p actual training submission',Descrizione=(folder/'kaggle_launch.json').read_text())
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='Task41p raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        report=receipt['report'];obs=put('observations','summary',Nome='Task41p repair results and regressions',Descrizione=json.dumps(report),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='Task41p existing-head repair outcome',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(report.get('arms',report)),Limitazioni='No independent confirmation or Task42 authorization; original FP32 failures218/234 retained; no qualified speedup until all quality gates pass.')
        for name in plan['arms']:put('runs',name,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task41p memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41p schema preview valid; no writes')
