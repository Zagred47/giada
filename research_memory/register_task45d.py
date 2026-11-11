"""Preregister45d factors, claims, runs and later raw result; SQLite only."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');ap.add_argument('--refresh',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45d_protocol_v1';folder.mkdir(exist_ok=True);p=json.loads((repo/'experiments/task45d_preregistration_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if args.result or args.refresh else repo/'experiments/results/task45c_cost_recovery_v2/result_memory_receipt.json'
    baseline={} if args.validate_only else json.loads(previous.read_text())['verification'];m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task45d-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','matrix',Nome='45d support, feature learnability and execution matrix',Tipo='Esplorativo',Stato='Concluso' if args.result else 'Preregistrato',Descrizione=json.dumps(p),**{'Obiettivo informativo':'Distinguish optimizer truncation, data support, frozen features, timing objective and execution duplication without changing original scientific gates.'})
    if not args.result:
        protocol=put('protocols','matrix',Nome='45d preregistered paired contrasts',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(p),**{'Criteri di successo':'Unchanged current/state/timing/compartment gates at all original prefixes; no new per-identity DEV/stress/regression failures. Exact execution classified separately from approximate diagnostics.','Regole di arresto':'Input/GPU/hash/finiteness/reference/parity failure stops interpretation; preserve partial artifacts; no automatic restart or promotion to46/47.'})
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):
                key=arm+'-'+mode;aa=put('arms',key,Nome='45d '+key,Ruolo='Baseline' if arm in ('none','C0','C1','R41','R111') else 'Trattamento',Protocollo=[protocol]);put('runs',key,Nome='45d '+key,Stato='Pianificata',Braccio=[aa],Seed='17/29/43; same matched geometry streams, DT0.1ms, all100/500/1000ms prefixes; no47')
        claims=[put('claims',h['id'],Nome='45d '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['contrast'],Limiti='Planned controlled contrast, not an established causal explanation. '+p['execution_scope']) for h in p['hypotheses']]
        put('experiments','matrix',Ipotesi=claims)
        put('observations','45c-evidence',Nome='45c motivates45d without overstating convergence or speed',Descrizione=json.dumps(dict(Nap_requested_norm=[31.637,83.997,18.653],cap=2,refinements=4,remaining_seed29_norm=77.997,original45_C0_J_1000_passed=68,safe_hyper_100_C0_J=[66,69],safe_hyper_1000_C0_J=[64,64],stress_C0_J=66,Ih_worst_timing_C0_I1_ms=[.9,3.4],cost_J_integrated642_ms=214.051,formula32_ms=106.085,formula64_ms=129.373,interpretation='Capped training incomplete; duplicate execution and precision contributions must be measured, not assumed.')))
        put('decisions','authorized',Nome='User authorizes45d private notebook and local-only DB backup',Esito='Continuare',Motivazione='Explicit authorization: all comparisons same notebook, unchanged thresholds and abstraction, register SQLite, push and run Kaggle with external outputs. Existing private authorized checkpoint Dataset reused; no new native/DB upload.',**{'Data':'2026-11-11'})
    else:
        receipt=json.loads((repo/'artifacts/task45d_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','output',Nome='45d persisted complete or partial output',Tipo='Report',Percorso=str(path),Versione='v1',**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']});obs=[]
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='45d '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        put('findings','result',Nome='45d full factorial and execution outcome requires scoped audit',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(receipt['report']),Limitazioni='Technical completion is not scientific success. Source precision and old41 limits remain explicit; no automatic46/47 or speedup claim.')
        for arm in p['coupled_arms']:
            for mode in ('imposed','autonomous'):put('runs',arm+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid'];write_json(folder/('result_memory_receipt.json' if args.result else 'preregistration_receipt.json'),dict(valid=True,verification=verification,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('45d SQLite verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('45d schema valid; no writes')
