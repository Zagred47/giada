"""Prospective recovery, then complete results, through compact SQLite only."""
import argparse,hashlib,json,zipfile,subprocess
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');ap.add_argument('--running',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41n_recovery_v1';folder.mkdir(exist_ok=True)
    plan=json.loads((repo/'experiments/task41n_recovery_v1.json').read_text());m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    previous=(folder/'launch_memory_receipt.json' if (folder/'launch_memory_receipt.json').exists() else folder/'preregistration_receipt.json') if a.result or a.running else ROOT/'data/payload_migration_receipt.json'
    baseline={} if a.validate_only else json.loads(previous.read_text(encoding='utf-8'))['verification']
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41n-recovery-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code;return rows[0]['record_id']
    if a.running:
        if (folder/'technical_attempt_v1.json').exists():put('observations','technical-v1',Nome='Task41n Kaggle notebook-format failure before execution',Descrizione=(folder/'technical_attempt_v1.json').read_text())
        launch=json.loads((folder/'kaggle_launch.json').read_text());status=json.loads((folder/'status.json').read_text())
        put('observations','launch',Nome='Task41n recovery actual private Kaggle submission',Descrizione=json.dumps(dict(launch=launch,status=status,input=json.loads((folder/'input_verified.json').read_text()))))
        put('experiments','diagnostic',Stato='In corso')
        for name in plan['arms']:put('runs',name,Stato='In corso',**{'Validità tecnica':'Actual pinned private Kaggle submission, scheduler state recorded; evaluation only, frozen source, no training/selection. Original FP32 floors remain failed.'})
    elif not a.result:
        exp=put('experiments','diagnostic',Nome='Task41n evaluation-only diagnostic completion',Tipo='Forensic',Stato='Preregistrato',Descrizione='Complete autonomous/stress/cost only from existing frozen41n state. Original confirmation remains blocked by FP32 canonical floor; no new independent confirmation.',**{'Obiettivo informativo':'Measure all missing contrasts without changing weights, selection, native inputs, thresholds or failed-floor interpretation.'})
        art=put('artifacts','plan',Nome='Task41n immutable evaluation recovery plan',Tipo='Report',Versione='v1',Percorso='experiments/task41n_recovery_v1.json',**{'SHA-256':hashlib.sha256((repo/'experiments/task41n_recovery_v1.json').read_bytes()).hexdigest()})
        obs=put('observations','authorized',Nome='User authorization and frozen recovery provenance',Descrizione=json.dumps(plan),**{'Artefatti dettagliati':[art]})
        protocol=put('protocols','completion',Nome='Task41n missing-evaluations-only recovery',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Artefatto preregistrazione':[art],'Criteri di successo':'All measurements complete, immutable input/weights preserved, exact execution parity. Original FP32 floor failures remain visible and primary confirmation cannot pass. Original scientific thresholds unchanged.','Regole di arresto':'Technical hardware/hash/parity failure: archive partial result, no duplicate or training restart.'})
        impl=put('implementations','worker',Nome='Task41n evaluation-only recovery worker',Commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41n_recovery.py','Ambiente':'Pinned original Torch2.11.0+cu130/TeslaT4; verify frozen implementation parity. No NEURON process or teacher regeneration.'})
        for name in plan['arms']:
            arm=put('arms',name,Nome='Task41n recovery '+name,Ruolo='Baseline' if name in ('dual','H_control','dual_fp32') else 'Trattamento',Protocollo=[protocol],Modello=[ref(f'models-giada-task41n-{name}-v1')],Implementazione=[impl],Partizioni=[ref('partitions-giada-task41n-confirmation-v1')],**{'Configurazione residua':'Original frozen41n arm; original numerical policy and all thresholds unchanged. Evaluation-only diagnostic continuation.'})
            put('runs',name,Nome='Task41n missing evaluation '+name,Stato='Pianificata',Braccio=[arm],Seed='Frozen models17,29,43; original41n native/stress input seeds unchanged',**{'Validità tecnica':'CPU equivalence and immutable archive tests passed. Actual GPU identity/parity required; no new eligibility selection.'})
        put('findings','scope',Nome='Original41n floor failures constrain recovery interpretation',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='Recovery is authorized only to complete measurements. FP32 autonomous canonical response checks218/234 stay failed; matched64 passed. No original confirmation reinstatement or qualified speedup.')
    else:
        backup=json.loads((repo/'artifacts/task41n_recovery_kaggle_external/external_backup_receipt.json').read_text());assert backup['valid'] and backup['runtime_independent']
        archive=__import__('pathlib').Path(backup['local'])
        with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==backup['remote']['sha256']
        art=put('artifacts','final',Nome='Task41n complete diagnostic recovery outside runtime',Tipo='Report',Versione='v1',Percorso=archive.relative_to(repo).as_posix(),**{'SHA-256':backup['remote']['sha256'],'Dimensione byte':archive.stat().st_size})
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            report=backup['report']
            if not backup['technical_failure']:assert report['complete'] and report['training_updates_performed']==0 and not report['selection_recomputed'] and not report['thresholds_changed'] and not report['original_confirmation_passed']
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='Task41n recovered original '+name,Descrizione=z.read(name).decode('utf-8'),**{'Artefatti dettagliati':[art]})
        exp=put('experiments','diagnostic',Stato='Concluso')
        for name in plan['arms']:put('runs',name,Stato='Interrotta' if backup['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art],'Validità tecnica':'Technical partial output, not model NO-GO; inspect preserved logs. No automatic retry.' if backup['technical_failure'] else 'Evaluation-only diagnostic completion; immutable source and weights checked; no training/selection. Original FP32 canonical floor failures retained; not a successful original confirmation.'})
        obs=put('observations','summary',Nome='Task41n recovered autonomous,stress,cost summary',Descrizione=json.dumps(report),**{'Artefatti dettagliati':[art]})
        put('findings','complete',Nome='Task41n technical partial recovery' if backup['technical_failure'] else 'Task41n missing measurements completed with failed-floor caveat',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(report if backup['technical_failure'] else {k:report[k] for k in ('decision','summaries','stress','equivalent_implementations','candidate_measurements_passed')}),Limitazioni='Original FP32 floor remains failed; no original confirmation, no new selection or qualified speedup;Task42 not authorized.')
    print('Staged41n recovery '+('result' if a.result else 'protocol')+'; schema validation before writes',flush=True)
    change=m.commit_batch()
    if a.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('result_memory_receipt.json' if a.result else ('launch_memory_receipt.json' if a.running else 'preregistration_receipt.json')),dict(valid=True,verification=verification,change=change,records=ids,airtable_accessed=False,training_updates=0,thresholds_changed=False,DB_backup='local payload-v2 refresh pending'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task41n recovery mirror verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Recovery schema preview valid; no writes')
