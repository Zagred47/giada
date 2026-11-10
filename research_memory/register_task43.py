"""Task42 scoped audit and Task43 prospective frozen transfer, SQLite only."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task43_protocol_v1'
    plan=json.loads((repo/'experiments/task43_preregistration_v1.json').read_text())
    previous=folder/'preregistration_receipt.json' if a.result else repo/'experiments/results/task42_transport_recovery_v1/result_memory_receipt.json'
    baseline={} if a.validate_only else json.loads(previous.read_text())['verification']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task43-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','transfer',Nome='Task43 frozen41r heterogeneous node-order transfer',Tipo='Esplorativo',Stato='Concluso' if a.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Separate node indexing errors, local heterogeneous transfer and free feedback without changing abstraction, weights or thresholds.'})
    if not a.result:
        audit=dict(task='42',archive_sha256='4c687f91d46af71348818b5299af2ea878b08bdc58c4a99b101b3c8887af1627',
            current_and_compartment_passed_per_mode=180,total_per_mode=180,passive_native_checks_passed=15,
            autonomous_max_abs_voltage_error_mv=.0003439609469993,imposed_max_abs_voltage_error_mv=.00001135266976604,
            timing_evaluable_per_learned_mode=660,timing_total_per_learned_mode=13860,K_Pst_timing_evaluable=0,Nap_Et2_timing_evaluable=198,
            native_voltage_range_mv=[-75.82586196856,-30.87325658766],duration_ms=40,
            original_task41_stress_passed=66,original_task41_stress_total=72,
            original_task41_passed=False,all_parameters_finite=True,frozen_weights_unchanged=True,
            graph_execution='All15 execution cells eager fallback; no qualified speedup',
            interpretation='No new threshold failures in this short axial panel. Autonomous accumulation exists even at coupling0, not evidence of axial instability. Limited event coverage cannot repair or waive original six41 timing failures.')
        obs=put('observations','42-audit',Nome='Detailed42 transfer audit and event coverage',Descrizione=json.dumps(audit))
        finding=put('findings','42-scoped-transfer',Nome='42 succeeds within short-panel scope, not original41 timing qualification',Esito='Misto',Osservazioni=[obs],Risultato=audit['interpretation'],Limitazioni='40ms; K_Pst timing0 evaluable;660/13860 total timing evaluable per learned mode. Eager GPU fallback; no integrated speedup claim.')
        put('lessons','42-coverage',Nome='Pass counts require event-support denominators',Lezione='Report event-evaluable support by channel separately from current amplitude gates; do not equate absent timing events with success.',Risultati=[finding],Limiti='42 short current-clamp panel only; original long-tail stress suite unchanged.')
        protocol=put('protocols','paired',Nome='43 canonical/permuted frozen heterogeneous200ms transfer',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Original current, gate, compartment and native-floor gates; separately audited node-order numerical equivalence. No metric averaging waiver.','Regole di arresto':'Stop on immutable source or frozen weights mismatch; native floor failure blocks candidate judgment but retains diagnostics.'})
        for label,hypothesis in plan['hypotheses'].items():
            put('claims',label,Nome='43 '+label,Tipo='Ipotesi',Stato='Aperta',Enunciato=hypothesis,Limiti='Exploratory transfer only, no retraining or task41 retrospective promotion.')
        for order in ('canonical','permuted'):
            for mode in ('formula','41r_imposed','41r_autonomous'):
                name=order+'-'+mode
                arm=put('arms',name,Nome='43 '+name,Ruolo='Baseline' if mode=='formula' else 'Trattamento',Protocollo=[protocol])
                put('runs',name,Nome='43 '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen;60 physical cases/order')
        put('decisions','authorized',Nome='User authorizes exploratory43 with frozen41r and unchanged gates',Esito='Continuare',Motivazione='Explicit user instruction after42 audit; accepted41 limitations retained. New200ms references specific to43, not regenerated42 results.',Risultati=[finding],**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task43_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='43 persisted full output and frozen checkpoints',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='43 raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='43 frozen transfer output',Descrizione=json.dumps(receipt['report']),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='43 frozen transfer outcome requires scoped audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(receipt['report']),Limitazioni='Original41 failures retained; no automatic next-task promotion or speedup claim.')
        for order in ('canonical','permuted'):
            for mode in ('formula','41r_imposed','41r_autonomous'):put('runs',order+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if a.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if a.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task43 memory verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task43 schema preview valid; no writes')
