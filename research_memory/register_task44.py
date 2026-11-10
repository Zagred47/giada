"""Scoped43 conclusions and prospective44, lossless SQLite-only registration."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task44_protocol_v1'
    plan=json.loads((repo/'experiments/task44_preregistration_v1.json').read_text())
    prior=json.loads((repo/'experiments/results/task43_protocol_v1/result_memory_receipt.json').read_text())
    baseline={} if a.validate_only else json.loads((folder/'preregistration_receipt.json').read_text())['verification'] if a.result else prior['verification']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task44-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','localization',Nome='Task44 frozen41r spatial and channel-family composition localization',Tipo='Esplorativo',Stato='Concluso' if a.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Localize inherited local approximation, Ca-mediated interactions and axial feedback with paired frozen masks.'})
    if not a.result:
        audit=dict(task='43',archive_sha256='6e0d2272dad0a2c799afb83705e1bb4c29bbf4620db98fbb64d5f566fd5c0e18',archive_bytes=1284783196,
            learned_mode_passed=180,learned_mode_total=180,node_orders=['canonical','permuted'],permutation_passed=60,permutation_total=60,
            max_permutation_voltage_difference_mv=4.121147867408581e-13,all_parameters_finite=True,weights_logical_sha256=plan['freeze']['logical_weights_sha256'],
            native_floors_passed=True,timing_evaluable_per_order_mode=4992,timing_total_per_order_mode=13860,K_Pst_timing_evaluable=195,Nap_Et2_timing_evaluable=1260,
            K_Pst_max_timing_error_ms=0.,Nap_Et2_max_timing_error_ms=.1,imposed_max_abs_voltage_error_mv=.00010494226908264181,autonomous_max_abs_voltage_error_mv=.00880952036563798,
            autonomous_worst_pooled_voltage_rmse_mv=.0013009936343820483,native_voltage_range_mv=[-75.93881194026973,46.27798280845335],duration_ms=200,
            training_updates=0,thresholds_changed=False,original41_failures_resolved=False,speedup_claim=False,
            interpretation='Frozen heterogeneous multicompartment transfer succeeds within declared200ms panel. Index permutation negligible. Autonomous accumulation larger than imposed but within original gates; worst absolute error already occurs at coupling0. Broader timing coverage does not repair original six41 failures.')
        artifact=prior['records']['artifacts-giada-task43-final-v1'];experiment43=prior['records']['experiments-giada-task43-transfer-v1']
        obs=put('observations','43-audit',Nome='43 detailed paired transfer audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':[artifact]})
        finding=put('findings','43-scoped-transfer',Nome='43 frozen transfer passes broader declared panel, not universal qualification',Esito='Misto',Esperimenti=[experiment43],Osservazioni=[obs],Risultato=audit['interpretation'],Limitazioni='Original41 NO-GO retained;200ms and declared geometries only, no integrated speedup claim. Eager GPU, single-transient current evaluator.')
        put('lessons','43-localize-next',Nome='Transfer success enables causal composition localization',Lezione='Keep frozen all-channel candidate; use masked formula/learned interventions to separate sites/families and imposed/autonomous feedback before any refit.',Risultati=[finding],Limiti='Masked analytic controls are diagnostic, not a new reduced-abstraction candidate.')
        protocol=put('protocols','paired',Nome='44 eight batched masks with frozen41r and unchanged physics',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Original native/current/gate/compartment thresholds; mandatory legacy endpoint parity and null-substitution controls. Propagation sequences diagnostic only.','Regole di arresto':'Source/weight mismatch, failed execution preflight or null control stops run; native floor failure blocks learned quality judgment, not evidence of model failure.'})
        for label,hypothesis in plan['hypotheses'].items():put('claims',label,Nome='44 '+label,Tipo='Ipotesi',Stato='Aperta',Enunciato=hypothesis,Limiti='Frozen diagnostic localization; no additive-error assumption or architecture change.')
        for armname in plan['arms']:
            for mode in ('imposed','autonomous'):
                name=armname+'-'+mode
                arm=put('arms',name,Nome='44 '+name,Ruolo='Baseline' if armname=='none' else 'Trattamento',Protocollo=[protocol])
                put('runs',name,Nome='44 '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen;18 physical cases per seed')
        put('decisions','authorized',Nome='User authorizes exploratory44 with paired masked controls',Esito='Continuare',Motivazione='Explicit instruction after43 audit; fixed41r, original thresholds and abstraction; DB backup local only.',Risultati=[finding],**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task44_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='44 full persisted output and frozen weights',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='44 raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='44 paired localization outcome',Descrizione=json.dumps(receipt['report']),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='44 output requires scoped causal audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(receipt['report']),Limitazioni='Original41 limits and thresholds unchanged; no automatic45 authorization or speedup qualification.')
        for armname in plan['arms']:
            for mode in ('imposed','autonomous'):put('runs',armname+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if a.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if a.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task44 memory verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task44 schema preview valid; no writes')
