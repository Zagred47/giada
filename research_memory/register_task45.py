"""Detailed44 localization audit and conditional frozen45, mirror SQLite only."""
import argparse,json,hashlib,zipfile
from pathlib import Path
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--result',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task45_protocol_v1';plan=json.loads((repo/'experiments/task45_preregistration_v1.json').read_text())
    prior=json.loads((repo/'experiments/results/task44_protocol_v1/result_memory_receipt.json').read_text())
    baseline={} if a.validate_only else (json.loads((folder/'preregistration_receipt.json').read_text())['verification'] if a.result else prior['verification'])
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task45-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    exp=put('experiments','coverage',Nome='Task45 controlled coverage first, frozen41r, optional refit deferred',Tipo='Esplorativo',Stato='Concluso' if a.result else 'Preregistrato',Descrizione=json.dumps(plan),**{'Obiettivo informativo':'Determine whether expanded causal histories and diagnostic geometries expose a need for controlled refit; no unnecessary training after44 all-pass.'})
    if not a.result:
        audit=dict(task='44',archive_sha256='9822f047a231eb0d23bb3f5ff5777bcde8c31cb5e9ec2ae8fff96926f76fa860',archive_bytes=442243567,
            arms=8,passed_per_arm_mode=54,total_per_arm_mode=54,all_parameters_and_traces_finite=True,weights_logical_sha256=plan['freeze']['logical_weights_sha256'],
            native_floors_and_legacy_parity_passed=True,null_control_max_error=0.,new_autonomous_failures=0,new_coupled_failures=0,
            autonomous_max_voltage_error_mv=dict(site_proximal=.0027005420727874707,site_distal=.00880952036563798,calcium=.0007486850400866274,sodium=.007829845142765635,potassium=.0020499646645095737,ih=.00037035638794691295,all=.00880952036563798),
            imposed_all_max_voltage_error_mv=.00010494226908264181,
            site_controls_offsite_error_disconnected_mv=0.,maximum_offsite_error_coupled_mv=.0006166768167901182,
            worst_all_case=dict(geometry='chain4',protocol='recovery',coupling=0.,seed=29),
            same_worst_case_sodium_only_max_voltage_error_mv=.007829845142765635,
            autonomous_family_nonadditivity_max_voltage_rmse_mv=3.777292093272618e-6,
            autonomous_family_nonadditivity_median_fraction=.0014211011951675257,
            crossing_count_mismatches=0,maximum_crossing_shift_ms=0.,crossing_resolution_ms=.1,
            all_autonomous_reference_upcrossings=72,upcrossing_sites_with_model_seed=36,
            original41_failures_resolved=False,speedup_claim=False,
            interpretation='All declared44 arms pass. Sodium dominates voltage error among tested families, while Ih has largest normalized current error: normalized current ranking does not equal electrical impact. Off-site error is exactly zero disconnected, small when coupled; no new coupling/feedback failures. Not a newly trained model or universal qualification.')
        art=prior['records']['artifacts-giada-task44-final-v1'];oldexp=prior['records']['experiments-giada-task44-localization-v1']
        obs=put('observations','44-audit',Nome='44 detailed spatial and family localization audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':[art]})
        finding=put('findings','44-localization',Nome='44 local approximation propagates controllably; no frozen refit trigger in opened panel',Esito='Misto',Esperimenti=[oldexp],Osservazioni=[obs],Risultato=audit['interpretation'],Limitazioni='Opened43 protocols,200ms, declared small geometries. Original six41 timing failures remain. Propagation0mV crossings descriptive, no speedup qualification.')
        put('lessons','44-current-vs-voltage',Nome='Rank channel error by downstream causal effect, not only normalized current',Lezione='Use matched site/family substitutions and imposed/autonomous contrasts. Largest normalized current error need not dominate voltage; keep Ca-mediated and nonlinear interactions explicit.',Risultati=[finding],Limiti='44 declared panel only; no universal channel ranking.')
        protocol=put('protocols','frozen-first',Nome='45 prospective frozen coverage before optional refit',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(plan),**{'Criteri di successo':'Original gates on all eligible full and prefix trajectories; reference errors block model judgment. Report coverage and remaining non-evaluable events.','Regole di arresto':'Immutable source/weight or execution parity/null-control failure stops run. Any proposed refit requires valid-reference evidence and separate decision; no automatic optimizer.'})
        for label,hypothesis in plan['hypotheses'].items():put('claims',label,Nome='45 '+label,Tipo='Ipotesi',Stato='Aperta',Enunciato=hypothesis,Limiti='Prospective exploratory frozen coverage, not Task47 confirmation; training mini scaling deferred.')
        for armname in plan['arms']:
            for mode in ('imposed','autonomous'):
                name=armname+'-'+mode;arm=put('arms',name,Nome='45 '+name,Ruolo='Baseline' if armname=='none' else 'Trattamento',Protocollo=[protocol])
                put('runs',name,Nome='45 '+name,Stato='Pianificata',Braccio=[arm],Seed='17/29/43 frozen;27 physical cases,100/500/1000ms prefixes')
        put('decisions','conditional-refit',Nome='User authorizes45 frozen; optional refit contingent, not automatic',Esito='Continuare',Motivazione='44 had no failing eligible arms. Respect explicit frozen41r instruction: first audit expanded coverage; preserve roadmap optional refit and training mini scaling as deferred conditional work, not falsely completed.',Risultati=[finding],**{'Data':'2026-11-10'})
    else:
        receipt=json.loads((repo/'artifacts/task45_kaggle_external/external_backup_receipt.json').read_text());assert receipt['valid'] and receipt['runtime_independent'];path=Path(receipt['local'])
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['remote']['sha256']
        art=put('artifacts','final',Nome='45 full persisted frozen coverage output',Tipo='Report',Versione='v1',Percorso=str(path),**{'SHA-256':receipt['remote']['sha256'],'Dimensione byte':receipt['remote']['bytes']})
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):put('observations','raw-'+name,Nome='45 raw '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]})
        obs=put('observations','summary',Nome='45 frozen coverage and conditional refit outcome',Descrizione=json.dumps(receipt['report']),**{'Artefatti dettagliati':[art]})
        put('findings','outcome',Nome='45 frozen coverage result requires scoped audit',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(receipt['report']),Limitazioni='No training performed, original41 limitations retained. No automatic refit/46/47 or speedup qualification.')
        for armname in plan['arms']:
            for mode in ('imposed','autonomous'):put('runs',armname+'-'+mode,Stato='Interrotta' if receipt['technical_failure'] else 'Completata',**{'Artefatti prodotti':[art]})
    m.commit_batch()
    if a.validate_only:return
    verify=m.verify_incremental(baseline);assert verify['valid']
    write_json(folder/('result_memory_receipt.json' if a.result else 'preregistration_receipt.json'),dict(valid=True,verification=verify,records=ids,airtable_accessed=False))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('Task45 memory verified/exported.',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task45 schema preview valid; no writes')
