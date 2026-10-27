"""Seal existing prospective41l records; local SQLite only, no remote sync."""
import argparse,hashlib,json
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');mode=ap.add_mutually_exclusive_group();mode.add_argument('--running',action='store_true');mode.add_argument('--submitted',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41l_protocol_v1'
    spec=json.loads((repo/'experiments/task41l_preregistration_v1.json').read_text())
    local=json.loads((folder/'local_verification.json').read_text());gpu=json.loads((folder/'gpu_preflight.json').read_text())
    digest=hashlib.sha256((repo/'experiments/task41l_preregistration_v1.json').read_bytes()).hexdigest()
    assert local['valid'] and gpu['valid'] and gpu['all_parameters_finite']
    assert local['protocol_sha256']==gpu['protocol_sha256']==digest
    baseline_path=folder/'memory_receipt.json' if args.running or args.submitted else repo/'experiments/results/task41k_audited_v1/memory_receipt.json'
    if args.running and (folder/'launch_submitted_memory_receipt.json').exists():baseline_path=folder/'launch_submitted_memory_receipt.json'
    baseline=json.loads(baseline_path.read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={};arts=[]
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41l-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def artifact(path):
        raw=path.read_bytes();name=path.relative_to(repo).as_posix()
        rid=put('artifacts',name,Nome='Task41l '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts.append(rid);return rid
    date=datetime.now(timezone.utc).date().isoformat()
    exp=ref('experiments-giada-task41l-regression-attribution-v1')
    if args.submitted:
        launch=json.loads((folder/'kaggle_active_launch.json').read_text())
        status_path=repo/'artifacts/task41l_kaggle_external/pipeline_status.json'
        observed=json.loads(status_path.read_text())
        assert launch['mode']=='train' and launch['revision']==gpu['revision']
        assert observed['launch']==launch and observed['state'] in ('running','queued','complete')
        artifact(folder/'kaggle_active_launch.json');artifact(status_path)
        obs=put('observations','submitted',Nome='Task41l actual Kaggle submission',Descrizione=json.dumps(dict(launch=launch,status=observed,first_training_backup_verified=False)),**{'Artefatti dettagliati':arts})
        put('findings','submitted',Nome='Task41l scientific run submitted',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Pinned private Kaggle run submitted after CPU/GPU preflight and DB preregistration.',Limitazioni='No scientific result. First external training checkpoint verification pending; finite workflow retrieves and verifies each phase before continuation.')
        put('experiments','regression-attribution',Stato='In corso')
        for arm in spec['arms']:
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='In corso',**{'Hardware e ambiente':json.dumps(gpu),'Validità tecnica':'Kaggle submission verified; training checkpoint/evaluation pending.'})
        put('actions','launch',Stato='In corso',Descrizione='Actual private Kaggle submission; finite external-backup pipeline active. No completed-training claim.')
    elif args.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['all_recovered_tensors_finite']
        artifact(folder/'launch.json');artifact(folder/'kaggle_active_launch.json')
        obs=put('observations','launch',Nome='Task41l verified external checkpoint',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':arts})
        put('findings','launch',Nome='Task41l finite checkpoint off runtime',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Actual pinned Kaggle run, finite checkpoint and external SHA/CRC verified. Scientific evaluation pending.',Limitazioni='Technical progress only, not a scientific GO.')
        put('experiments','regression-attribution',Stato='In corso')
        for arm in spec['arms']:
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='In corso',**{'Hardware e ambiente':json.dumps(gpu),'Validità tecnica':'Actual external finite checkpoint verified. No scientific result yet.'})
        put('actions','launch',Stato='Conclusa',Descrizione='Kaggle launch and external finite checkpoint verified; phased pipeline active.')
    else:
        for name in ('experiments/task41l_preregistration_v1.json','experiments/task41l_design.md','notebooks/giada_task41l_regression_attribution.ipynb',
                     'src/giada_teacher/task41l_regression_attribution.py','src/giada_teacher/task41l_support.py','src/giada_teacher/task41l_frozen_attribution.py','src/giada_teacher/task41l_evaluation.py',
                     'scripts/run_task41l_regression_attribution.py','scripts/task41l_kaggle_pipeline.py','scripts/task41l_gpu_probe.py','scripts/build_task41l_protocol.py','tests/test_task41l_regression_attribution.py','research_memory/register_task41l.py'):
            artifact(repo/name)
        prereg=arts[0];artifact(folder/'local_verification.json');artifact(folder/'gpu_preflight.json')
        if (folder/'gpu_preflight_before_phase_split.json').exists():artifact(folder/'gpu_preflight_before_phase_split.json')
        history=json.loads((folder/'preflight_history.json').read_text()) if (folder/'preflight_history.json').exists() else []
        if history:artifact(folder/'preflight_history.json')
        # Keep original hypothesis wording, evidence and stable identities.
        claims=[ref('claims-giada-task41l-'+h['id']+'-v1') for h in spec['hypotheses']]
        put('experiments','regression-attribution',Stato='Preregistrato',Ipotesi=claims,Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Separate amplitude,coverage,canonical group protection,channel feedback and numerical interaction;compression parked.'})
        gen=put('generators','fit',Nome='Task41l nested controlled FIT generators',Versione='v1',Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':arts})
        ds=put('datasets','fit-selection',Nome='Task41l nested1x4x FIT and disjoint selection',Versione='v1',Generatore=[gen],Schema='18occupancies and causalV,Ca; canonical labels only, no new inference input.',Copertura=spec['support'])
        parts=[put('partitions',name,Nome='Task41l '+name,Dataset=[ds],Ruolo=role,**{'Regola di assegnazione':json.dumps(spec['new_seeds']),'Restrizioni di accesso':'Opened panels after freeze, diagnostics only; not independent confirmation.'}) for name,role in [('fit','Fit'),('development','Development')]]
        protocol=put('protocols','design',Nome='Task41l matched regression attribution',Versione='v1-sealed',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
                     **{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[prereg],'Hash preregistrazione':digest,'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':date})
        impl=put('implementations','worker',Nome='Task41l vectorized seeds worker',Commit=gpu['revision'],**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41l_regression_attribution.py','Artefatti codice':arts,'Ambiente':json.dumps(gpu)})
        aa={}
        for arm in spec['arms']:
            model=put('models',arm['name'],Nome='Task41l '+arm['name'],Ruolo='Surrogate',Descrizione=json.dumps(arm),**{'Contratto di stato':'Same18occupancies,4004params,dt0.1ms,bounded affine update,causalV/Ca.','Contratto di input e output':'Same learned gating;readout,synapses,membrane unchanged;primaryFP32.'})
            aa[arm['name']]=put('arms',arm['name'],Nome='Task41l '+arm['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(arm)})
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='Pianificata',Seed=str(seed),Braccio=[aa[arm['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=arm,budget=spec['budget'],seeds=spec['new_seeds'])),'Validità tecnica':'CPU and GPU preflight passed; scientific run not launched at registration.'})
        for comparison in spec['registered_comparisons']:
            label={'A':'amplitude','D':'coverage','P':'protection'}[comparison['name'][0]]
            left,right=comparison['control'],comparison['treatment'];bits=''.join(left[i] for i in (3,5,7))
            put('contrasts',label+'-'+bits,Nome='Task41l '+comparison['name'],Protocollo=[protocol],Bracci=[aa[left],aa[right]],**{'Contrasto e coefficienti':'Treatment-control,pairedseeds/cases;allthresholds unchanged.','Soglia interpretativa':'Report continuous effects,new/resolved/remaining failures,3seeds;all original predicates required.','Confondenti controllati':spec['budget']+' '+spec['support']})
        for name,value in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['cost']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
            metric=put('metrics',name,Nome='Task41l '+name,Formula=json.dumps(value));put('evaluations',name,Nome='Task41l '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(value))
        obs=put('observations','preflight',Nome='Task41l CPU and actual GPU preflight',Descrizione=json.dumps(dict(local=local,gpu=gpu,technical_history=history)),**{'Artefatti dettagliati':arts})
        finding=put('findings','preflight',Nome='Task41l technical execution verified',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato=json.dumps(gpu),Limitazioni='Technical only;no scientific qualification or speedup claimed.')
        dec=put('decisions','authorized',Nome='Task41l user-approved factorial repair',Esito='Continuare',Risultati=[finding,ref('findings-giada-task41k-result-outcome-v1')],Motivazione='User explicitly approves notebook,DB,push,GPUpreflight,Kaggle andexternalbackups. DBbackup local. All original hypotheses retained, compression separate.',**{'Data':date})
        put('actions','launch',Nome='Task41l pinned Kaggle launch',Stato='In corso',Descrizione='Register before scientific training;external phasearchive SHA/CRC before continuation.',**{'Decisioni di origine':[dec]})
    print(json.dumps(dict(staged_records=len(ids),running=args.running)),flush=True)
    change=m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    receipt_name='launch_memory_receipt.json' if args.running else 'launch_submitted_memory_receipt.json' if args.submitted else 'memory_receipt.json'
    write_json(folder/receipt_name,dict(valid=True,verification=verification,change=change,records=ids,registration_before_scientific_run=not(args.running or args.submitted),planned_training_runs=24,protocol_sha256=digest,airtable_accessed=False,DB_backup='local refresh pending'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41l local memory verified and exported',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Preview valid; no DB writes')
