"""Prospective41j and actual running receipt. No Airtable operations."""
import argparse,hashlib,json
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--running',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41j_protocol_v1'
    spec=json.loads((repo/'experiments/task41j_preregistration_v1.json').read_text())
    local=json.loads((folder/'local_verification.json').read_text());gpu=json.loads((folder/'gpu_preflight.json').read_text())
    digest=hashlib.sha256((repo/'experiments/task41j_preregistration_v1.json').read_bytes()).hexdigest()
    assert local['valid'] and gpu['valid'] and gpu['all_parameters_finite']
    assert local['protocol_sha256']==gpu['protocol_sha256']==digest
    baseline_path=folder/'memory_receipt.json' if a.running else repo/'experiments/results/task41i_recovery_v1/memory_receipt.json'
    baseline=json.loads(baseline_path.read_text())['verification']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={};arts=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41j-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def artifact(path):
        raw=path.read_bytes();name=path.relative_to(repo).as_posix()
        rid=put('artifacts',name,Nome='Task41j '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=name,
                **{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        arts.append(rid);return rid
    date=datetime.now(timezone.utc).date().isoformat()
    if a.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['all_recovered_tensors_finite']
        artifact(folder/'launch.json');artifact(folder/'kaggle_active_launch.json')
        obs=put('observations','launch',Nome='Task41j real worker and verified external checkpoint',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':arts})
        put('findings','launch',Nome='Task41j finite original checkpoint off runtime',Esito='Positivo',Esperimenti=[ref('experiments-giada-task41j-comparison-v1')],Osservazioni=[obs],Risultato='Actual pinned Kaggle worker, finite checkpoint and SHA/CRC-verified archive outside runtime.',Limitazioni='Technical progress only, scientific result pending.')
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41j-comparison-v1','Stato':'In corso'})
        for arm in spec['arms']:
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='In corso',**{'Hardware e ambiente':json.dumps(gpu),'Validità tecnica':'Actual finite checkpoint and external backup verified; scientific evaluation pending.'})
        put('actions','launch',Stato='Conclusa',Descrizione='Kaggle launch and first external finite checkpoint verified.')
    else:
        for name in ('experiments/task41j_preregistration_v1.json','experiments/task41j_design.md','notebooks/giada_task41j_operator_reset.ipynb',
                     'src/giada_teacher/task41j_operator_reset.py','src/giada_teacher/task41j_evaluation.py','scripts/run_task41j_operator_reset.py',
                     'scripts/task41j_kaggle_pipeline.py','scripts/task41j_gpu_probe.py','scripts/build_task41j_protocol.py',
                     'tests/test_task41j_operator_reset.py','research_memory/register_task41j.py'):
            artifact(repo/name)
        prereg=arts[0];artifact(folder/'local_verification.json');artifact(folder/'gpu_preflight.json')
        hypotheses=[]
        for h in spec['hypotheses']:
            hypotheses.append(put('claims',h['id'],Nome='Task41j '+h['id'],Tipo='Ipotesi',Stato='Aperta',Enunciato=h['statement'],**{'Condizioni di falsificazione':h['falsifier']},Limiti='Prospective test; past DB similarities do not establish current causality.'))
        exp=put('experiments','comparison',Nome='Task41j joint relearning versus compact ionic operator',Stato='Preregistrato',Ipotesi=hypotheses,Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Distinguish constrained repair space, compact approximation/learnability, cumulative inference cost and numerical seams.'})
        gen=put('generators','fit',Nome='Task41j common atomic,path,feedback generators',Versione='v1',Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':arts})
        ds=put('datasets','fit-selection',Nome='Task41j disjoint paired FIT and selection',Versione='v1',Generatore=[gen],Schema='Same V/Ca,18gates,36 learned coefficients; canonical labels only; no new inference input.',Copertura=spec['support'])
        parts=[put('partitions',name,Nome='Task41j '+name,Dataset=[ds],Ruolo=role,**{'Regola di assegnazione':json.dumps(spec['new_seeds']),'Restrizioni di accesso':'Opened panels after freeze only; not independent confirmation.'}) for name,role in [('fit','Fit'),('development','Development')]]
        protocol=put('protocols','comparison',Nome='Task41j registered three-arm reset',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
            **{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[prereg],'Hash preregistrazione':digest,'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':date})
        impl=put('implementations','worker',Nome='Task41j heterogeneous-arm vectorized CUDA worker',Commit=gpu['revision'],**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41j_operator_reset.py','Artefatti codice':arts,'Ambiente':json.dumps(gpu)})
        aa={}
        for arm in spec['arms']:
            model=put('models',arm['name'],Nome='Task41j '+arm['name'],Ruolo='Surrogate',Descrizione=json.dumps(arm),**{'Contratto di stato':'18occupancies,dt0.1ms,bounded affine recurrence,all u/z learnable. Same abstraction.','Contratto di input e output':'Same causal V/Ca,calcium-only SK head,ionic readout,synapses and membrane unchanged.'})
            aa[arm['name']]=put('arms',arm['name'],Nome='Task41j '+arm['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(arm)})
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Nome=f"Task41j {arm['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[aa[arm['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=arm,budget=spec['budget'],new_seeds=spec['new_seeds']))})
        for name,left,right in [('representation','J_chain_joint','J_compact64'),('capacity','J_compact64','J_compact128')]:
            put('contrasts',name,Nome='Task41j '+name,Protocollo=[protocol],Bracci=[aa[left],aa[right]],**{'Contrasto e coefficienti':'Matched same-panel quality/severity and actual paired time,allseeds;full original gates mandatory.','Soglia interpretativa':'No new global regression; report all seed effects. Scientific GO requires frozen nominee to pass every original gate. Cost minimum10% unchanged.','Confondenti controllati':spec['budget']+' Distinct parameter counts and compact initial gap explicitly measured.'})
        for name,value in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['speed']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
            metric=put('metrics',name,Nome='Task41j '+name,Formula=json.dumps(value));put('evaluations',name,Nome='Task41j '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(value))
        obs=put('observations','preflight',Nome='Task41j CPU and actual GPU technical preflight',Descrizione=json.dumps(dict(local=local,gpu=gpu)),**{'Artefatti dettagliati':arts})
        finding=put('findings','preflight',Nome='Task41j verified equivalence,gradients and CUDA capture',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Sequential/parallel recurrence and gradients match; frozen H initially exact, compact gap not assumed zero; seed isolation,finite auxiliary gradients and actual CUDA capture verified.',Limitazioni='Technical test only; does not establish scientific qualification or speedup.')
        dec=put('decisions','authorized',Nome='Task41j approved operator strategy reset',Esito='Continuare',Risultati=[finding,ref('findings-giada-task41i-result-outcome-v1')],Motivazione='User approved joint optimization/compact comparison,unchanged thresholds/abstraction,Kaggle with external backups;DB local.',**{'Data':date})
        put('actions','launch',Nome='Task41j pinned console launch',Stato='In corso',Descrizione='Register before scientific launch;verify every phase outside runtime before continuation.',**{'Decisioni di origine':[dec]})
    print(json.dumps(dict(staged_records=len(ids),running=a.running)),flush=True)
    change=m.commit_batch()
    if a.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if a.running else 'memory_receipt.json'),dict(valid=True,verification=verification,change=change,records=ids,
        registration_before_scientific_run=not a.running,planned_training_runs=9,airtable_accessed=False,DB_backup='local refresh pending'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41j local memory verified and exported',flush=True)


if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Preview valid; no DB writes')
