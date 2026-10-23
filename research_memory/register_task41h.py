"""Prospective and observed Task41h records, exclusively in the local mirror."""
import argparse,hashlib,json,subprocess
from datetime import datetime
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--running',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41h_protocol_v1'
    spec=json.loads((repo/'experiments/task41h_preregistration_v1.json').read_text())
    local=json.loads((folder/'local_verification.json').read_text());gpu=json.loads((folder/'gpu_preflight.json').read_text())
    digest=hashlib.sha256((repo/'experiments/task41h_preregistration_v1.json').read_bytes()).hexdigest()
    assert local['valid'] and gpu['valid'] and local['protocol_sha256']==gpu['protocol_sha256']==digest
    baseline=json.loads((folder/'memory_receipt.json' if a.running else repo/'experiments/results/task41g_audited_v1/completion_memory_receipt.json').read_text())['verification']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={};artifacts=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        table={'finding':'findings','decision':'decisions'}.get(code.split('-')[0],code.split('-')[0])
        assert table in m.contract.tables
        rows=m.query(f"SELECT _record_id id FROM [{table}] WHERE [Codice stabile]='{code}'")['rows'];assert len(rows)==1,code
        return rows[0]['id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41h-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def artifact(path):
        raw=path.read_bytes();name=path.relative_to(repo).as_posix()
        rid=put('artifacts',name,Nome='Task41h '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        artifacts.append(rid);return rid
    if a.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['all_recovered_tensors_finite']
        for name in ('launch.json','dispatch_request.json','gpu_preflight.json'):artifact(folder/name)
        obs=put('observations','launch',Nome='Task41h actual console worker and external recovery',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':artifacts})
        put('findings','launch',Nome='Task41h launch and finite off-runtime checkpoint verified',Esito='Positivo',Esperimenti=[ref('experiments-giada-task41h-factorial-v1')],Osservazioni=[obs],Risultato='Pinned console worker and SHA/CRC verified checkpoint outside runtime.',Limitazioni='Technical launch only; scientific result pending; thresholds unchanged.')
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41h-factorial-v1','Stato':'In corso'})
        m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41h-worker-v1','Commit':launch['request']['revision']})
        for arm in spec['arms']:
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='In corso',**{'Hardware e ambiente':json.dumps(gpu),'Validità tecnica':'Finite external '+launch['last_recovered_checkpoint']+'; scientific outcome pending.'})
        put('actions','launch',Stato='Conclusa',Descrizione='Actual console launch and independent recovery verified; scientific run continues.')
    else:
        paths=['experiments/task41h_preregistration_v1.json','notebooks/giada_task41h_path_repair.ipynb','src/giada_teacher/task41h_path_repair.py','src/giada_teacher/task41h_evaluation.py','scripts/run_task41h_path_repair.py','scripts/build_task41h_protocol.py','scripts/task41h_dispatch.py','scripts/task41h_gpu_probe.py','scripts/task41h_local_checks.py','scripts/colab_task41h_driver.py','scripts/launch_task41h_console.ps1','scripts/task41b_console_manager.py','scripts/task41b_local_supervisor.py','tests/test_task41h_path_repair.py','research_memory/register_task41h.py']
        for name in paths:artifact(repo/name)
        prereg=artifacts[0]
        for name in ('local_verification.json','gpu_preflight.json','task41g_frozen.pt'):artifact(folder/name)
        exp=put('experiments','factorial',Nome='Task41h history, objective dose, data and capacity with frozen numerical attribution',Stato='Preregistrato',Ipotesi=[ref(h['code']) for h in spec['hypothesis_links']],Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Separate return-history support, auxiliary exposure, sample count and capacity; split voltage-state precision from solver cancellation; measure compact execution without changing quality gates.'})
        gen=put('generators','fit',Nome='Task41h paired atomic and return-history generator',Versione='v1',Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':artifacts})
        ds=put('datasets','fit-selection',Nome='Task41h disjoint training and selection',Versione='v1',Generatore=[gen],Schema='Same causal inputs and18 explicit gate states; M-only equilibrium residual; frozen z and17other states; no endpoint leakage.',Copertura=spec['support'])
        parts=[put('partitions',n,Nome='Task41h '+n,Dataset=[ds],Ruolo=r,**{'Regola di assegnazione':json.dumps(spec['new_seeds']),'Restrizioni di accesso':'Opened historical panels diagnostic only; not selection; no new independent confirmation claimed.'}) for n,r in [('fit','Fit'),('development','Development')]]
        protocol=put('protocols','factorial',Nome='Task41h10 arms,30 members and frozen sidecars',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[prereg],'Hash preregistrazione':digest,'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':datetime.now().date().isoformat()})
        impl=put('implementations','worker',Nome='Task41h vectorized CUDA worker',Commit=gpu['revision'],**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41h_path_repair.py','Artefatti codice':artifacts,'Ambiente':json.dumps(gpu)})
        arms={}
        for arm in spec['arms']:
            model=put('models',arm['name'],Nome='Task41h '+arm['name'],Ruolo='Surrogate',Descrizione=json.dumps(arm),**{'Contratto di stato':'18states,0.1ms; learnedQ0 parent; only M equilibrium corrected; no reduction of abstraction.','Contratto di input e output':'Same V,Ca,18states causal tensor; ionic readout, synapses, membrane unchanged in primary evaluation.'})
            arms[arm['name']]=put('arms',arm['name'],Nome='Task41h '+arm['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(arm)})
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Nome=f"Task41h {arm['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[arms[arm['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=arm,checkpoints=spec['checkpoints'],new_seeds=spec['new_seeds']))})
        for axis,left,right in spec['contrasts']:put('contrasts',axis+'-'+left+'-'+right,Nome=f'Task41h {axis}: {left}->{right}',Protocollo=[protocol],Bracci=[arms[left],arms[right]],**{'Contrasto e coefficienti':'Treatment minus control; other factors fixed; capacity contrast intentional.','Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['budget']})
        for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['speed']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
            metric=put('metrics',name,Nome='Task41h '+name,Formula=json.dumps(definition));put('evaluations',name,Nome='Task41h '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(definition))
        obs=put('observations','preflight',Nome='Task41h CPU and actual T4 preflight',Descrizione=json.dumps(dict(local=local,gpu=gpu)),**{'Artefatti dettagliati':artifacts})
        finding=put('findings','preflight',Nome='Task41h paired contracts and finite GPU gradients verified',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='CPU unit tests, exact affine composition forward/gradient, protected gates and z, actual CUDA graph and auxiliary update, original precision bitwise parity and finite split policies.',Limitazioni='Technical preflight, not scientific training or confirmation.')
        decision=put('decisions','authorized',Nome='Task41h user authorization',Esito='Continuare',Risultati=[finding],Motivazione='User authorized unchanged thresholds, shared notebook, DB registration, push and Colab console with external experimental backups; DB backup local only.',**{'Data':datetime.now().date().isoformat()})
        put('actions','launch',Nome='Task41h console launch',Stato='In corso',Descrizione='Prospective records and actual GPU preflight committed before scientific training.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),running=a.running)),flush=True)
    change=m.commit_batch()
    if a.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if a.running else 'memory_receipt.json'),dict(valid=True,records=ids,verification=verification,change=change,planned_training_runs=30,registration_before_scientific_run=not a.running,airtable_accessed=False,DB_backup='localonly'))
    print('Verified SQLite registration; exporting local readable mirror',flush=True)
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41h local mirror export completed',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Preview passed; no DB writes',flush=True)
