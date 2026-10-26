"""Prospective Task41i records; only the authoritative local SQLite mirror."""
import argparse,hashlib,json
from datetime import datetime,timezone
from functools import lru_cache
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');ap.add_argument('--running',action='store_true');a=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41i_protocol_v1'
    spec=json.loads((repo/'experiments/task41i_preregistration_v1.json').read_text(encoding='utf-8'))
    local=json.loads((folder/'local_verification.json').read_text());gpu=json.loads((folder/'gpu_preflight.json').read_text())
    digest=hashlib.sha256((repo/'experiments/task41i_preregistration_v1.json').read_bytes()).hexdigest()
    assert local['valid'] and gpu['valid'] and gpu['all_parameters_finite']
    assert local['protocol_sha256']==gpu['protocol_sha256']==digest
    baseline=json.loads((folder/'memory_receipt.json' if a.running else repo/'experiments/results/task41h_recovery_v1/memory_receipt.json').read_text())['verification']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={};artifacts=[]
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        table=code.split('-')[0]
        rows=m.query(f"SELECT _record_id id FROM [{table}] WHERE [Codice stabile]='{code}'")['rows']
        assert len(rows)==1,code
        return rows[0]['id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41i-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    def artifact(path):
        raw=path.read_bytes();name=path.relative_to(repo).as_posix()
        rid=put('artifacts',name,Nome='Task41i '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        artifacts.append(rid);return rid
    date=datetime.now(timezone.utc).date().isoformat()
    if a.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['all_recovered_tensors_finite']
        for name in ('launch.json','dispatch_request.json','gpu_preflight.json'):artifact(folder/name)
        obs=put('observations','launch',Nome='Task41i actual console launch and off-runtime checkpoint',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':artifacts})
        put('findings','launch',Nome='Task41i finite checkpoint recovered outside runtime',Esito='Positivo',Esperimenti=[ref('experiments-giada-task41i-factorial-v1')],Osservazioni=[obs],Risultato='Pinned worker running; external archive SHA/CRC and finite tensors verified.',Limitazioni='Technical launch only. Scientific result pending; no GO or speedup claim.')
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41i-factorial-v1','Stato':'In corso'})
        for arm in spec['arms']:
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Stato='In corso',**{'Hardware e ambiente':json.dumps(gpu),'Validità tecnica':'Original source-bound worker; finite external checkpoint verified; scientific result pending.'})
        put('actions','launch',Stato='Conclusa',Descrizione='Actual console worker and independent checkpoint recovery verified; scientific evaluation pending.')
    else:
        paths=['experiments/task41i_preregistration_v1.json','notebooks/giada_task41i_selective_repair.ipynb','src/giada_teacher/task41i_selective_repair.py','src/giada_teacher/task41i_evaluation.py','scripts/run_task41i_selective_repair.py','scripts/build_task41i_protocol.py','scripts/task41i_dispatch.py','scripts/task41i_kaggle_pipeline.py','scripts/task41i_gpu_probe.py','scripts/task41i_local_checks.py','scripts/colab_task41i_driver.py','scripts/launch_task41i_console.ps1','scripts/task41b_console_manager.py','scripts/task41b_local_supervisor.py','tests/test_task41i_selective_repair.py','research_memory/register_task41i.py']
        for path in paths:artifact(repo/path)
        prereg=artifacts[0]
        for name in ('local_verification.json','gpu_preflight.json','task41h_frozen.pt'):artifact(folder/name)
        history=[]
        if (folder/'preflight_history.json').exists():
            artifact(folder/'preflight_history.json')
            history=json.loads((folder/'preflight_history.json').read_text())
        exp=put('experiments','factorial',Nome='Task41i selective Ca/Nap equilibrium and crossing supervision',Stato='Preregistrato',Ipotesi=[ref(h['code']) for h in spec['hypothesis_links']],Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Separate causal reachability, crossing supervision and local resolution; distinguish optimization from approximation; measure qualified numerical and whole-chain execution policies.'})
        gen=put('generators','fit',Nome='Task41i paired full-path generator',Versione='v1',Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':artifacts})
        ds=put('datasets','fit-selection',Nome='Task41i disjoint FIT and selection',Versione='v1',Generatore=[gen],Schema='Same causal V,Ca and18 recurrent gates. 1200ms labels; common query union; official event definition retained. No opened-case training.',Copertura=spec['support'])
        parts=[put('partitions',name,Nome='Task41i '+name,Dataset=[ds],Ruolo=role,**{'Regola di assegnazione':json.dumps(spec['new_seeds']),'Restrizioni di accesso':'Historical panels after freeze only; diagnostic, not independent confirmation or candidate selection.'}) for name,role in [('fit','Fit'),('development','Development')]]
        protocol=put('protocols','factorial',Nome='Task41i eight arms and frozen precision/execution sidecars',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),**{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[prereg],'Hash preregistrazione':digest,'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':date})
        impl=put('implementations','worker',Nome='Task41i vectorized CUDA worker',Commit=gpu['revision'],**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41i_selective_repair.py','Artefatti codice':artifacts,'Ambiente':json.dumps(gpu)})
        aa={}
        for arm in spec['arms']:
            model=put('models',arm['name'],Nome='Task41i '+arm['name'],Ruolo='Surrogate',Descrizione=json.dumps(arm),**{'Contratto di stato':'18 gates and0.1ms unchanged; all relaxation rates frozen; only Ca.m/Nap.m equilibrium corrections. No canonical inference fallback.','Contratto di input e output':'Same causal numeric inputs, ionic readout, synapses and membrane equation.'})
            aa[arm['name']]=put('arms',arm['name'],Nome='Task41i '+arm['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(arm)})
            for seed in spec['seeds']:put('runs',arm['name']+'-'+str(seed),Nome=f"Task41i {arm['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[aa[arm['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=arm,checkpoints=spec['checkpoints'],new_seeds=spec['new_seeds']))})
        for axis,left,right in spec['contrasts']:put('contrasts',axis+'-'+left+'-'+right,Nome=f'Task41i {axis}: {left}→{right}',Protocollo=[protocol],Bracci=[aa[left],aa[right]],**{'Contrasto e coefficienti':'Treatment/control per-head severity. Nap factor: Nap only; other factors: Ca and Nap separately. Global conjunction still mandatory.','Soglia interpretativa':'All paired seeds improve and median severity ratio≤0.8; no added global regression. Otherwise mixed/null/negative, not a waiver.','Confondenti controllati':spec['budget']})
        for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['speed']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
            metric=put('metrics',name,Nome='Task41i '+name,Formula=json.dumps(definition));put('evaluations',name,Nome='Task41i '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(definition))
        obs=put('observations','preflight',Nome='Task41i technical CPU and GPU preflight',Descrizione=json.dumps(dict(local=local,gpu=gpu)),**{'Artefatti dettagliati':artifacts})
        if history:
            prior=put('observations','technical-history',Nome='Task41i technical preflight failures before scientific training',Descrizione=json.dumps(history),**{'Artefatti dettagliati':artifacts})
            put('findings','capture-repair',Nome='Task41i capture-unsafe CPU Boolean removed without loss change',Esito='Misto',Esperimenti=[exp],Osservazioni=[prior,obs],Risultato='Initial GPU capture rejected CPU-to-CUDA Boolean construction. Equivalent always-censored tensor expression avoids the transfer; local exact-expression test and final CUDA preflight verify repair.',Limitazioni='Technical preflight only; no scientific training or scientific NO-GO arose from this error.')
        finding=put('findings','preflight',Nome='Task41i causal masks and informative metrics verified',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='Same initial function, protected states, reachable Nap, positive/negative component metric controls, full-path affine identity, finite CUDA graph and auxiliary feedback gradients; whole-chain compile parity tested.',Limitazioni='Technical evidence only; does not establish learnability, qualification or speedup.')
        decision=put('decisions','authorized',Nome='Task41i approved continuation',Esito='Continuare',Risultati=[finding],Motivazione='User approved the eight-arm experiment. Explicit execution priority: Kaggle, Colab primary, Colab reserve. Pinned source and external experiment backups; thresholds unchanged; DB stays local.',**{'Data':date})
        put('actions','launch',Nome='Task41i pinned GPU launch',Stato='In corso',Descrizione='Register and verify protocol before scientific training; Kaggle first, then two Colab accounts. External experiment backup required; DB backup local.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(staged_records=len(ids),running=a.running)),flush=True)
    change=m.commit_batch()
    if a.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if a.running else 'memory_receipt.json'),dict(valid=True,verification=verification,change=change,records=ids,registration_before_scientific_run=not a.running,planned_training_runs=24,airtable_accessed=False,DB_backup='local only; refresh after launch observation'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Task41i local registration verified and exported',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Preview passed; no DB writes',flush=True)
