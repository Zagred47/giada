"""Prospective Task41b 2x2x2 experiment, SQLite only."""
import hashlib,json
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41b_protocol_v1'
    spec=json.loads((repo/'experiments/task41b_preregistration_v1.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text());ep=json.loads((folder/'evaluation_preflight.json').read_text())
    assert pf['valid'] and pf['full_parameters_finite'] and ep['valid'] and not ep['training_performed']
    assert pf['source_sha256']==hashlib.sha256((repo/'src/giada_teacher/task41b_local_precision.py').read_bytes()).hexdigest()
    assert pf['protocol_sha256']==hashlib.sha256((repo/'experiments/task41b_preregistration_v1.json').read_bytes()).hexdigest()
    m=ValidatedBatchMirror();ids={};arts={}
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41b-local-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=['experiments/task41b_preregistration_v1.json','src/giada_teacher/task41b_local_precision.py','src/giada_teacher/task41b_evaluation.py',
        'scripts/run_task41b_local_precision.py','scripts/build_task41b_protocol.py','scripts/task41b_gpu_probe.py','scripts/task41b_evaluation_probe.py',
        'scripts/prepare_task41b_preflight.py','scripts/task41b_console_manager.py','scripts/task41b_local_supervisor.py',
        'scripts/launch_task41b_console.ps1','scripts/colab_task41b_driver.py','scripts/prepare_task41b_dispatch.py',
        'scripts/record_task41b_dispatch.py','tests/test_task41b_local_precision.py','notebooks/giada_task41b_local_precision.ipynb',
        'experiments/giada_modular_surrogate_roadmap.md','research_memory/register_task41b.py']
    paths += ['experiments/results/task41b_protocol_v1/'+n for n in ('gpu_preflight.json','evaluation_preflight.json','opened_inputs_manifest.json')]
    for name in paths:
        path=repo/name;arts[name]=put('artifacts',name,Nome='Task41b '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',
            Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})
    claims=[ref('claims-giada-task41-audit-'+h['id']+'-v1') for h in spec['hypotheses']]
    exp=put('experiments','suite',Nome='Task41b local ionic precision: support x resolution x objective',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['training'],**{'Obiettivo informativo':'Separate FIT coverage, learned local resolution, objective conditioning and optimization budget for three localized functions; preserve all original quality gates.'})
    gen=put('generators','atomic',Nome='Task41b independent atomic and conditioned current generator',Versione='v1',Procedura=spec['data'],
        **{'Seed e RNG':json.dumps({k:spec[k] for k in ('fit_seed','selection_seed','development_seed','stream_seed')}),'Artefatti codice':list(arts.values())})
    dataset=put('datasets','new',Nome='Task41b new FIT and development',Versione='v1',Generatore=[gen],Schema='V,Ca,initial18gates,offlineu/z,3held/pulsestages;8new1200msdevelopmenthistories',
        Copertura=spec['data'],Fingerprint=hashlib.sha256((repo/'experiments/task41b_preregistration_v1.json').read_bytes()).hexdigest(),Artefatti=list(arts.values()),**{'Numero esempi':8456})
    partitions=[]
    for name,role,rule in [('fit_uniform','Fit','4096 independent uniform anchors;fit_seed141101'),('fit_sensitive','Fit','4096 canonical FIT-sensitivity-stratified anchors;samecountandpilot'),
        ('atomic_selection','Development','256 independent anchors;selection_seed141103'),('development','Development','8new1200ms histories;development_seed141107;all3frozenparentseeds'),
        ('opened_regression','Diagnostico','Allopened39j/40b/41panels afterfreeze;nevercheckpointselectionorfreshconfirmation')]:
        partitions.append(put('partitions',name,Nome='Task41b '+name,Dataset=[dataset],Ruolo=role,
            **{'Regola di assegnazione':rule,'Restrizioni di accesso':spec['selection']+' '+spec['regression']}))
    protocol=put('protocols','suite',Nome='Task41b paired local kinetic repair',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec),
        **{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[arts['experiments/task41b_preregistration_v1.json']],
            'Hash preregistrazione':hashlib.sha256((repo/'experiments/task41b_preregistration_v1.json').read_bytes()).hexdigest(),
            'Criteri di successo':spec['qualification'],'Regole di arresto':'Failed numeric/equivalence/frozenbaseline floor or nonfinite parameters stops scientific interpretation. Preserve checkpoint and technical report. No automatic scientific restart or threshold change.','Registrato il':'2026-10-21'})
    impl=put('implementations','worker',Nome='Task41b 24 independent vectorized members',
        **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41b_local_precision.py','Artefatti codice':list(arts.values()),
            'Ambiente':'Colab primary TeslaT4,PyTorch2.11.0+cu130;console,externalPCrecovery;DBbackup localonly. '+spec['execution']})
    armids={}
    for a in [dict(name='B6',frozen=True)]+spec['arms']:
        name=a['name'];model=ref('models-giada-task40b-calcium-B6-v1') if name=='B6' else put('models',name,Nome='Task41b '+name,Ruolo='Surrogate',
            Descrizione=spec['training']+' '+json.dumps(a))
        armids[name]=put('arms',name,Nome='Task41b '+name,Ruolo='Baseline' if name=='B6' else 'Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,
            **{'Configurazione residua':json.dumps(a)})
        for seed in spec['seeds']:
            put('runs',name+'-'+str(seed),Nome=f'Task41b {name} seed{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[armids[name]],
                **{'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,checkpoints=spec['checkpoints'],training=name!='B6',parent_frozen=True))})
    for axis in ('support','basis','conditioned'):
        for a in spec['arms']:
            for b in spec['arms']:
                others=set(('support','basis','conditioned'))-{axis}
                if all(a[k]==b[k] for k in others) and ((axis=='support' and a[axis]=='uniform' and b[axis]=='sensitive') or (axis=='basis' and a[axis]==32 and b[axis]==64) or (axis=='conditioned' and not a[axis] and b[axis])):
                    put('contrasts',axis+'-'+a['name']+'-'+b['name'],Nome='Task41b '+axis+' '+a['name']+'→'+b['name'],Protocollo=[protocol],Bracci=[armids[a['name']],armids[b['name']]],
                        **{'Contrasto e coefficienti':json.dumps(dict(control=a,treatment=b)),'Soglia interpretativa':spec['contrasts'],'Confondenti controllati':spec['training']+' '+spec['selection']})
    for name,value in [('quality',spec['quality']),('selection',spec['selection']),('cost',spec['cost']),('causal',spec['qualification'])]:
        metric=put('metrics',name,Nome='Task41b '+name,Formula=json.dumps(value))
        put('evaluations',name,Nome='Task41b '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions,Ruolo='Diagnostica',Target=json.dumps(value))
    for name,value in [('gpu',pf),('evaluation',ep)]:
        obs=put('observations','preflight-'+name,Nome='Task41b '+name+' preflight',Descrizione=json.dumps(value),Numerosità=24,
            **{'Artefatti dettagliati':[arts['experiments/results/task41b_protocol_v1/'+('gpu_preflight.json' if name=='gpu' else 'evaluation_preflight.json')]]})
        put('findings','preflight-'+name,Nome='Task41b technical '+name+' controls passed',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],
            Risultato='ActualT4preflight passed;independentAdam24members,eager/graphparity,frozenB6reproduced,newdevelopmentfloorandshadowCUDAqualificationchecked.',
            Limitazioni='Technical preflight,not scientific training or qualification. No Task42 authorization.')
    put('actions','execute',Nome='Execute Task41b after pinned push',Stato='Proposta',Descrizione='Authorized byuser21October;actualdispatchreceiptmustfollowlaunch;externalcheckpointsPCandSQLitebackup local.',
        **{'Decisioni di origine':[ref('decisions-giada-task41-audit-no-go-next-diagnosis-v1')]})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41-audit-propose41b-v1','Stato':'Conclusa','Descrizione':'User approved Task41b implementation and consolelaunch21October2026;formalpreregistration created withunchangedthresholds.'})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();print('Task41b preregistration committed;exporting snapshot',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,launched=False,airtable_accessed=False,DBbackup='localonly'))
    print('Task41b preregistration verified',flush=True)

if __name__=='__main__':main()
