"""Prospective41f: source-bound hypotheses, factorial, 36 runs; SQLite only."""
import hashlib
import json
import sys
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41f_protocol_v1'
    spec=json.loads((repo/'experiments/task41f_preregistration_v1.json').read_text())
    local=json.loads((folder/'local_verification.json').read_text())
    assert local['valid'] and local['unit_tests_passed'] and local['CPU_preflight']['valid']
    assert local['protocol_sha256']==hashlib.sha256((repo/'experiments/task41f_preregistration_v1.json').read_bytes()).hexdigest()
    m=ResultPreview() if '--validate-only' in sys.argv else ValidatedBatchMirror();ids={};artifacts=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41f-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=list(spec['sources'])+['experiments/task41f_preregistration_v1.json','scripts/build_task41f_protocol.py',
        'scripts/task41f_gpu_probe.py','scripts/task41f_runtime_probe.py','scripts/prepare_task41f_dispatch.py',
        'scripts/task41f_local_checks.py','scripts/prepare_task41f_preflight.py','scripts/record_task41f_dispatch.py','scripts/colab_terminal_download.py',
        'scripts/launch_task41f_console.ps1','scripts/colab_task41f_driver.py','scripts/task41b_console_manager.py',
        'scripts/task41b_local_supervisor.py','tests/test_task41f_precision_relaxation.py','research_memory/register_task41f.py',
        'experiments/results/task41f_protocol_v1/local_verification.json','experiments/results/task41f_protocol_v1/source_receipt.json',
        'experiments/giada_modular_surrogate_roadmap.md']
    prereg=None
    for path in dict.fromkeys(paths):
        raw=(repo/path).read_bytes();art=put('artifacts',path,Nome='Task41f '+path,Tipo='Codice' if PathSuffix(path) in ('.py','.ps1') else 'Report',
            Versione='v1',Percorso=path,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        artifacts.append(art)
        if path=='experiments/task41f_preregistration_v1.json':prereg=art
    exp=put('experiments','factorial',Nome='Task41f precision, relaxation and execution transfer',Stato='Preregistrato',
        Ipotesi=[ref(h['code']) for h in spec['hypothesis_links']],Descrizione=spec['training_policy'],
        **{'Obiettivo informativo':'Distinguish numerical response floor, equilibrium versus relaxation, capacity, rare support, objective transfer, boundary predicate and execution granularity.'})
    generator=put('generators','fit',Nome='Task41f independent atomic and physical-history FIT generator',Versione='v1',
        Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':artifacts})
    dataset=put('datasets','fit-selection-development',Nome='Task41f disjoint FIT, selection and development',Versione='v1',Generatore=[generator],
        Schema='V,Ca,18occupancies,conductances,current/synapticcausalinputs,canonicalu/zlabels;learned192/384/384activeweights;noopenedtrajectoriesintraining.',Copertura=spec['support'])
    partitions=[put('partitions',name,Nome='Task41f '+name,Dataset=[dataset],Ruolo=role,
        **{'Regola di assegnazione':'Explicitnewseeds;selectionphysicalhistories+100000offset;atomicselectionindependent.',
           'Restrizioni di accesso':'Frozenopeneddiagnosticsareknownandrecordedbeforetraining;newweightsselectedonlyonnewdevelopmentandselectionlabels.'}) for name,role in [('fit','Fit'),('development','Development')]]
    protocol=put('protocols','factorial',Nome='Task41f preregistered12arm matrix and frozen controls',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],
        Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':partitions,'Artefatto preregistrazione':[prereg],
        'Hash preregistrazione':local['protocol_sha256'],'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':'2026-10-22'})
    impl=put('implementations','worker',Nome='Task41f36member vectorizedGPU worker and console',
        **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41f_precision_relaxation.py',
           'Artefatti codice':artifacts,'Ambiente':'DeterministicCUDA,TF32off;independentAdam;atomicgraph;FP64feedback;primaryFP32deployment;GPUpreflightrequiredbeforelaunch.'})
    arms={}
    for a in spec['arms']:
        model=put('models',a['name'],Nome='Task41f '+a['name'],Ruolo='Surrogate',Descrizione=json.dumps(a),
            **{'Contratto di stato':'Same18recurrentoccupancies,fixed0.1ms;learnedlogitcorrectionsonly;currentreadout/membrane/synapsesunchanged.',
               'Contratto di input e output':'V,Ca,18states->next18states;targetsCa_HVA.m/h,Nap_Et2.m;allotherlearnedcoefficientsfrozen.'})
        arms[a['name']]=put('arms',a['name'],Nome='Task41f '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=partitions,
            **{'Configurazione residua':json.dumps(a)})
        for seed in spec['seeds']:
            put('runs',a['name']+'-'+str(seed),Nome=f"Task41f {a['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[arms[a['name']]],
                **{'Configurazione effettiva':json.dumps(dict(arm=a,new_seeds=spec['new_seeds'],checkpoints=spec['checkpoints'],parent='P110'))})
    for axis,aa,bb in spec['contrasts']:
        put('contrasts',axis+'-'+aa+'-'+bb,Nome=f'Task41f {axis} {aa}->{bb}',Protocollo=[protocol],Bracci=[arms[aa],arms[bb]],
            **{'Contrasto e coefficienti':'Treatmentminuscontrol;otheraxesfixed;U128vsUZ64equalactiveweightbudget.',
               'Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['training_policy']+' '+spec['budget']})
    for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['speed']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
        metric=put('metrics',name,Nome='Task41f '+name,Formula=json.dumps(definition))
        put('evaluations',name,Nome='Task41f '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=partitions,
            Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(definition))
    obs=put('observations','local-preflight',Nome='Task41f actual local verification',Descrizione=json.dumps(local),**{'Artefatti dettagliati':artifacts})
    finding=put('findings','local-preflight',Nome='Task41f CPUpreflight and field verification',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],
        Risultato='ActualCPUpreflightandunittestsvalidatecanonicalboundary,192/384/384capacity,zero-parentparity,unmodifiedcoefficients,compactinference,independentAdam,finitefeedbackgradientsandoracleprecision.',
        Limitazioni='Technicalverificationonly;GPUpreflightmandatory;scientifictrainingnotyetlaunched;thresholdsunchanged.')
    decision=put('decisions','authorized',Nome='AuthorizeTask41f source-bound console launch',Esito='Continuare',Risultati=[finding],
        Motivazione='Userexplicitlyauthorizedallcomparisonsinsamenotebook,SQLite,push,Colabconsoleandexternalexperimentbackups;DBbackuponlylocal.',**{'Data':'2026-10-22'})
    put('actions','launch',Nome='Task41f GPUpreflight and launch',Stato='In corso',Descrizione='Authorized;scientificworkerpendingGPUpreflightanddispatchreceipt.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),planned_runs=36)),flush=True)
    m.commit_batch();print('Task41f committed;exporting and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,planned_training_runs=36,
        registration_before_scientific_run=True,airtable_accessed=False,backup_policy='localonly',launched=False))
    print('Task41f prospectiveDBregistrationverified',flush=True)


def PathSuffix(name):
    from pathlib import Path
    return Path(name).suffix


if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41f field contracts and relations valid;no DBwrites',flush=True)
