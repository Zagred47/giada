"""Prospective41e semantic registration. SQLite only, before scientific training."""
import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete


def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task41e_protocol_v1'
    spec=json.loads((repo/'experiments/task41e_preregistration_v1.json').read_text())
    checks=json.loads((folder/'local_verification.json').read_text())
    pf=json.loads((folder/'cpu_preflight.json').read_text())
    assert checks['valid'] and pf['valid'] and pf['all_parameters_finite']
    assert checks['protocol_sha256']==hashlib.sha256((repo/'experiments/task41e_preregistration_v1.json').read_bytes()).hexdigest()
    m=ResultPreview() if '--validate-only' in sys.argv else ValidatedBatchMirror()
    ids={};arts={}
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
        assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41e-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=list(spec['sources'])+['experiments/task41e_preregistration_v1.json','scripts/build_task41e_protocol.py',
        'scripts/task41e_local_checks.py','scripts/task41e_gpu_probe.py','scripts/prepare_task41e_dispatch.py',
        'scripts/prepare_task41e_preflight.py','scripts/record_task41e_dispatch.py',
        'scripts/launch_task41e_console.ps1','scripts/colab_task41e_driver.py','scripts/task41b_console_manager.py',
        'scripts/task41b_local_supervisor.py','tests/test_task41e_transfer.py','research_memory/register_task41e.py',
        'experiments/giada_modular_surrogate_roadmap.md','experiments/results/task41e_protocol_v1/source_receipt.json',
        'experiments/results/task41e_protocol_v1/local_verification.json','experiments/results/task41e_protocol_v1/cpu_preflight.json',
        'experiments/results/task41e_protocol_v1/unit_tests.txt']
    for name in dict.fromkeys(paths):
        path=repo/name;content=path.read_bytes()
        arts[name]=put('artifacts',name,Nome='Task41e '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',
                      Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(content).hexdigest(),'Dimensione byte':len(content)})
    claims=[ref(h['code']) for h in spec['hypothesis_links']]
    exp=put('experiments','factorial',Nome='Task41e objective and coupled transfer diagnosis',Stato='Preregistrato',
            Ipotesi=claims,Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Distinguish saturated-target objective,gradient balance,feedback paths/precision,horizon/support,joint correction/tail protection,selection masking and duplicated inference.'})
    findings=[ref('findings-giada-task41d-result-'+key+'-v1') for key in ('objective','feedback','interaction','selection','quality','cost')]
    decision=put('decisions','authorized',Nome='Authorize41e matched transfer contrasts and frozen probes',Esito='Continuare',
                 Motivazione='UserexplicitlyauthorizedoneGPU notebook,allcomparisons,unchangedthresholds,Gitpush,Colabconsolelaunch,externalexperimentbackups,localonlyDBbackup.',
                 Risultati=findings,**{'Data':'2026-10-22'})
    generator=put('generators','fit-history',Nome='Task41e independent physical history and atomic label generator',Versione='v1',
                  Procedura=spec['support']+' '+spec['protection'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':list(arts.values())})
    dataset=put('datasets','fit-development',Nome='Task41e independent FIT,selection and development',Versione='v1',Generatore=[generator],
                Schema='V,Ca,18initialgates,11conductance multipliers,causal externalcurrent/synapticconductances;precomputedcanonical labels andpairedresponses;two64weightcorrections.',
                Copertura=spec['support']+' '+spec['budgets'])
    parts=[]
    for name,role in [('fit','Fit'),('development','Development')]:
        parts.append(put('partitions',name,Nome='Task41e '+name,Dataset=[dataset],Ruolo=role,
                         **{'Regola di assegnazione':'Materialized disjointseedsubstreams;selectionhistories use+100000offset.',
                            'Restrizioni di accesso':'Opened regressions read only after allprimary/secondary selection weights andnomination arefrozen.'}))
    protocol=put('protocols','factorial',Nome='Task41e preregistered shared-control matrices',Versione='v1',Modalità='Ricomposizione',
                 Esperimento=[exp],Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':parts,
                 'Artefatto preregistrazione':[arts['experiments/task41e_preregistration_v1.json']],
                 'Hash preregistrazione':hashlib.sha256((repo/'experiments/task41e_preregistration_v1.json').read_bytes()).hexdigest(),
                 'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':'2026-10-22'})
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    impl=put('implementations','worker',Nome='Task41e36member matched GPU worker and console recovery',Commit=revision,
             **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41e_transfer.py',
                'Artefatti codice':list(arts.values()),'Ambiente':'DeterministicCUDA,TF32disabled;atomicCUDAgraph;perheadclipping,independentAdam;GPUpreflightmandatorybeforedispatch.'})
    arm_ids={};run_ids=[]
    for a in spec['arms']:
        model=put('models',a['name'],Nome='Task41e '+a['name'],Ruolo='Surrogate',Descrizione=json.dumps(a)+' FrozenP110 plus128activeweightsperseed.',
                  **{'Contratto di stato':'Same18occupancies andfixed0.1msupdate;onlyu0/u8correctionslearned.',
                     'Contratto di input e output':'V,Ca,18gates->next18gates;physical currentreadout,calcium,membrane andsynapseinterface unchanged.'})
        arm_ids[a['name']]=put('arms',a['name'],Nome='Task41e '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],
                              Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(a)})
        for seed in spec['seeds']:
            run_ids.append(put('runs',a['name']+'-'+str(seed),Nome=f"Task41e {a['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),
                               Braccio=[arm_ids[a['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=a,seeds=spec['new_seeds'],checkpoints=spec['checkpoints'],parent='P110'))}))
    for name,code in [('P110','models-giada-task41b-local-P110-v1'),('D_C01','models-giada-task41d-C01-v1'),('D_N000','models-giada-task41d-N000-v1')]:
        put('arms','control-'+name,Nome='Task41e frozen '+name,Ruolo='Baseline',Protocollo=[protocol],Modello=[ref(code)],Implementazione=[impl],Partizioni=parts,
            **{'Configurazione residua':'Frozencontrol;notnewtraining;threeoriginalmodelseeds;originalcheckpointprovenance andunchangedthresholds.'})
    for axis,a,b in spec['contrasts']:
        put('contrasts',axis+'-'+a+'-'+b,Nome=f'Task41e {axis} {a}->{b}',Protocollo=[protocol],Bracci=[arm_ids[a],arm_ids[b]],
            **{'Contrasto e coefficienti':'Treatmentminuscontrol atfixedremainingfactors;all3seeds.',
               'Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['training_policy']+' '+spec['budgets']})
    for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),
                            ('cost',spec['cost']),('diagnostics',spec['frozen_probes']),('optimizer',spec['balance_policy'])]:
        metric=put('metrics',name,Nome='Task41e '+name,Formula=json.dumps(definition))
        put('evaluations',name,Nome='Task41e '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,
            Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(definition))
    observation=put('observations','local-preflight',Nome='Task41e actual local verification',Descrizione=json.dumps(dict(checks=checks,preflight=pf)),
                    **{'Artefatti dettagliati':[arts['experiments/results/task41e_protocol_v1/local_verification.json'],arts['experiments/results/task41e_protocol_v1/cpu_preflight.json']]})
    put('findings','local-preflight',Nome='Task41e local verification passed before GPU dispatch',Esito='Positivo',Esperimenti=[exp],Osservazioni=[observation],
        Risultato='Nine tests andfull36member CPUpreflight pass:targetcensoring,physicalfeedback finite-difference gradient,isolatedAdam,exactparentpreservation,fusedcoefficientparity andfiniteauxiliarygradients.',
        Limitazioni='Technical verificationonly;actualGPUpreflightrequiredbeforescientificrun;noquality,speedorTask42claim.')
    put('actions','launch',Nome='Prepare andlaunchTask41e pinned Colab console worker',Stato='In corso',
        Descrizione='Authorizedbyuser;pendingactualGPUpreflightandpinneddispatch. ExternalexperimentPCbackups;DBbackuplocalonly.',
        **{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),runs=len(run_ids))),flush=True)
    m.commit_batch();print('Task41e committed;exporting and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot())
    verification=m.verify();assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=verification,planned_training_runs=36,
                airtable_accessed=False,backup_policy='DBlocalonly',registration_before_scientific_run=True,launched=False))
    print('Task41e prospective registration verified',flush=True)


if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Prospective41e fields,cardinalities andexistingreferences valid;no DBwrites',flush=True)
