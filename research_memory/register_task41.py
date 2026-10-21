"""Prospective independent Task41, frozen B6 and local-only memory."""
import json,hashlib
from pathlib import Path
from functools import lru_cache
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from src.giada_teacher import task41_independent_confirmation as c
def main():
    repo=ROOT.parent;spec,_,_,_=c.contract(repo);folder=repo/'experiments/results/task41_protocol_v1'
    proof=json.loads((folder/'old_native_phase_preflight.json').read_text());assert proof['valid'] and not proof['fresh_confirmation_accessed']
    m=ValidatedBatchMirror();ids={};arts={}
    @lru_cache(None)
    def ref(code):
        if code in ids:return ids[code]
        rr=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rr)==1,code
        return rr[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41-independent-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    paths=[c.CONFIG,'src/giada_teacher/task41_independent_confirmation.py','src/giada_teacher/task33_observable_synaptic_feedback.py',
        'scripts/run_task41_independent_confirmation.py','scripts/build_task41_protocol.py','scripts/probe_task41_gpu.py',
        'scripts/task41_local_preflight.py','tests/test_task41_independent_confirmation.py','notebooks/giada_task41_independent_confirmation.ipynb',
        'experiments/results/task41_protocol_v1/planned_cases.json','experiments/results/task41_protocol_v1/old_native_phase_preflight.json',
        'scripts/colab_task41_driver.py','scripts/launch_task41_console.ps1','scripts/prepare_task41_dispatch.py',
        'scripts/task39e_console_manager.py','scripts/task39e_local_supervisor.py','research_memory/register_task41.py']
    for name in paths:
        path=repo/name;arts[name]=put('artifacts',name,Nome='Task41 '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
            Versione='v1',Percorso=name,**{'SHA-256':c.p.digest(path),'Dimensione byte':path.stat().st_size})
    claims=[put('claims',str(k+1),Nome=h,Tipo='Ipotesi',Stato='Aperta',Enunciato=h,
        Limiti='New confirmation not yet opened. All original B6 seeds retained; no learning or threshold relaxation.') for k,h in enumerate(spec['hypotheses'])]
    exp=put('experiments','suite',Nome='Task41 independent level1 B6 confirmation',Stato='Preregistrato',Ipotesi=claims,
        Descrizione=spec['design'],**{'Obiettivo informativo':'Test generalization of the frozen ionic operator to new native protocols,seeds,panels andlongconditionedhistories;separate accuracy and executioncost.'})
    generator=put('generators','fresh',Nome='Task41 canonical native and imposed stress generator',Versione='v1',Procedura=json.dumps(dict(design=spec['design'],protocols=spec['protocols'],panels=spec['panels'],phase=spec['native_current_policy'])),
        **{'Seed e RNG':json.dumps(dict(native=spec['native_seeds'],stress=spec['stress_seeds'],calibration=spec['calibration_seed'])),'Artefatti codice':list(arts.values())})
    dataset=put('datasets','fresh',Nome='Task41 prospective sealed native/stress inputs',Versione='v1',Generatore=[generator],
        Schema='Native V,Ca,18states,causal presynaptic conductance,injection,11multipliers;imposed stress controls.',Copertura=spec['design'],
        Fingerprint=c.p.digest(repo/c.CONFIG),Artefatti=[arts[c.CONFIG],arts['experiments/results/task41_protocol_v1/planned_cases.json']],**{'Numero esempi':189})
    parts=[]
    for name,role,rule in [('calibration','Diagnostico','One disjoint seed104003,mechanism/interface only;not fitting'),
        ('native','Test sigillato','176 fresh native cases;open only after candidate freeze and disjoint calibration'),
        ('stress','Test sigillato','12 fresh1200msconditioned stress histories;no selection or training')]:
        parts.append(put('partitions',name,Nome='Task41 '+name,Dataset=[dataset],Ruolo=role,
            **{'Regola di assegnazione':rule,'Restrizioni di accesso':spec['freeze']}))
    protocol=put('protocols','suite',Nome='Task41 independent frozen protocol',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],
        Procedura=json.dumps(spec),**{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[arts[c.CONFIG]],
        'Hash preregistrazione':c.p.digest(repo/c.CONFIG),'Criteri di successo':spec['qualification'],
        'Regole di arresto':'Native/shadow or numerical floor failure:stop B6 judgment;fresh scientific failure:no automatic retry or tuning. External logs preserved.','Registrato il':'2026-10-21'})
    impl=put('implementations','worker',Nome='Task41 isolated native/CUDA console worker',
        **{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41_independent_confirmation.py',
        'Artefatti codice':list(arts.values()),'Ambiente':'Colab primary T4;official console;separate NEURON8.2.7andCUDAprocesses;external PC recovery;DBbackup localonly.'})
    models={'B6':ref('models-giada-task40b-calcium-B6-v1')}
    for name in ('formula','LUT_control'):models[name]=put('models',name,Nome='Task41 '+name,Ruolo='Baseline',Descrizione='Diagnostic numerical/cost baseline only;not alternative candidate. '+str(spec['cost']['baselines']))
    arms={}
    for name in models:
        for mode in ('imposed','autonomous'):
            label=name+'-'+mode;arms[label]=put('arms',label,Nome='Task41 '+label,Ruolo='Trattamento' if name=='B6' else 'Baseline',Protocollo=[protocol],
                Modello=[models[name]],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(dict(candidate=name=='B6',mode=mode,training=False,model_selection=False))})
            for seed in spec['model_seeds']:
                put('runs',label+'-'+str(seed),Nome=f'Task41 {label} seedlabel{seed}',Stato='Pianificata',Seed=str(seed),Braccio=[arms[label]],
                    **{'Configurazione effettiva':json.dumps(dict(mode=mode,model=name,seed=seed,seed_kind='original frozen model seed' if name=='B6' else 'deterministic replica label,notindependenttraining',native_seeds=spec['native_seeds'],cases=176,training=False))})
    for key,value in [('quality',spec['current_quality']),('causal',spec['compartment_gates']),('stress',dict(duration_ms=1200.,quality=spec['current_quality'])),('cost',spec['cost']),('phase',spec['native_current_policy'])]:
        metric=put('metrics',key,Nome='Task41 '+key,Formula=json.dumps(value))
        put('evaluations',key,Nome='Task41 '+key,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,
            Ruolo='Diagnostica' if key=='phase' else ('Secondaria' if key=='cost' else 'Primaria'),Target=json.dumps(value))
    for mode in ('imposed','autonomous'):
        put('contrasts',mode,Nome='Task41 B6 vs formula/LUT '+mode,Protocollo=[protocol],Bracci=[arms[n+'-'+mode] for n in models],
            **{'Contrasto e coefficienti':'Paired same inputs,initial state,parameters and stochastic stream;all original3modelseeds;no posthoc selection.',
            'Soglia interpretativa':spec['qualification'],'Confondenti controllati':spec['phase']+' '+spec['freeze']})
    obs=put('observations','old-native-phase',Nome='Task41 preparation: opened native phase identity',Descrizione=json.dumps(proof,separators=(',',':')),Numerosità=32,Valore=32,
        **{'Artefatti dettagliati':[arts['experiments/results/task41_protocol_v1/old_native_phase_preflight.json']],
        'Specifica di valutazione':[ref('evaluations-giada-task41-independent-phase-v1')]})
    finding=put('findings','old-native-phase',Nome='Align staggered gate state before independent current comparison',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],
        Risultato='Opened32cases:unaligned endpoint reference passes7;staggered reference passes32.NMODL one-step identity atV_next/Ca_next maxerror2.01e-14.NoB6,solverorthresholdchange.',
        Limitazioni='Old opened technical controls only,notTask41confirmation.FutureteacherVCa are never candidateinputsinautonomousmode;derivedcurrentisnotendpoint h.ica.')
    put('lessons','staggered-phase',Nome='Distinguish endpoint samples from split-operator state',Lezione='Verify the actual NMODL update phase before scoring an explicit split operator. Keep unaligned comparisons as diagnostics; do not tune thresholds or silently use future teacher input.',Risultati=[finding],Limiti='Exact native one-step identity established on opened fixtures;fresh floor still required.')
    put('actions','execute',Nome='Execute preregistered independent Task41',Stato='Proposta',Descrizione='Launch after GPU preflight,pinnedpush andverifiedexternalPCrecovery.NoDriveDBupload.',
        **{'Decisioni di origine':[ref('decisions-giada-task40b-result-nominate-B6-for41-v1')]})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,records=ids,verification=v,launched=False,fresh_confirmation_accessed=False,airtable_accessed=False))
    print('Task41 preregistration verified',flush=True)
if __name__=='__main__':main()
