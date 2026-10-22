"""Prospective41g and observed console dispatch, local authoritative mirror."""
import argparse
import hashlib
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');parser.add_argument('--running',action='store_true');args=parser.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41g_protocol_v1'
    spec=json.loads((repo/'experiments/task41g_preregistration_v1.json').read_text())
    local=json.loads((folder/'local_verification.json').read_text())
    assert local['valid'] and local['unit_tests_passed'] and local['CPU_preflight']['valid']
    assert local['protocol_sha256']==hashlib.sha256((repo/'experiments/task41g_preregistration_v1.json').read_bytes()).hexdigest()
    m=ResultPreview() if args.validate_only else ValidatedBatchMirror();ids={};artifacts=[]
    @lru_cache(None)
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41g-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    if args.running:
        launch=json.loads((folder/'launch.json').read_text());assert launch['valid'] and launch['all_recovered_tensors_finite']
        for name in ('launch.json','gpu_preflight.json','dispatch_request.json'):
            path=folder/name;raw=path.read_bytes();artifacts.append(put('artifacts','dispatch-'+name,Nome='Task41g actual '+name,Tipo='Report',Versione='v1',Percorso=path.relative_to(repo).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
        obs=put('observations','actual-dispatch',Nome='Task41g actual GPU run and external checkpoint',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':artifacts})
        put('findings','actual-dispatch',Nome='Task41g pinned GPU execution and backup verified',Esito='Positivo',
            Esperimenti=[ref('experiments-giada-task41g-factorial-v1')],Osservazioni=[obs],
            Risultato='Actual worker started;GPU preflight passed;off-runtime checkpoint SHA/CRC verified and all tensors finite.',Limitazioni='Timestamped launch observation; scientific result pending; thresholds unchanged.')
        m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41g-factorial-v1','Stato':'In corso'})
        m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41g-worker-v1','Commit':launch['request']['revision']})
        for a in spec['arms']:
            for seed in spec['seeds']:m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41g-{a['name']}-{seed}-v1",'Stato':'In corso','Validità tecnica':'Observed GPU worker and finite external '+launch['last_recovered_checkpoint']+';result pending.'})
        m.local_upsert('actions',{'Codice stabile':'actions-giada-task41g-launch-v1','Stato':'Conclusa','Descrizione':'Actual console launch and off-runtime checkpoint verified; scientific run continues.'})
    else:
        paths=list(spec['sources'])+['experiments/task41g_preregistration_v1.json','scripts/build_task41g_protocol.py',
            'scripts/task41g_dispatch.py','scripts/task41g_gpu_probe.py','scripts/task41g_local_checks.py',
            'scripts/launch_task41g_console.ps1','scripts/colab_task41g_driver.py','scripts/task41b_console_manager.py',
            'scripts/task41b_local_supervisor.py','scripts/colab_terminal_download.py','tests/test_task41g_preservation.py',
            'research_memory/register_task41g.py','experiments/results/task41g_protocol_v1/local_verification.json',
            'experiments/results/task41g_protocol_v1/source_receipt.json','experiments/giada_modular_surrogate_roadmap.md']
        prereg=None
        for name in dict.fromkeys(paths):
            path=repo/name;raw=path.read_bytes();a=put('artifacts',name,Nome='Task41g '+name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',Versione='v1',Percorso=name,**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
            artifacts.append(a)
            if name=='experiments/task41g_preregistration_v1.json':prereg=a
        exp=put('experiments','factorial',Nome='Task41g precision and preservation-aware equilibrium repair',Stato='Preregistrato',Ipotesi=[ref(h['code']) for h in spec['hypothesis_links']],Descrizione=spec['training_policy'],**{'Obiettivo informativo':'Distinguish state/readout precision, interfering heads, optimization precision, crossing objective, tail coverage and approximation floor.'})
        generator=put('generators','fit',Nome='Task41g new atomic, held-tail and causal-history generator',Versione='v1',Procedura=spec['support'],**{'Seed e RNG':json.dumps(spec['new_seeds']),'Artefatti codice':artifacts})
        dataset=put('datasets','fit-selection-development',Nome='Task41g disjoint FIT, selection and development',Versione='v1',Generatore=[generator],Schema='Same V,Ca,18occupancies and causal inputs; label-only future currents;64 or192active equilibrium weights, allz frozen.',Copertura=spec['support'])
        parts=[put('partitions',name,Nome='Task41g '+name,Dataset=[dataset],Ruolo=role,**{'Regola di assegnazione':'New explicit seed family641xxx;preservationFIT+200000,tail+19001,projection+29001;physicalselection+100000.','Restrizioni di accesso':'Opened diagnostic panels never train or select new weights.'}) for name,role in [('fit','Fit'),('development','Development')]]
        protocol=put('protocols','factorial',Nome='Task41g12arms and frozen attribution',Versione='v1',Modalità='Ricomposizione',Esperimento=[exp],Procedura=json.dumps(spec,ensure_ascii=False),**{'Partizioni autorizzate':parts,'Artefatto preregistrazione':[prereg],'Hash preregistrazione':local['protocol_sha256'],'Criteri di successo':spec['qualification'],'Regole di arresto':spec['stop_rules'],'Registrato il':datetime.now().date().isoformat()})
        impl=put('implementations','worker',Nome='Task41g36member GPU worker and console',**{'Repository URL':'https://github.com/Zagred47/giada','Percorso entrypoint':'scripts/run_task41g_preservation.py','Artefatti codice':artifacts,'Ambiente':'DeterministicCUDA;TF32off;independentAdam moments and member LR;pointCUDAgraph;frozen precision factorial;external experiment checkpoints,localDBbackup.'})
        aa={}
        for a in spec['arms']:
            model=put('models',a['name'],Nome='Task41g '+a['name'],Ruolo='Surrogate',Descrizione=json.dumps(a),**{'Contratto di stato':'18states,0.1ms;Q0learnedparent;zfixed;onlylearned equilibrium correction.','Contratto di input e output':'V,Ca,18states->18states;currentreadout,membrane,synapses unchanged.'})
            aa[a['name']]=put('arms',a['name'],Nome='Task41g '+a['name'],Ruolo='Trattamento',Protocollo=[protocol],Modello=[model],Implementazione=[impl],Partizioni=parts,**{'Configurazione residua':json.dumps(a)})
            for seed in spec['seeds']:put('runs',a['name']+'-'+str(seed),Nome=f"Task41g {a['name']} seed{seed}",Stato='Pianificata',Seed=str(seed),Braccio=[aa[a['name']]],**{'Configurazione effettiva':json.dumps(dict(arm=a,parent='E_Q0',new_seeds=spec['new_seeds'],checkpoints=spec['checkpoints']))})
        for axis,a,b in spec['contrasts']:put('contrasts',axis+'-'+a+'-'+b,Nome=f'Task41g {axis}: {a}->{b}',Protocollo=[protocol],Bracci=[aa[a],aa[b]],**{'Contrasto e coefficienti':'Treatmentminuscontrol,otheraxesfixed;headcountcontrast64vs192activeweights is intentional,notcapacitymatched.','Soglia interpretativa':spec['contrast_rule'],'Confondenti controllati':spec['budget']})
        for name,definition in [('quality',spec['quality']),('compartment',spec['compartment_gates']),('selection',spec['selection']),('cost',spec['speed']),('diagnostics',spec['diagnostics']),('optimizer',spec['objective'])]:
            metric=put('metrics',name,Nome='Task41g '+name,Formula=json.dumps(definition));put('evaluations',name,Nome='Task41g '+name,Versione='v1',Metrica=[metric],Protocollo=[protocol],Partizioni=parts,Ruolo='Primaria' if name in ('quality','compartment') else 'Secondaria',Target=json.dumps(definition))
        obs=put('observations','local-preflight',Nome='Task41g actual CPU verification',Descrizione=json.dumps(local),**{'Artefatti dettagliati':artifacts})
        finding=put('findings','local-preflight',Nome='Task41g CPU contract and gradients',Esito='Positivo',Esperimenti=[exp],Osservazioni=[obs],Risultato='ZeroQ0parity;maskedheadandzpreservation;compactforward;independentAdamversusPyTorch;finiteheldtailandfeedbackgradients;FP32precision-coreidentity.',Limitazioni='Technical only; GPUpreflight required before scientific run.')
        decision=put('decisions','authorized',Nome='Task41g user-authorized console experiment',Esito='Continuare',Risultati=[finding],Motivazione='User authorized all comparisons,DBregistration,push,Colabconsole,externalexperimentbackup;DBlocalonly.',**{'Data':datetime.now().date().isoformat()})
        put('actions','launch',Nome='Task41g GPUpreflight and console launch',Stato='In corso',Descrizione='Scientific launch pending GPUpreflight and committed prospective registration.',**{'Decisioni di origine':[decision]})
    print(json.dumps(dict(stage='staged',records=len(ids),running=args.running)),flush=True)
    m.commit_batch()
    if not args.running:write_json(folder/'registration_committed.json',dict(valid=True,records=ids,registration_before_scientific_run=True,logical_verification_pending=True,airtable_accessed=False))
    print('Task41g committed; exporting and verifying',flush=True)
    write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/('launch_memory_receipt.json' if args.running else 'memory_receipt.json'),dict(valid=True,records=ids,verification=verification,planned_training_runs=36,registration_before_scientific_run=not args.running,airtable_accessed=False,DB_backup='localonly'))
    print('Task41g verified inSQLite',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Task41g field contracts and links valid;no DB writes',flush=True)
