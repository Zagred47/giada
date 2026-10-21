"""Record verified actual console dispatch, not a scientific success claim."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
folder=ROOT/'experiments/results/task41_protocol_v1';external=ROOT/'artifacts/task41_retry_external'
status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
request=json.loads((folder/'retry_dispatch_request.json').read_text())
assert status['revision']==request['revision'] and backup['valid'] and backup['runtime_independent']
assert hashlib.sha256(Path(backup['local']).read_bytes()).hexdigest()==backup['remote']['sha256']
launch=dict(valid=True,task='41',session='giada-task41-primary',revision=request['revision'],status=status,backup=backup,scientific_result_pending_audit=True)
run_state='Pianificata' if status.get('technical_failure') else ('Completata' if status['state']=='finished' else 'In corso')
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
for name in ('launch.json','dispatch_request.json','retry_dispatch_request.json','gpu_preflight.json','environment_preflight.json','launcher_amendment.json','local_checks.json',
             'execution_amendment.json','synaptic_semantics_probe.json','attempt1_failure_report.json','attempt1_native_interface_report.json','attempt1_candidate_contract.json','attempt1_preregistration.json'):
    path=folder/name;rel=path.relative_to(ROOT).as_posix()
    arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41-dispatch-'+name+'-v1','Nome':'Task41 '+name,
        'Tipo':'Report','Versione':'v1','Percorso':rel,'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
for rel in ('scripts/launch_task41_console.ps1','src/giada_teacher/task41_independent_confirmation.py','scripts/run_task41_independent_confirmation.py',
            'experiments/task41_preregistration_v1.json','scripts/build_task41_protocol.py','tests/test_task41_independent_confirmation.py'):
    path=ROOT/rel
    m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41-independent-'+rel+'-v1',
        'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size,'Versione':'v1-technical-interface-fix'})
spec=json.loads((ROOT/'experiments/task41_preregistration_v1.json').read_text())
m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task41-independent-suite-v1','Versione':'v1-technical-interface-amendment',
    'Procedura':json.dumps(spec),'Hash preregistrazione':hashlib.sha256((ROOT/'experiments/task41_preregistration_v1.json').read_bytes()).hexdigest()})
old_status=json.loads((ROOT/'artifacts/task41_primary_external/job_status.json').read_text())
m.local_upsert('runs',{'Codice stabile':'runs-giada-task41-native-attempt1-technical-v1','Nome':'Task41 first native control: omitted fixed-step synaptic age flag',
    'Stato':'Fallita','Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(old_status),
    'Validità tecnica':'Technical stop before B6 evaluation. V/Ca floors pass in all4panels;160/176shadow receptor checks fail. Four opened paired controls identify omitted ems_fixed_step flag, no threshold or model change.'})
for mode in ('imposed','autonomous'):
    for model in ('B6','formula','LUT_control'):
        for seed in (17,29,43):
            # Match preregistered codes; no synthetic duplicate runs.
            code=f'runs-giada-task41-independent-{model}-{mode}-{seed}-v1'
            matches=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
            if not matches: raise RuntimeError('Missing preregistered run '+code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':run_state,'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(launch)})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41-independent-suite-v1','Stato':'In corso','Descrizione':json.dumps(launch)})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task41-independent-execute-v1','Stato':'In corso','Descrizione':json.dumps(launch)})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41-independent-worker-v1','Commit':request['revision']})
print('Task41 actual dispatch staged',flush=True);m.commit_batch()
write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,airtable_accessed=False,result_pending_audit=True))
