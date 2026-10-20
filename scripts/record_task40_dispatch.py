"""Record actual dispatch only after runtime-independent recovery verification."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
from src.giada_teacher import task40_causal_reinsertion as t
folder=ROOT/'experiments/results/task40_protocol_v1';external=ROOT/'artifacts/task40_primary_external'
status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
assert status['phase'] in ('scientific_worker','archive','done') and not status.get('technical_failure')
assert backup['valid'] and backup['runtime_independent']
assert hashlib.sha256(Path(backup['local']).read_bytes()).hexdigest()==backup['remote']['sha256']
request=json.loads((folder/'dispatch_request.json').read_text());assert status['revision']==request['revision']
launch=dict(valid=True,task='40',session='giada-task40-preflight',profile='primary',revision=request['revision'],status=status,backup=backup,scientific_result_pending_audit=True)
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
for path in [folder/'gpu_preflight.json',folder/'local_checks.json',folder/'launch.json',folder/'dispatch_request.json',
             ROOT/'tests/test_task40_causal_reinsertion.py',ROOT/'scripts/prepare_task40_dispatch.py',
             ROOT/'scripts/launch_task40_console.ps1',ROOT/'scripts/record_task40_dispatch.py',
             ROOT/'scripts/task39e_console_manager.py',ROOT/'scripts/task39e_local_supervisor.py',
             ROOT/'scripts/run_task40_causal_reinsertion.py',ROOT/'src/giada_teacher/task40_causal_reinsertion.py']:
    rel=path.relative_to(ROOT).as_posix()
    arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task40-causal-'+rel+'-v1','Nome':'Task40 effective '+path.name,'Tipo':'Report',
        'Versione':'launch-v1','Percorso':rel,'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
for name in ('I3','J1'):
    for family in t.FAMILIES:
        for mode in ('imposed','autonomous'):
            for seed in (17,29,43):
                m.local_upsert('runs',{'Codice stabile':f'runs-giada-task40-causal-{name}-{family}-{mode}-{seed}-v1','Stato':'Completata' if status['state']=='finished' else 'In corso',
                    'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(launch),'Validità tecnica':'GPU graph parity passed; immutable recovery verified outside VM. Scientific audit pending.'})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task40-causal-suite-v1','Stato':'In corso'})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task40-causal-execute-v1','Stato':'In corso','Descrizione':json.dumps(launch)})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task40-causal-worker-v1','Commit':request['revision']})
print('Task40 dispatch staged',flush=True)
m.commit_batch();write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,airtable_accessed=False,result_pending_audit=True))
print('Task40 dispatch registered',flush=True)
