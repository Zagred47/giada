"""Register actual Task40b dispatch and effective tested execution amendment."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
from src.giada_teacher import task40b_calcium_precision as b
folder=ROOT/'experiments/results/task40b_protocol_v1';external=ROOT/'artifacts/task40b_primary_external'
status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
assert status['phase'] in ('scientific_worker','archive','done') and not status.get('technical_failure')
assert backup['valid'] and backup['runtime_independent']
assert hashlib.sha256(Path(backup['local']).read_bytes()).hexdigest()==backup['remote']['sha256']
request=json.loads((folder/'dispatch_request.json').read_text());assert status['revision']==request['revision']
launch=dict(valid=True,task='40b',session='giada-task40b-primary',profile='primary',revision=request['revision'],status=status,backup=backup,scientific_result_pending_audit=True)
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
paths=[folder/'gpu_preflight.json',folder/'local_checks.json',folder/'execution_amendment.json',folder/'launch.json',folder/'dispatch_request.json',
       ROOT/b.CONFIG,ROOT/'notebooks/giada_task40b_calcium_precision.ipynb',ROOT/'tests/test_task40b_calcium_precision.py',
       ROOT/'scripts/prepare_task40b_dispatch.py',ROOT/'scripts/launch_task40b_console.ps1',ROOT/'scripts/record_task40b_dispatch.py',
       ROOT/'scripts/task39e_console_manager.py',ROOT/'scripts/task39e_local_supervisor.py',ROOT/'scripts/run_task40b_calcium_precision.py',ROOT/'src/giada_teacher/task40b_calcium_precision.py']
for path in paths:
    rel=path.relative_to(ROOT).as_posix()
    arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task40b-calcium-'+rel+'-v1','Nome':'Task40b effective '+path.name,'Tipo':'Report',
        'Versione':'launch-v1','Percorso':rel,'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
spec=json.loads((ROOT/b.CONFIG).read_text())
m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task40b-calcium-suite-v1','Versione':'v1-execution-amended',
    'Procedura':json.dumps(spec),'Hash preregistrazione':hashlib.sha256((ROOT/b.CONFIG).read_bytes()).hexdigest(),
    'Regole di arresto':'Nonfinite parameters or failed numerical/optimizer floors stop technical interpretation. Eager vectorized training verified; backward graph capture amendment made before scientific launch.'})
for name in ('B0','B1','B2','B3','B4','B5','B6','B7'):
    for seed in (17,29,43):
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task40b-calcium-{name}-{seed}-v1','Stato':'Completata' if status['state']=='finished' else 'In corso',
            'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(launch),'Validità tecnica':'GPU optimizer and inference parity passed; external immutable recovery verified. Detailed scientific audit pending.'})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task40b-calcium-suite-v1','Stato':'In corso'})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task40b-calcium-execute-v1','Stato':'In corso','Descrizione':json.dumps(launch)})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task40b-calcium-worker-v1','Commit':request['revision'],'Ambiente':spec['execution']})
print('Task40b dispatch staged',flush=True)
m.commit_batch();write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,airtable_accessed=False,result_pending_audit=True))
print('Task40b dispatch registered',flush=True)
