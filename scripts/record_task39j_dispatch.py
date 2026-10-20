"""Record a verified running worker and its first OFF-VM checkpoint."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
from src.giada_teacher import task39j_atomic_relaxation as j

folder=ROOT/'experiments/results/task39j_protocol_v1';external=ROOT/'artifacts/task39j_primary_external'
status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
assert status['state']=='running' and status['phase']=='scientific_worker'
assert backup['valid'] and backup['runtime_independent']
assert hashlib.sha256(Path(backup['local']).read_bytes()).hexdigest()==backup['remote']['sha256']
request=json.loads((folder/'dispatch_request.json').read_text());assert status['revision']==request['revision']
launch=dict(valid=True,task='39j',session='giada-task39j-preflight',profile='primary',revision=request['revision'],status=status,backup=backup,scientific_result_pending=True)
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
for path in [ROOT/'src/giada_teacher/task39j_atomic_relaxation.py',ROOT/'scripts/probe_task39j_gpu.py',ROOT/'scripts/task39e_console_manager.py',
             ROOT/'scripts/task39e_local_supervisor.py',ROOT/'scripts/launch_task39j_console.ps1',ROOT/'scripts/prepare_task39j_dispatch.py',
             ROOT/'scripts/colab_task39j_driver.py',ROOT/'scripts/record_task39j_dispatch.py',folder/'gpu_preflight.json',folder/'launch.json',folder/'dispatch_request.json']:
    rel=path.relative_to(ROOT).as_posix()
    code='artifacts-giada-task39j-atomic-'+rel+'-v1'
    arts.append(m.local_upsert('artifacts',{'Codice stabile':code,'Nome':'Task39j effective '+path.name,'Tipo':'Codice' if path.suffix=='.py' else 'Report',
        'Versione':'launch-v1','Percorso':rel,'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
for a in j.arms():
    for seed in (17,29,43):m.local_upsert('runs',{'Codice stabile':f'runs-giada-task39j-atomic-{a["name"]}-{seed}-v1',
        'Stato':'In corso','Artefatti prodotti':arts,'Configurazione effettiva':json.dumps(dict(arm=a,seed=seed,launch=launch)),
        'Validità tecnica':'GPU smoke and recurrence/selection preflight valid; SHA-verified initial checkpoint outsideVM; result pending.'})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task39j-atomic-suite-v1','Stato':'In corso'})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task39j-atomic-execute-v1','Stato':'In corso','Descrizione':json.dumps(launch)})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task39j-atomic-worker-v1','Commit':request['revision']})
m.local_upsert('observations',{'Codice stabile':'observations-giada-task39j-atomic-preflight-v1','Nome':'39j finite GPU smoke, recurrence and external recovery',
    'Descrizione':json.dumps(dict(gpu=json.loads((folder/'gpu_preflight.json').read_text()),backup=backup)),'Numerosità':24,'Artefatti dettagliati':arts})
print('39j launch staged',flush=True)
m.commit_batch();write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,airtable_accessed=False,scientific_result_pending=True))
print('39j launch registered and verified',flush=True)
