"""Record actual console launch only after an external checkpoint is verified."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
folder=ROOT/'experiments/results/task41b_protocol_v1';external=ROOT/'artifacts/task41b_primary_external'
request=json.loads((folder/'dispatch_request.json').read_text());status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
assert request['revision']==status['revision'] and backup['valid'] and backup['runtime_independent']
assert not status.get('technical_failure',False),'Technical failure is not a successful launch'
archive=Path(backup['local']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==backup['remote']['sha256']
import zipfile
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert 'checkpoint_0.pt' in z.namelist(),'No actual training checkpoint recovered'
launch=dict(valid=True,launched=True,session='giada-task41b-primary',revision=request['revision'],status=status,external_backup=backup,
    DBbackup='localonly',thresholds_changed=False,result_pending_audit=True)
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
# Refresh this launch recorder's final reviewed bytes relative to its prospective
# pre-dispatch registration. Scientific worker/protocol hashes remain unchanged.
recorder=ROOT/'scripts/record_task41b_dispatch.py'
m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41b-local-scripts/record_task41b_dispatch.py-v1',
    'SHA-256':hashlib.sha256(recorder.read_bytes()).hexdigest(),'Dimensione byte':recorder.stat().st_size})
for path in (folder/'dispatch_request.json',folder/'launch.json'):
    arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41b-dispatch-'+path.name+'-v1','Nome':'Task41b '+path.name,'Tipo':'Report',Versione:'v1',Percorso:path.relative_to(ROOT).as_posix(),
        'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
spec=json.loads((ROOT/'experiments/task41b_preregistration_v1.json').read_text())
for name in ['B6']+[a['name'] for a in spec['arms']]:
    for seed in spec['seeds']:
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41b-local-{name}-{seed}-v1','Stato':'In corso',
            'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(dict(gpu='TeslaT4',session=launch['session'],revision=request['revision'])),
            'Validità tecnica':'Actualconsolejobstarted;firstcheckpointarchive recoveredoutsideVM withSHA/CRCverified. Scientificresultpendingaudit.'})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41b-local-suite-v1','Stato':'In corso'})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41b-local-worker-v1','Commit':request['revision']})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task41b-local-execute-v1','Stato':'Conclusa','Descrizione':'Consolelaunchverified withactualstatusandruntimeindependentcheckpoint. Completionandscientificauditremainpending.'})
print('Task41b actual launch staged',flush=True)
m.commit_batch();write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,launched=True,revision=request['revision'],result_pending_audit=True,airtable_accessed=False))
print('Task41b actual launch registered and verified',flush=True)
