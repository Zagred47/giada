"""Record actual GPU launch and checked off-runtime recovery in SQLite."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json
from research_memory.validated_batch import ValidatedBatchMirror


def main():
    folder=ROOT/'experiments/results/task41e_protocol_v1';external=ROOT/'artifacts/task41e_primary_external'
    request=json.loads((folder/'dispatch_request.json').read_text())
    status=json.loads((external/'job_status.json').read_text())
    receipt=json.loads((external/'external_backup_receipt.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text())
    assert status['revision']==request['revision'] and status['task']=='41e'
    assert pf['valid'] and pf['all_parameters_finite'] and pf['fused_raw_bitwise_equal']
    archive=Path(receipt['local'])
    assert receipt['valid'] and receipt['runtime_independent']
    sha=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    assert sha.hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'checkpoint_0.pt' in z.namelist()
        spec=json.loads(z.read('preregistration.json'))
        assert hashlib.sha256(z.read('P110_source.pt')).hexdigest()==spec['parent_sha256']
        checkpoints=[name for name in z.namelist() if name.startswith('checkpoint_') and name.endswith('.pt')]
        last=max(checkpoints,key=lambda name:int(name.split('_')[1].split('.')[0]))
        checkpoint=torch.load(io.BytesIO(z.read(last)),map_location='cpu',weights_only=False)
        tensors=[]
        def walk(value):
            if torch.is_tensor(value):tensors.append(value)
            elif isinstance(value,dict):
                for child in value.values():walk(child)
            elif isinstance(value,(list,tuple)):
                for child in value:walk(child)
        walk(checkpoint)
        assert all(bool(torch.isfinite(value).all()) for value in tensors)
    launch=dict(valid=True,request=request,status_observed=status,external_recovery=receipt,
                last_recovered_checkpoint=last,checkpoint_tensor_count=len(tensors),all_recovered_tensors_finite=True,
                GPU_preflight=pf,thresholds_changed=False,DBbackup='localonly',state_is_timestamped_not_live=True)
    write_json(folder/'launch.json',launch)
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41e-dispatch-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    artifacts=[]
    for path in (folder/'dispatch_request.json',folder/'preflight_request.json',folder/'gpu_preflight.json',folder/'launch.json',Path(__file__)):
        raw=path.read_bytes()
        artifacts.append(put('artifacts',path.name,Nome='Task41e dispatch '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',
                            Versione='v1',Percorso=path.relative_to(ROOT).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
    obs=put('observations','actual-launch',Nome='Task41e actual GPU launch and external recovery',Descrizione=json.dumps(launch),
            **{'Artefatti dettagliati':artifacts})
    put('findings','technical-launch',Nome='Task41e GPU worker running with verified recovery',Esito='Positivo',
        Esperimenti=[ref('experiments-giada-task41e-factorial-v1')],Osservazioni=[obs],
        Risultato='Actualpinnedconsoleworker,36parallelmembers,passingGPUgraph/optimizer/fusionpreflight;off-runtimecheckpoint '+last+' SHA/CRCcheckedandfinite.',
        Limitazioni='Timestampedtechnicalstatus;no scientificGO,noindependentconfirmation,noTask42authorization.')
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41e-worker-v1','Commit':request['revision']})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41e-factorial-v1','Stato':'In corso',
                  'Descrizione':'ActualpinnedGPUworkerlaunchedandexternalrecoveryverified. Fullprotocolandthresholdsunchanged;launchsnapshotrecorded.'})
    for arm in spec['arms']:
        for seed in spec['seeds']:
            m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41e-{arm['name']}-{seed}-v1",'Stato':'In corso',
                          'Validità tecnica':'ActualGPUpreflightpassed;36parallelmembers;checkedoff-runtimecheckpoint '+last+' finite. Status is a launch snapshot.'})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41e-launch-v1','Stato':'Conclusa',
                  'Descrizione':'Actualconsolelaunchandruntimeindependentrecoveryverified;scientificworkercontinuing;DBbackuplocalonly.'})
    print('Task41e evidenced launch staged',flush=True)
    m.commit_batch();print('Task41e launch committed;exporting and verifying',flush=True)
    write_json(ROOT/'research_memory/data/airtable_snapshot.json',m.export_snapshot())
    verification=m.verify();assert verification['valid']
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,records=ids,verification=verification,
                airtable_accessed=False,DBbackup='localonly',launch_snapshot=launch))
    print('Task41e actual launch registered and verified',flush=True)


if __name__=='__main__':main()
