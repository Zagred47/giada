"""Record actual GPU launch and checked off-runtime recovery in SQLite."""
import hashlib
import argparse
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
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['primary','reserve'],default='primary');args=parser.parse_args()
    folder=ROOT/'experiments/results/task41e_protocol_v1';external=ROOT/('artifacts/task41e_'+args.profile+'_external')
    request=json.loads((folder/'dispatch_request.json').read_text())
    status=json.loads((external/'job_status.json').read_text())
    receipt=json.loads((external/'external_backup_receipt.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text())
    assert request['profile']==pf['account_profile']==args.profile
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
    launch['allocation']=json.loads((folder/'allocation.json').read_text())
    amendment=json.loads((folder/'pretraining_amendment.json').read_text())
    assert amendment['valid'] and amendment['unchanged_scientific_contract'] and not amendment['thresholds_changed']
    assert amendment['after_protocol_sha256']==pf['protocol_sha256']
    write_json(folder/'launch.json',launch)
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41e-dispatch-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    artifacts=[];artifact_ids={}
    paths=[folder/name for name in ('dispatch_request.json','preflight_request.json','gpu_preflight.json','gpu_evaluation_smoke.json',
        'allocation.json','launch.json','pretraining_amendment.json','local_verification.json','cpu_preflight.json')]
    paths += [ROOT/'experiments/task41e_preregistration_v1.json',ROOT/'src/giada_teacher/task41e_transfer_evaluation.py',
              ROOT/'scripts/launch_task41e_console.ps1',ROOT/'scripts/task41e_gpu_probe.py',
              ROOT/'scripts/task41e_evaluation_smoke.py',ROOT/'scripts/prepare_task41e_dispatch.py',Path(__file__)]
    for path in paths:
        raw=path.read_bytes()
        artifact=put('artifacts',path.name,Nome='Task41e dispatch '+path.name,Tipo='Codice' if path.suffix in ('.py','.ps1') else 'Report',
                     Versione='v1-technical-fix1',Percorso=path.relative_to(ROOT).as_posix(),**{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)})
        artifacts.append(artifact);artifact_ids[path.name]=artifact
    amendment_observation=put('observations','pretraining-amendment',Nome='Task41e pretraining Jacobian technical amendment',
        Descrizione=json.dumps(amendment),**{'Artefatti dettagliati':[artifact_ids['pretraining_amendment.json'],artifact_ids['task41e_transfer_evaluation.py']]})
    put('findings','pretraining-amendment',Nome='Task41e Jacobian evaluator corrected before training',Esito='Positivo',
        Esperimenti=[ref('experiments-giada-task41e-factorial-v1')],Osservazioni=[amendment_observation],
        Risultato='SyntheticGPUevaluationpreflightcaughtinference-cache/autograderror. IdenticalordinaryconstantsfixinputJacobian;fullreplay,precision,oracleandselectionpathssmoketested. Entireold/newprotocolsaved;scientificcomparisonsandthresholdsequal.',
        Limitazioni='Technicalfixbeforetraining,notascientificmodelimprovement.')
    m.local_upsert('protocols',{'Codice stabile':'protocols-giada-task41e-factorial-v1','Versione':'v1-technical-fix1',
        'Procedura':json.dumps(spec,ensure_ascii=False),'Hash preregistrazione':amendment['after_protocol_sha256'],
        'Artefatto preregistrazione':[artifact_ids['task41e_preregistration_v1.json']]})
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
