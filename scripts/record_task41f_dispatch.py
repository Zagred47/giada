"""Record actual41f GPU execution and SHA/CRC-verified external checkpoint."""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json
from research_memory.validated_batch import ValidatedBatchMirror


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',default='reserve',choices=['primary','reserve']);args=parser.parse_args()
    folder=ROOT/'experiments/results/task41f_protocol_v1';external=ROOT/f'artifacts/task41f_{args.profile}_external'
    request=json.loads((folder/'dispatch_request.json').read_text());status=json.loads((external/'job_status.json').read_text())
    pf=json.loads((folder/'gpu_preflight.json').read_text());receipt=json.loads((external/'external_backup_receipt.json').read_text())
    assert pf['valid'] and pf['all_parameters_finite'] and pf['frozen_failure_reproduction_passed']
    assert request['revision']==status['revision']==pf['revision'] and pf['account_profile']==args.profile
    assert receipt['valid'] and receipt['runtime_independent']
    archive=Path(receipt['local']);sha=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    assert sha.hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        checkpoints=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')]
        assert checkpoints,'Actual training checkpoint required; preflight-only archive insufficient'
        name=max(checkpoints,key=lambda n:int(n.split('_')[1].split('.')[0]))
        state=torch.load(io.BytesIO(z.read(name)),map_location='cpu',weights_only=False);tensors=[]
        def walk(value):
            if torch.is_tensor(value):tensors.append(value)
            elif isinstance(value,dict):
                for child in value.values():walk(child)
            elif isinstance(value,(tuple,list)):
                for child in value:walk(child)
        walk(state);assert all(bool(torch.isfinite(v).all()) for v in tensors)
    launch=dict(valid=True,request=request,GPU_preflight=pf,status_observed=status,external_backup=receipt,
        last_recovered_checkpoint=name,all_recovered_tensors_finite=True,thresholds_changed=False,
        scientific_result_pending=True,DB_backup='localonly',status_is_timestamped=True)
    write_json(folder/'launch.json',launch)
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41f-dispatch-{suffix}-v1'
        rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    artifacts=[]
    for path in [folder/n for n in ('dispatch_request.json','gpu_preflight.json','runtime_probe.json','launch.json')]+[Path(__file__)]:
        raw=path.read_bytes();artifacts.append(put('artifacts',path.name,Nome='Task41f dispatch '+path.name,
            Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(ROOT).as_posix(),
            **{'SHA-256':hashlib.sha256(raw).hexdigest(),'Dimensione byte':len(raw)}))
    obs=put('observations','actual-launch',Nome='Task41f actual GPU launch and checked external checkpoint',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':artifacts})
    put('findings','technical-launch',Nome='Task41f GPU training started with recovery verified',Esito='Positivo',
        Esperimenti=[ref('experiments-giada-task41f-factorial-v1')],Osservazioni=[obs],
        Risultato='Pinned36memberGPUworker;passingGPUpreflight,originalfrozenfailurecasesreproduced,full4096graphparity. Actualoff-runtime '+name+' SHA/CRCverified andfinite.',
        Limitazioni='Launchsnapshotonly;scientificresultpending;noTask42authorization;thresholdsunchanged.')
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41f-worker-v1','Commit':request['revision']})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41f-factorial-v1','Stato':'In corso','Descrizione':'ActualT4GPUtrainingwith36members,source-boundconsoleworkerandverifiedexternalcheckpoint;scientificresultpending.'})
    for arm in pf['arms']:
        for seed in (17,29,43):
            m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41f-{arm}-{seed}-v1','Stato':'In corso',
                'Validità tecnica':'GPUpreflightpassed,sourcepinned,actualoff-runtime'+name+' finite. Timestampedlaunchstatus;resultpending.'})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41f-launch-v1','Stato':'Conclusa','Descrizione':'Actualconsolelaunchandexternalcheckpointverified;experimentcontinues.'})
    print('Task41f actuallaunchstaged',flush=True);m.commit_batch()
    write_json(ROOT/'research_memory/data/airtable_snapshot.json',m.export_snapshot());verification=m.verify();assert verification['valid']
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,records=ids,verification=verification,airtable_accessed=False,DBbackup='localonly',launch=launch))
    print('Task41f actualGPUlaunchandbackuprecordedandverified',flush=True)


if __name__=='__main__':main()
