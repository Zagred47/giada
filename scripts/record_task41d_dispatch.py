"""Register evidenced console launch and off-runtime recovery; no remote DB."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research_memory.mirror import write_json
from research_memory.validated_batch import ValidatedBatchMirror

def main():
    folder=ROOT/'experiments/results/task41d_protocol_v1';external=ROOT/'artifacts/task41d_primary_external'
    request=json.loads((folder/'dispatch_request.json').read_text());status=json.loads((external/'job_status.json').read_text());receipt=json.loads((external/'external_backup_receipt.json').read_text())
    assert status['revision']==request['revision'] and status['task']=='41d'
    archive=Path(receipt['local']);assert receipt['valid'] and receipt['runtime_independent']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['remote']['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'checkpoint_0.pt' in z.namelist()
        spec=json.loads(z.read('preregistration.json'))
        assert hashlib.sha256(z.read('P110_source.pt')).hexdigest()==json.loads((ROOT/'experiments/task41c_preregistration_v1.json').read_text())['sources']['experiments/results/task41c_protocol_v1/P110_frozen.pt']
        checkpoints=[n for n in z.namelist() if n.startswith('checkpoint_') and n.endswith('.pt')]
        last=max(checkpoints,key=lambda n:int(n.split('_')[1].split('.')[0]));data=torch.load(io.BytesIO(z.read(last)),map_location='cpu',weights_only=False)
        tensors=[]
        def walk(a):
            if torch.is_tensor(a):tensors.append(a)
            elif isinstance(a,dict):
                for v in a.values():walk(v)
            elif isinstance(a,(list,tuple)):
                for v in a:walk(v)
        walk(data);finite=all(bool(torch.isfinite(a).all()) for a in tensors)
        assert finite,'Nonfinite recovery requires technical audit'
    launch=dict(valid=True,request=request,status_observed=status,external_recovery=receipt,last_recovered_checkpoint=last,checkpoint_tensor_count=len(tensors),all_recovered_tensors_finite=finite,thresholds_changed=False,DBbackup='localonly',state_is_timestamped_not_live=True)
    write_json(folder/'launch.json',launch)
    m=ValidatedBatchMirror();ids={}
    def ref(code):
        rows=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows'];assert len(rows)==1,code
        return rows[0]['record_id']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41d-dispatch-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    artifacts=[]
    for path in (folder/'dispatch_request.json',folder/'launch.json',Path(__file__)):
        artifacts.append(put('artifacts',path.name,Nome='Task41d dispatch '+path.name,Tipo='Codice' if path.suffix=='.py' else 'Report',Versione='v1',Percorso=path.relative_to(ROOT).as_posix(),**{'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size}))
    obs=put('observations','actual-launch',Nome='Task41d actual pinned GPU console launch and external recovery',Descrizione=json.dumps(launch),**{'Artefatti dettagliati':artifacts})
    put('findings','technical-launch',Nome='Task41d worker running with verified off-runtime checkpoint',Esito='Positivo',Esperimenti=[ref('experiments-giada-task41d-factorial-v1')],Osservazioni=[obs],Risultato='Pinnedrevision,actualworker,statusandexternalrecoveryverified. Lastrecoveredcheckpoint'+last+'finite;notascientificGO.',Limitazioni='Timestampedlaunchsnapshot;laterstatusmustbereadfromjob_statusandfinalarchive. NoindependentconfirmationorTask42permission.')
    m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41d-worker-v1','Commit':request['revision']})
    m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41d-factorial-v1','Stato':'In corso','Descrizione':'ActualconsoleGPUworkerlaunchedandexternalrecoveriesverified;protocolandthresholdsunchanged. Launchstatusistimestampedinobservations.'})
    for a in spec['arms']:
        for seed in spec['seeds']:m.local_upsert('runs',{'Codice stabile':f"runs-giada-task41d-{a['name']}-{seed}-v1",'Stato':'In corso','Validità tecnica':'PinnedGPUworker;36parallelmembers;externalcheckpointSHA/CRCverified;finitetensorsat'+last})
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task41d-launch-v1','Stato':'Conclusa','Descrizione':'Launchverified;userapprovedexternalPCbackupsrunning;DBbackupremainslocal. Seeimmutablelaunchreceipt.'})
    print('Task41d evidenced launch staged',flush=True)
    m.commit_batch();write_json(ROOT/'research_memory/data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'launch_memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,DBbackup='localonly',launch_snapshot=launch))
    print('Task41d launch registered and verified',flush=True)
if __name__=='__main__':main()
