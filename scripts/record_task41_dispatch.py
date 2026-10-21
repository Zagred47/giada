"""Record verified actual console dispatch, not a scientific success claim."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research_memory.mirror import ROOT as MEMORY,write_json
from research_memory.validated_batch import ValidatedBatchMirror
folder=ROOT/'experiments/results/task41_protocol_v1';external=ROOT/'artifacts/task41_primary_external'
status=json.loads((external/'job_status.json').read_text());backup=json.loads((external/'external_backup_receipt.json').read_text())
request=json.loads((folder/'dispatch_request.json').read_text())
assert status['revision']==request['revision'] and backup['valid'] and backup['runtime_independent']
assert hashlib.sha256(Path(backup['local']).read_bytes()).hexdigest()==backup['remote']['sha256']
launch=dict(valid=True,task='41',session='giada-task41-primary',revision=request['revision'],status=status,backup=backup,scientific_result_pending_audit=True)
run_state='Completata' if status['state']=='finished' else 'In corso'
write_json(folder/'launch.json',launch)
m=ValidatedBatchMirror();arts=[]
for name in ('launch.json','dispatch_request.json','gpu_preflight.json','environment_preflight.json'):
    path=folder/name;rel=path.relative_to(ROOT).as_posix()
    arts.append(m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task41-dispatch-'+name+'-v1','Nome':'Task41 '+name,
        'Tipo':'Report','Versione':'v1','Percorso':rel,'SHA-256':hashlib.sha256(path.read_bytes()).hexdigest(),'Dimensione byte':path.stat().st_size})['record_id'])
for mode in ('imposed','autonomous'):
    for model in ('B6','formula','LUT_control'):
        for seed in (17,29,43):
            # Match preregistered codes; no synthetic duplicate runs.
            code=f'runs-giada-task41-independent-{model}-{mode}-{seed}-v1'
            matches=m.query("SELECT record_id FROM v_records WHERE stable_code='"+code+"'")['rows']
            if not matches: raise RuntimeError('Missing preregistered run '+code)
            m.local_upsert('runs',{'Codice stabile':code,'Stato':run_state,'Artefatti prodotti':arts,'Hardware e ambiente':json.dumps(launch)})
m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41-independent-suite-v1','Stato':run_state})
m.local_upsert('actions',{'Codice stabile':'actions-giada-task41-independent-execute-v1','Stato':run_state,'Descrizione':json.dumps(launch)})
m.local_upsert('implementations',{'Codice stabile':'implementations-giada-task41-independent-worker-v1','Commit':request['revision']})
print('Task41 actual dispatch staged',flush=True);m.commit_batch()
write_json(MEMORY/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,airtable_accessed=False,result_pending_audit=True))
