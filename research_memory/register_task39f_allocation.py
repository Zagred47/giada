"""Record unavailable Colab allocation without fabricating failed model runs."""
import hashlib,json
from .mirror import ROOT,write_json
from .validated_batch import ValidatedBatchMirror
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task39f_protocol_v1'
    data=json.loads((folder/'allocation_status.json').read_text());assert not data['training_started']
    m=ValidatedBatchMirror();ids={};arts=[]
    for rel in ('experiments/results/task39f_protocol_v1/allocation_status.json','experiments/results/task39f_protocol_v1/dispatch_request.json','scripts/launch_task39f_console.ps1','research_memory/register_task39f_allocation.py'):
        p=repo/rel;code='artifacts-giada-task39f-console-'+p.name+'-v1'
        rid=m.local_upsert('artifacts',{'Codice stabile':code,'Nome':'Task39f console '+p.name,'Tipo':'Codice' if p.suffix in ('.py','.ps1') else 'Report','Versione':'v1','Percorso':rel,'SHA-256':hashlib.sha256(p.read_bytes()).hexdigest(),'Dimensione byte':p.stat().st_size})['record_id'];ids[code]=rid;arts.append(rid)
    code='analyses-giada-task39f-allocation-v1'
    ids[code]=m.local_upsert('analyses',{'Codice stabile':code,'Nome':'Task39f console allocation unavailable','Origine':'Post hoc','Versione':'v1','Procedura':'whoami;sessions;newT4 x3;usage','Configurazione':json.dumps(data),'Input artefatti':arts})['record_id']
    m.local_upsert('actions',{'Codice stabile':'actions-giada-task39f-temporal-execute-v1','Stato':'Proposta','Descrizione':json.dumps(data)})
    print(json.dumps(dict(stage='staged',records=len(ids))),flush=True)
    m.commit_batch();write_json(ROOT/'data/airtable_snapshot.json',m.export_snapshot());v=m.verify();assert v['valid']
    write_json(folder/'allocation_memory_receipt.json',dict(valid=True,records=ids,verification=v,airtable_accessed=False,training_started=False))
    print(json.dumps(dict(valid=True,records=len(ids))),flush=True)
if __name__=='__main__':main()
