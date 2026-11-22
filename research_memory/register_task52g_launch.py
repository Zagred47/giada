"""Record an observed52g worker, not an inferred scientific result."""
import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task52g_protocol_v1';old=json.loads((folder/'preregistration_receipt.json').read_text());launch=json.loads((folder/'launch_receipt.json').read_text());state=json.loads((folder/'remote_status.json').read_text());inputs=json.loads((folder/'input_verified.json').read_text())
    assert state['state']=='running' and state['revision']==launch['revision'] and inputs['valid']
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids=[]
    art=m.local_upsert('artifacts',{'Codice stabile':'artifacts-giada-task52g-launch-v1','Nome':'52g verified console launch','Tipo':'Log','Versione':'v1','Percorso':(folder/'launch_receipt.json').relative_to(repo).as_posix()})['record_id']
    for key,rid in old['records'].items():
        if key.startswith('runs-'):
            m.local_upsert('runs',{'Stato':'In corso','Artefatti prodotti':[art],'Configurazione effettiva':json.dumps(dict(launch=launch,state=state)),'Validità tecnica':'Worker observed running; scientific outcome pending.'},record_id=rid);ids.append(rid)
    m.local_upsert('observations',{'Codice stabile':'observations-giada-task52g-launch-v1','Nome':'52g runtime andbackup transport verified','Descrizione':json.dumps(dict(launch=launch,state=state,inputs=inputs)),'Artefatti dettagliati':[art]})
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid'];write_json(folder/'launch_memory_receipt.json',dict(valid=True,verification=v,runs=ids,scientific_result_claimed=False,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52g running state verified in DB',flush=True)
if __name__=='__main__':main()
