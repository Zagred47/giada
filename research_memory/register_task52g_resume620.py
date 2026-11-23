import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror

def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52g_resume620_v1'
    old=json.loads((repo/'experiments/results/task52g_resume_v1/preregistration_receipt.json').read_text())
    spec=json.loads((repo/'experiments/task52g_resume620_v1.json').read_text())
    preflight=json.loads((F/'local_preflight.json').read_text());assert preflight['valid']
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids=[]
    for key,data in [('interruption',dict(last_log=640,saved_checkpoint=620,runtime_disconnected_in_browser=True,final_report_available=False)),('preflight',preflight),('scope',spec)]:
        ids.append(m.local_upsert('observations',{'Codice stabile':'observations-giada-task52g-resume620-'+key+'-v1','Nome':'52g recovery620 '+key,'Descrizione':json.dumps(data)})['record_id'])
    ids.append(m.local_upsert('decisions',{'Codice stabile':'decisions-giada-task52g-resume620-v1','Nome':'Resume52g from recovery620','Esito':'Continuare','Motivazione':'User authorized resume. Restore model, Adam, RNG and exact diagnostic sums from recovery620. Keep original anchor and original selection checkpoints;620 is not a selection point. No threshold or abstraction changes;52j cap unchanged.'})['record_id'])
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid']
    write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False))
    print('Recovery620 registered and verified')
if __name__=='__main__':main()
