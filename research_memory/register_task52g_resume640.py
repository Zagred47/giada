import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52g_resume640_v1'
    old=json.loads((repo/'experiments/results/task52g_resume620_v1/preregistration_receipt.json').read_text())
    pre=json.loads((F/'local_preflight.json').read_text());assert pre['valid']
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids=[]
    for key,data in [('interruption',dict(last_log=640,saved_checkpoint=640,no_active_colab_sessions=True,final_report_available=False)),('preflight',pre)]:
        ids.append(m.local_upsert('observations',{'Codice stabile':'observations-giada-task52g-resume640-'+key+'-v1','Nome':'52g Kaggle640 '+key,'Descrizione':json.dumps(data)})['record_id'])
    ids.append(m.local_upsert('decisions',{'Codice stabile':'decisions-giada-task52g-resume640-v1','Nome':'Migrate52g recovery640 to Kaggle','Esito':'Continuare','Motivazione':'User requested Kaggle after repeated Colab disconnections. Resume641-800 from verified640, preserving Adam, RNG, exact diagnostic sums, original anchor and selection checkpoints. No threshold or abstraction changes;52j cap unchanged. Upload only private recovery input; reuse four existing datasets.'})['record_id'])
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid']
    write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));print('Kaggle640 registered and verified',flush=True)
if __name__=='__main__':main()
