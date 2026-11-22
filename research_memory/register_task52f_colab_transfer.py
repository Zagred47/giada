"""Record provider recovery separately from unchanged scientific protocol."""
import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror

def main():
    repo=ROOT.parent;folder=repo/'experiments/results/task52f_resume_colab_v1'
    old=json.loads((repo/'experiments/results/task52f_resume_v1/preregistration_receipt.json').read_text())
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash']
    ids=[]
    for key,title,path in (
        ('cancelled','52f Kaggle retry cancelled before any execution',repo/'experiments/results/task52f_resume_retry_v1/cancelled_before_colab.json'),
        ('transfer','52f exact600 continuation on Colab',folder/'dispatch_request.json'),
        ('launch','52f Colab console launch and verified inputs',folder/'launch_receipt.json')):
        data=json.loads(path.read_text())
        ids.append(m.local_upsert('observations',{'Codice stabile':f'observations-giada-task52f-colab-{key}-v1','Nome':title,'Descrizione':json.dumps(data)})['record_id'])
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False,db_backup_local=True))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('52f provider recovery registered and verified')

if __name__=='__main__':main()
