import json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
def main():
    repo=ROOT.parent;F=repo/'experiments/results/task52g_resume_v1';old=json.loads((repo/'experiments/results/task52g_protocol_v1/launch_memory_receipt.json').read_text());spec=json.loads((repo/'experiments/task52g_resume_v1.json').read_text());preflight=json.loads((F/'local_preflight.json').read_text());assert preflight['valid']
    m=IncrementalBatchMirror();assert m.baseline_hash==old['verification']['snapshot_hash'];ids=[]
    for key,name,data in [('interruption','52g runtime unavailable after observed740; saved600',dict(last_log=740,saved_checkpoint=600,scientific_no_go=False,final_evaluation_available=False)),('preflight','52g exact continuation and original anchor preflight',preflight),('scope','52g recovery scope',spec)]:
        ids.append(m.local_upsert('observations',{'Codice stabile':'observations-giada-task52g-resume-'+key+'-v1','Nome':name,'Descrizione':json.dumps(data)})['record_id'])
    ids.append(m.local_upsert('decisions',{'Codice stabile':'decisions-giada-task52g-resume-v1','Nome':'Resume52g from600 without changing thresholds','Esito':'Continuare','Motivazione':'User authorized continuation from saved600, restoring Adam and PCG64. Original frozen preservation reference must not move to600. Complete601-800 andoriginalevaluations. Detached manager avoids dependence on foreground SSH lifetime; VM expiry remains possible. Same52j ceiling.'})['record_id'])
    m.commit_batch();v=m.verify_incremental(old['verification']);assert v['valid'];write_json(F/'preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52g resume registered and verified',flush=True)
if __name__=='__main__':main()
