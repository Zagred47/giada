"""Technical routing retry, not a new scientific protocol or candidate."""
import argparse,json,time,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',action='store_true');ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task46_retry_v1';folder.mkdir(exist_ok=True)
    if not a.result and not a.validate_only:
        for _ in range(180):
            status=repo/'experiments/results/task46_protocol_v1/workflow.json'
            if status.exists() and json.loads(status.read_text()).get('state')=='complete':break
            time.sleep(5)
        else:raise RuntimeError('Original46DB/backup audit pending; do not start concurrent memory writer')
    prev=folder/'memory_receipt.json' if a.result else repo/'experiments/results/task46_protocol_v1/result_memory_receipt.json'
    baseline={} if a.validate_only else json.loads(prev.read_text())['verification'];m=ResultPreview() if a.validate_only else IncrementalBatchMirror();ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task46-cache-retry-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    if not a.result:
        put('observations','authorized',Nome='46 same-protocol source-cache isolation retry',Descrizione=json.dumps(dict(same_protocol=True,original_failure='Preflight andGPU both called exclusive verify_source(cache=verified_source46); mainGPU stopped before first coupled evaluation. Native references completed and preserved in original archive.',fix='Preflight uses sibling preflight46/models and its own source cache; mainGPU uses original result workspace. No weight/data/equation/dt/threshold/selection/training change.',regression='Two real same-weight loads with exclusive caches now succeed in separate folders; old repeated destination still raises FileExistsError.',authorization='Normal technical implementation fix within user task46 instruction; no new payload upload or candidate change.')))
    else:
        r=json.loads((repo/'artifacts/task46_retry_external/external_backup_receipt.json').read_text());assert r['valid']
        art=put('artifacts','output',Nome='46 isolated-cache retry output',Tipo='Report',Versione='v1',Percorso=r['local'],**{'SHA-256':r['remote']['sha256'],'Dimensione byte':r['remote']['bytes']});obs=[]
        with zipfile.ZipFile(r['local']) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('.json'):obs.append(put('observations','raw-'+name,Nome='46 retry '+name,Descrizione=z.read(name).decode(),**{'Artefatti dettagliati':[art]}))
        obs.append(put('observations','comparison',Nome='46 retry paired intervention, feedback and cost audit',Descrizione=(folder/'comparison.json').read_text(),**{'Artefatti dettagliati':[art]}))
        exp=json.loads((repo/'experiments/results/task46_protocol_v1/preregistration_receipt.json').read_text())['records']['experiments-giada-task46-spatial-cost-v1']
        put('findings','result',Nome='46 same-protocol isolated-cache outcome',Esito='Misto',Esperimenti=[exp],Osservazioni=obs,Risultato=json.dumps(r['report']),Limitazioni='Original technical failure preserved. FrozenP, original45e/41limits and thresholds unchanged. Completion not release/47authorization or speedup claim.')
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/('result_memory_receipt.json' if a.result else 'memory_receipt.json'),dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('46retry memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('46retry schema valid; no writes')
