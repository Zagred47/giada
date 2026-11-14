"""Preserve misrouted47 technical failure; no scientific failure attributed."""
import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');a=ap.parse_args();repo=ROOT.parent;folder=repo/'experiments/results/task47_retry_v1'
    old=json.loads((repo/'experiments/results/task47_protocol_v1/preregistration_receipt.json').read_text());e=json.loads((folder/'technical_attempt_v1.json').read_text());assert e['valid'] and not e['scientific_task47_valid']
    m=ResultPreview() if a.validate_only else IncrementalBatchMirror();baseline={} if a.validate_only else old['verification'];ids={}
    if not a.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,key,**fields):
        code=table+'-giada-task47-routing-retry-'+key+'-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    obs=put('observations','routing-failure',Nome='47 original job completed but executed46child protocol',Descrizione=json.dumps(e))
    exp=old['records']['experiments-giada-task47-confirmation-v1']
    put('findings','technical',Nome='47 original attempt invalid as independent confirmation',Esito='Inconcludente',Esperimenti=[exp],Osservazioni=[obs],Risultato='Provider exit0 is not scientific confirmation: actual native/GPU geometry chain4/branch7, not preregistered chain5/branch9. No valid47pass/fail, no new evidence of model regression.',Limitazioni='Original provider archive persisted, SHA from provider receipt; local download/CRC verification pending. No result relabelled as independent success.')
    put('decisions','retry',Nome='Retry same47after child routing correction',Esito='Continuare',Motivazione='User instruction: risolvi. Explicit child47entrypoint plus actual subprocess regression test and fail-closed identity manifests. Same frozenP, inputs, quality thresholds and abstraction; no training/selection/new payload upload.',**{'Data':'2026-11-14'})
    m.commit_batch()
    if a.validate_only:return
    v=m.verify_incremental(baseline);assert v['valid'];write_json(folder/'preregistration_receipt.json',dict(valid=True,same_protocol=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('47 technical retry memory verified/exported',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('47 technical retry schema valid; no writes')
