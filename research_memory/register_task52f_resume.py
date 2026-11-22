import argparse,json
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();repo=ROOT.parent;old=json.loads((repo/'experiments/results/task52f_protocol_v1/preregistration_receipt.json').read_text());spec=json.loads((repo/'experiments/task52f_resume_v1.json').read_text());m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==old['verification']['snapshot_hash']
    def put(table,key,**fields):
        code=f'{table}-giada-task52f-resume-{key}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    put('observations','failure',Nome='52f original wall-time failure, recoverable600',Descrizione=json.dumps(spec))
    put('observations','preflight',Nome='52f split-continuous resume preflight',Descrizione=(repo/'experiments/results/task52f_resume_v1/local_preflight.json').read_text())
    put('decisions','resume',Nome='Resume52f from600 without restarting',Esito='Continuare',Motivazione='User authorized restoreAdamandweights600, reconstruct deterministic minibatchRNG, preserve saved selectionhistory, run601-800 andoriginalevaluations. No encoderwarmup repeated, no threshold/abstraction changes, no newscientificsubtask. Originalfailurearchive retained. Cap52j remains.')
    m.commit_batch()
    if args.validate_only:return
    v=m.verify_incremental(old['verification']);assert v['valid'];write_json(repo/'experiments/results/task52f_resume_v1/preregistration_receipt.json',dict(valid=True,verification=v,records=ids,airtable_accessed=False));m.export_snapshot_to(ROOT/'data/airtable_snapshot.json');print('52f resume DB verified',flush=True)
if __name__=='__main__':
    try:main()
    except PreviewComplete:print('52f resume schema valid')
