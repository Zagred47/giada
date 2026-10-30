"""Register stopped41n honestly: canonical floor, not a model NO-GO."""
import argparse,hashlib,json,zipfile
from .mirror import ROOT,write_json
from .incremental_batch import IncrementalBatchMirror
from .register_task41d_result import ResultPreview,PreviewComplete

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    repo=ROOT.parent;folder=repo/'experiments/results/task41n_audited_v1'
    audit=json.loads((folder/'audit.json').read_text());assert audit['valid'] and not audit['model_quality_judged']
    baseline={} if args.validate_only else json.loads((repo/'experiments/results/task41n_protocol_v1/launch_memory_receipt.json').read_text())['verification']
    m=ResultPreview() if args.validate_only else IncrementalBatchMirror();ids={}
    if not args.validate_only:assert m.baseline_hash==baseline['snapshot_hash']
    def put(table,suffix,**fields):
        code=f'{table}-giada-task41n-partial-{suffix}-v1';rid=m.local_upsert(table,{'Codice stabile':code,**fields})['record_id'];ids[code]=rid;return rid
    art=put('artifacts','final',Nome='Task41n stopped confirmation complete external archive',Tipo='Report',Percorso=audit['archive'],Versione='v1',**{'SHA-256':audit['sha256'],'Dimensione byte':audit['bytes']})
    with zipfile.ZipFile(repo/audit['archive']) as z:
        assert z.testzip() is None
        for name,sha in audit['original_reports_sha256'].items():
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==sha
            put('observations','raw-'+name,Nome='Task41n original '+name,Descrizione='Schema preview only' if args.validate_only else raw.decode('utf-8'),**{'Artefatti dettagliati':[art]})
    obs=put('observations','audit',Nome='Task41n canonical floor and exact execution audit',Descrizione=json.dumps(audit),**{'Artefatti dettagliati':[art]})
    exp=m.local_upsert('experiments',{'Codice stabile':'experiments-giada-task41n-confirmation-v1','Stato':'Concluso'})['record_id']
    for name in ('dual','donor_pruned','shared_features','H_control','dual_fp32'):
        m.local_upsert('runs',{'Codice stabile':f'runs-giada-task41n-{name}-v1','Stato':'Interrotta','Artefatti prodotti':[art],'Validità tecnica':'Preflight and imposed evaluation completed; autonomous model/stress/full cost not reached because original FP32 canonical floor failed2/288checks. Not a model quality NO-GO.'})
    put('findings','result',Nome='Task41n inconclusive independent confirmation',Esito='Misto',Esperimenti=[exp],Osservazioni=[obs],Risultato='GPU bitwise preflight passed all3implementations; imposed fresh traces preserve exact outputs. Original FP32 canonical autonomous floor failed2/288checks; matched mixed64 floor passed288/288. Model confirmation withheld.',Limitazioni='No autonomous model judgement,independent stress result or qualified speedup. Keep thresholds unchanged; preserve fresh exposure and do not silently rerun as independent.')
    print('Staged41n partial audit; validating',flush=True);change=m.commit_batch()
    if args.validate_only:return
    verification=m.verify_incremental(baseline);assert verification['valid']
    write_json(folder/'memory_receipt.json',dict(valid=True,verification=verification,change=change,records=ids,airtable_accessed=False,DB_backup='local refresh pending'))
    m.export_snapshot_to(ROOT/'data/airtable_snapshot.json')
    print('Stopped41n fully registered and verified',flush=True)

if __name__=='__main__':
    try:main()
    except PreviewComplete:print('Partial41n schema preview valid;no writes')
